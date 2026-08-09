"""
Client for the National Weather Service (NWS) API (api.weather.gov).

No API key required - the only requirement is a descriptive `User-Agent`
header (requests without one get 403). We harvest three kinds of free-text
weather documents per location and normalize them into the weather_documents
document schema used by the rest of the pipeline:

  - Active alerts        GET /alerts/active?point={lat},{lon}
                         narrative `description` + `instruction` fields
                         (e.g. "* WHAT...dangerous rip currents...").
  - Daily forecast       GET /gridpoints/{office}/{x},{y}/forecast
                         one document per period, body = `detailedForecast`
                         narrative (e.g. "Showers and thunderstorms likely.
                         Mostly cloudy, with a low around 72...").
  - Hourly forecast      GET /gridpoints/{office}/{x},{y}/forecast/hourly
                         one document per hour, body templated from
                         `shortForecast` + temperature + wind speed.

Locations are accepted in two forms:
  - City/state strings, e.g. "Chicago, IL"   -> geocoded to lat/lon via the
    free, key-less Open-Meteo Geocoding API (geocoding-api.open-meteo.com).
  - "lat,lon" strings,  e.g. "41.8781,-87.6298" -> used directly.

The client mirrors massive_client.py's shape (a thin requests wrapper with a
`get()` helper plus focused fetch methods), but swaps the Bearer auth header
for the User-Agent that api.weather.gov mandates.
"""

import hashlib
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

# Free, no-key geocoding service used to resolve "City, ST" -> lat/lon.
GEOCODING_BASE_URL = os.environ.get(
    "GEOCODING_API_BASE_URL", "https://geocoding-api.open-meteo.com/v1/search"
)
_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")

_DEFAULT_TIMEOUT = 30
# NWS asks clients to be polite; we sleep between each of our own calls to
# stay well within their generous rate limits when looping over locations.
_DEFAULT_REQUEST_DELAY = 0.25
_DEFAULT_USER_AGENT = (
    "databricks-lakebase-weather-app/1.0 "
    "(homework app; contact: dev@example.com)"
)


class WeatherClient:
    """Thin wrapper around api.weather.gov with geocoding + normalization."""

    def __init__(
        self,
        base_url: str | None = None,
        user_agent: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        request_delay: float = _DEFAULT_REQUEST_DELAY,
    ):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self.request_delay = request_delay
        # Every location that could not be resolved/geocoded, so the caller
        # can report it instead of failing the whole batch.
        self.unresolved_locations: list[str] = []
        self._session = requests.Session()
        self._session.headers.update(
            {
                # api.weather.gov REQUIRES a descriptive User-Agent.
                "User-Agent": user_agent or _DEFAULT_USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        time.sleep(self.request_delay)
        resp = self._session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Location resolution
    # ------------------------------------------------------------------
    def geocode(self, query: str) -> dict | None:
        """Resolve a "City, ST" string to lat/lon via the free Open-Meteo
        geocoding API. Returns {"lat","lon","city","state"} or None."""
        resp = requests.get(
            GEOCODING_BASE_URL,
            params={"name": query, "count": 1, "language": "en", "format": "json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return None
        r = results[0]
        return {
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"]),
            "city": r.get("name") or query,
            "state": r.get("admin1") or "",
        }

    def resolve_location(self, location: str) -> dict:
        """Normalize a user-supplied location string into a dict with lat,
        lon, and a human-readable label. Raises ValueError if unusable."""
        s = (location or "").strip()
        if not s:
            raise ValueError("empty location")

        # "lat,lon" form, used directly (no geocoding needed).
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2 and _is_float(parts[0]) and _is_float(parts[1]):
            return {
                "lat": float(parts[0]),
                "lon": float(parts[1]),
                "label": s,
            }

        # City/state form -> geocode via Open-Meteo.
        geo = self.geocode(s)
        if not geo:
            self.unresolved_locations.append(s)
            raise ValueError(f"could not geocode location {s!r}")
        label = f"{geo['city']}, {geo['state']}".strip(", ")
        return {"lat": geo["lat"], "lon": geo["lon"], "label": label}

    # ------------------------------------------------------------------
    # NWS fetch methods
    # ------------------------------------------------------------------
    def gridpoint(self, lat: float, lon: float) -> dict:
        """GET /points/{lat},{lon} -> the local forecast office + grid coords."""
        data = self.get(f"/points/{lat},{lon}")
        props = data["properties"]
        return {
            "office": props["gridId"],
            "x": props["gridX"],
            "y": props["gridY"],
        }

    def fetch_alerts(self, lat: float, lon: float) -> list[dict]:
        """GET /alerts/active?point=... -> list of raw alert Feature dicts."""
        data = self.get("/alerts/active", params={"point": f"{lat},{lon}"})
        return data.get("features") or []

    def fetch_forecast(self, grid: dict) -> list[dict]:
        """GET /gridpoints/{office}/{x},{y}/forecast -> list of day/night periods."""
        path = f"/gridpoints/{grid['office']}/{grid['x']},{grid['y']}/forecast"
        data = self.get(path)
        return data["properties"].get("periods") or []

    def fetch_hourly_forecast(self, grid: dict) -> list[dict]:
        """GET /gridpoints/{office}/{x},{y}/forecast/hourly -> list of hourly periods."""
        path = f"/gridpoints/{grid['office']}/{grid['x']},{grid['y']}/forecast/hourly"
        data = self.get(path)
        return data["properties"].get("periods") or []

    # ------------------------------------------------------------------
    # Normalization -> weather_documents rows
    # ------------------------------------------------------------------
    def _normalize_alert(
        self, feature: dict, location_label: str, synced_at: str
    ) -> dict | None:
        """Turn one active-alert Feature into a document row.

        id:  the NWS alert id (a stable urn:oid:...).
        narrative_text: headline + description + instruction a la
        "* WHAT... ...\n\n* IMPACTS...".
        """
        p = feature.get("properties") or {}
        alert_id = p.get("id")
        if not alert_id:
            return None

        narrative = "\n\n".join(
            x for x in (p.get("description"), p.get("instruction")) if x
        )
        headline = p.get("headline") or p.get("event")
        if headline and narrative:
            narrative = f"{headline}\n\n{narrative}"
        narrative = narrative or headline or ""

        return {
            "id": str(alert_id),
            "location": location_label,
            "source_type": "alert",
            "headline": p.get("event") or p.get("headline"),
            "narrative_text": narrative,
            "effective_at": p.get("effective") or p.get("sent"),
            "payload": p,
            "synced_at": synced_at,
        }

    def _normalize_forecast(
        self,
        grid: dict,
        period: dict,
        source_type: str,
        location_label: str,
        synced_at: str,
    ) -> dict | None:
        """Normalize one forecast period into a document row.

        Daily periods embed the rich `detailedForecast` narrative; hourly
        periods usually only have a `shortForecast`, so we template a short
        narrative around it (temperature + wind) to keep them embeddable.

        id: a stable, deterministic key derived from the grid point + period
        start time (e.g. nws:LOT:76,73:forecast:2026-08-09T18:00:00-05:00),
        so re-syncs upsert instead of duplicating.
        """
        start_time = (period.get("startTime") or "").strip()
        if not start_time:
            return None

        headline = (period.get("name") or "").strip() or None

        if source_type == "forecast":
            narrative = (period.get("detailedForecast") or "").strip()
            if not narrative and headline:
                narrative = headline
            if not narrative:
                return None
        else:  # hourly forecast
            short = (period.get("shortForecast") or "").strip()
            if not short:
                return None
            parts = [short]
            temp = period.get("temperature")
            if temp is not None:
                parts.append(f"temperature around {temp}\u00b0F")
            wind = period.get("windSpeed")
            if wind:
                parts.append(f"wind {wind}")
            narrative = ". ".join(
                f"{p}." if not p.endswith(".") else p for p in parts
            )

        return {
            "id": f"nws:{grid['office']}:{grid['x']},{grid['y']}:{source_type}:{start_time}",
            "location": location_label,
            "source_type": source_type,
            "headline": headline,
            "narrative_text": narrative,
            "effective_at": start_time,
            "payload": period,
            "synced_at": synced_at,
        }

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def sync(
        self,
        locations: list[str],
        limit: int | None = None,
        include_alerts: bool = True,
        include_forecast: bool = True,
        include_hourly: bool = True,
    ) -> list[dict]:
        """Fetch + normalize weather documents for a list of locations.

        limit: max documents collected per location (alerts + forecast +
        hourly combined), mirroring the news pipeline's per-ticker limit.

        Returns a flat list of document rows ready for upsert into
        weather_documents. Unresolvable locations are recorded on
        self.unresolved_locations and skipped rather than failing the batch.
        """
        synced_at = datetime.now(timezone.utc).isoformat()
        all_docs: list[dict] = []

        for raw_loc in locations:
            try:
                resolved = self.resolve_location(raw_loc)
            except ValueError:
                continue  # already recorded on self.unresolved_locations

            try:
                grid = self.gridpoint(resolved["lat"], resolved["lon"])
            except Exception:
                self.unresolved_locations.append(raw_loc)
                continue

            label = resolved["label"]
            docs: list[dict] = []

            if include_alerts:
                try:
                    for feature in self.fetch_alerts(resolved["lat"], resolved["lon"]):
                        doc = self._normalize_alert(feature, label, synced_at)
                        if doc:
                            docs.append(doc)
                except Exception:
                    # Alerts are best-effort; a failure here shouldn't kill
                    # the forecast harvesting for the same location.
                    pass

            if include_forecast:
                try:
                    for period in self.fetch_forecast(grid):
                        doc = self._normalize_forecast(
                            grid, period, "forecast", label, synced_at
                        )
                        if doc:
                            docs.append(doc)
                except Exception:
                    pass

            if include_hourly:
                try:
                    for period in self.fetch_hourly_forecast(grid):
                        doc = self._normalize_forecast(
                            grid, period, "forecast_hourly", label, synced_at
                        )
                        if doc:
                            docs.append(doc)
                except Exception:
                    pass

            if limit is not None:
                docs = docs[: int(limit)]
            all_docs.extend(docs)

        return all_docs


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def stable_hash(*parts: str, length: int = 16) -> str:
    """Deterministic hex digest helper (unused by default ids, but handy if a
    source ever lacks a natural stable key like an alert id or start time)."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]