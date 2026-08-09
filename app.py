"""
Weather Intelligence Databricks App:
- Harvests unstructured weather text from the National Weather Service API
  (api.weather.gov) via weather_client.py
- Upserts normalized documents into Lakebase `weather_documents`
  (Databricks-managed Postgres) via lakebase.py
- Embeds a search query with sentence-transformers and runs pgvector cosine
  similarity search over `weather_embeddings` (populated by
  notebooks/ingest_weather_embeddings.py)

Endpoints:
  GET  /healthz           liveness check
  GET  /                  endpoint list
  GET  /weather/documents recent synced documents (read-only debug helper)
  POST /weather/sync      body {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
  POST /weather/search    body {"query": "flash flood risk this weekend", "top_k": 5}

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import json as _json
import logging
import os

from flask import Flask, jsonify, request

import lakebase
from weather_client import WeatherClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)

DOCUMENTS_TABLE = lakebase.WEATHER_DOCUMENTS_TABLE
EMBEDDINGS_TABLE = lakebase.WEATHER_EMBEDDINGS_TABLE
EMBEDDING_MODEL_NAME = lakebase.DEFAULT_EMBEDDING_MODEL

MAX_TOP_K = 20

# SentenceTransformer is heavy (~90MB) and slow to load; load it ONCE at
# module level (lazily, on first search) and reuse it for every request
# instead of re-loading per request.
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s ...", EMBEDDING_MODEL_NAME)
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model ready (%d-dim)", lakebase.DEFAULT_EMBEDDING_DIM)
    return _embedder


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so API clients' resp.json() calls never choke on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return jsonify(
        {
            "service": "weather intelligence (unstructured -> pgvector -> search)",
            "endpoints": {
                "POST /weather/sync": '{"locations": ["Chicago, IL"], "limit": 50}',
                "POST /weather/search": '{"query": "flash flood risk this weekend", "top_k": 5}',
                "GET /weather/documents": "list recently synced documents",
            },
        }
    )


@app.route("/weather/documents")
def list_weather_documents():
    """Read-only helper: list documents already synced into Lakebase."""
    try:
        limit = int(request.args.get("limit", 20))
    except (TypeError, ValueError):
        return jsonify({"error": "limit must be an integer"}), 400
    limit = max(1, min(limit, 100))

    rows = lakebase.run_query(
        f"SELECT id, location, source_type, headline, effective_at, synced_at "
        f"FROM {DOCUMENTS_TABLE} ORDER BY synced_at DESC LIMIT %s",
        (limit,),
    )
    return jsonify([_serialize_doc(r) for r in rows])


@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    """Fetch + normalize weather documents from NWS and upsert them into
    weather_documents. Body: {"locations": ["Chicago, IL"], "limit": 50}."""
    lakebase.ensure_weather_tables()
    client = WeatherClient()

    body = request.get_json(silent=True) or {}
    locations = body.get("locations")
    if (
        not locations
        or not isinstance(locations, list)
        or not all(isinstance(l, str) and l.strip() for l in locations)
    ):
        return jsonify(
            {"error": "locations must be a non-empty list of "
                      "city/state or 'lat,lon' strings"}
        ), 400
    locations = [l.strip() for l in locations]

    try:
        limit = int(body.get("limit", 50))
    except (TypeError, ValueError):
        return jsonify({"error": "limit must be an integer"}), 400

    docs = client.sync(locations, limit=limit)
    synced = _upsert_weather_batch(docs)

    resp = {
        "synced": synced,
        "locations": len(locations),
        "unresolved_locations": client.unresolved_locations,
    }
    if not docs:
        resp["message"] = (
            "No documents fetched. Check the location strings and that "
            "api.weather.gov is reachable."
        )
    return jsonify(resp)


@app.route("/weather/search", methods=["POST"])
def weather_search():
    """Embed the query and run cosine-similarity search over weather_embeddings.

    Body: {"query": "risk of flooding near rivers", "top_k": 5}.
    top_k is clamped to [1, 20].
    """
    lakebase.ensure_weather_tables()

    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        top_k = int(body.get("top_k", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400
    top_k = max(1, min(top_k, MAX_TOP_K))

    # Edge case: nothing embedded yet.
    count_rows = lakebase.run_query(
        f"SELECT COUNT(*) AS n FROM {EMBEDDINGS_TABLE}"
    )
    if not count_rows or not count_rows[0]["n"]:
        return jsonify(
            {
                "query": query,
                "top_k": top_k,
                "results": [],
                "message": (
                    "weather_embeddings is empty. Run POST /weather/sync, then "
                    "notebooks/ingest_weather_embeddings.py to embed the "
                    "documents before searching."
                ),
            }
        ), 200

    model = _get_embedder()
    query_vec = model.encode([query])[0].tolist()

    rows = lakebase.run_query(
        f"""
        SELECT d.id     AS document_id,
               d.location,
               d.source_type,
               d.headline,
               d.narrative_text,
               e.chunk_index,
               e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vec, query_vec, top_k),
    )

    results = []
    for r in rows:
        results.append(
            {
                "document_id": r["document_id"],
                "location": r["location"],
                "source_type": r["source_type"],
                "headline": r["headline"],
                "narrative_text": r["narrative_text"],
                "chunk_index": r["chunk_index"],
                "chunk_text": r["chunk_text"],
                "similarity": round(float(r["similarity"]), 6),
            }
        )

    return jsonify({"query": query, "top_k": top_k, "results": results})


def _upsert_weather_batch(docs: list[dict]) -> int:
    """Upsert normalized weather documents into weather_documents in one
    batched statement (psycopg2.extras.execute_values), deduping on id."""
    if not docs:
        return 0
    from psycopg2.extras import execute_values

    values = [
        (
            d["id"],
            d["location"],
            d["source_type"],
            d["headline"],
            d["narrative_text"],
            d["effective_at"],
            _json.dumps(d["payload"]),
            d["synced_at"],
        )
        for d in docs
    ]
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {DOCUMENTS_TABLE} (
                    id, location, source_type, headline,
                    narrative_text, effective_at, payload, synced_at
                )
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    location        = EXCLUDED.location,
                    source_type     = EXCLUDED.source_type,
                    headline        = EXCLUDED.headline,
                    narrative_text  = EXCLUDED.narrative_text,
                    effective_at    = EXCLUDED.effective_at,
                    payload         = EXCLUDED.payload,
                    synced_at       = EXCLUDED.synced_at
                """,
                values,
                page_size=500,
            )
            conn.commit()
    return len(values)


def _serialize_doc(row: dict) -> dict:
    """Make a lakebase row (RealDictRow) JSON-safe (datetime -> isoformat)."""
    out = dict(row)
    for key, val in out.items():
        if hasattr(val, "isoformat"):
            out[key] = val.isoformat()
    return out


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")