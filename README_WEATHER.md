# Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

Homework for the Databricks Lakebase bootcamp. Builds on the
`databricks-lakebase-app-day-2-corrected` reference app: instead of syncing
structured records + news articles from the Massive API, this project
harvests **unstructured weather text**, chunks/embeds it with
sentence-transformers, stores vectors in pgvector on Lakebase, and exposes a
Flask REST endpoint that does semantic search via pgvector's `<=>` operator.

```
NWS API ──sync──▶ weather_documents ──embed──▶ weather_embeddings ──search──▶ REST
(weather_client.py)          (ingest script, psycopg2)      (Flask app, pgvector <=>)
```

End-to-end query:

```bash
POST /weather/search {"query": "flash flood risk this weekend"}
```

returns the most semantically relevant weather documents, ranked by cosine
similarity.

## 1. Files

| File | Role |
|------|------|
| `weather_client.py` | NWS API client (mirrors `massive_client.py`): geocoding, alerts, daily + hourly forecasts, document normalization |
| `lakebase.py` | Lakebase connection helper (reuses homework1's env/secret + base64 handling) **plus** DDL/migrations for `weather_documents` / `weather_embeddings` |
| `app.py` | Flask app: `POST /weather/sync`, `POST /weather/search` (+ `GET /healthz`, `GET /weather/documents`) |
| `notebooks/ingest_weather_embeddings.py` | psycopg2 embedding pipeline: read unembedded docs → chunk → embed → write via `execute_values` |
| `sql/` | Optional manual setup scripts for both tables |
| `resources/ingest_weather_embeddings_job.yml` + `databricks.yml` | Optional Asset Bundle to schedule the ingest notebook as a Workflow |
| `requirements.txt`, `app.yaml`, `.env.example`, `setup_secrets.py` | Dependencies, deployment manifest, local env template, one-time secret setup |

## 2. Why this data source

I chose the **National Weather Service API** (`api.weather.gov`) for the
reasons the assignment recommends and because it lets the homework focus on
harvesting / vectorization / retrieval rather than auth plumbing:

- **No API key required.** Only mandate is a descriptive `User-Agent` header
  (NWS returns 403 without one) and being polite: the client sleeps
  `request_delay` (default 0.25s) between its own calls, far under NWS's
  generous limits.
- **Rich free text.** Alerts carry `description` + `instruction` narrative
  (`* WHAT... dangerous rip currents...`); the daily forecast carries a
  `detailedForecast` narrative per period; hourly periods carry
  `shortForecast` (+ temperature, wind).
- **Stable, dedupable ids.** Alerts have a `urn:oid:...` id; forecast records
  use the period `startTime`, giving a natural upsert key.

**Location resolution:** the sync body accepts `"Chicago, IL"` or `"lat,lon"`
strings. City/state strings are geocoded via the **Open-Meteo Geocoding API**
(free, key-less), then each lat/lon resolves to an NWS grid point via
`GET /points/{lat},{lon}`. Unresolvable locations are reported in the sync
response (`unresolved_locations`) instead of failing the batch.

## 3. Schema decisions

### `weather_documents` (raw docs, one row per normalized NWS item)

```sql
CREATE TABLE weather_documents (
    id             TEXT PRIMARY KEY,   -- alert urn:oid OR nws:<office>:<x,y>:<type>:<startTime>
    location       TEXT NOT NULL,      -- city/state label or "lat,lon"
    source_type    TEXT NOT NULL,      -- 'alert' | 'forecast' | 'forecast_hourly'
    headline       TEXT,               -- event / period name (e.g. "Flash Flood Warning")
    narrative_text TEXT NOT NULL,      -- free-text body to embed
    effective_at   TIMESTAMPTZ,        -- alert effective/sent, or period startTime
    payload        JSONB NOT NULL,     -- raw NWS properties, kept for provenance
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

This mirrors the reference `ticker_news_documents` pattern (raw docs in one
table, vectors in a separate table), so the ingestion script only reads plain
text columns and doesn't parse JSONB for the common case.

### `weather_embeddings` (pgvector)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE weather_embeddings (
    id           TEXT PRIMARY KEY,                    -- '<document_id>_<chunk_index>'
    document_id  TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    chunk_text   TEXT NOT NULL,
    embedding    VECTOR(384) NOT NULL,
    model_name   TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);
```

Decisions:
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` → **384-dim**,
  the SAME model as the existing ticker-news pipeline, so both datasets remain
  queryable with the same pgvector distance operators. `VECTOR(384)` and the
  schema stay in sync (the notebook maps model → dimension, like the
  reference).
- **Distance / index:** cosine similarity via `<=>` with an **HNSW**
  `vector_cosine_ops` index for fast approximate nearest-neighbor search.
  `ivfflat` is the alternative; HNSW needs no training split and is the better
  default at this scale.
- **Chunking:** `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` (sliding window, step
  `size - overlap`). Most NWS text is a single short period, so most documents
  stay one chunk; long combined alert `description` + `instruction` text is the
  case where chunking actually triggers. With 800 chars, one vector ≈ any
  typical NWS narrative, which is a good granularity for retrieval.
- **Write path:** psycopg2 only. The notebook uses
  `psycopg2.extras.execute_values` with `ON CONFLICT (id) DO NOTHING` and a
  `%s::vector` cast (psycopg2 adapts a plain Python list; no Spark JDBC, which
  doesn't work reliably against this Lakebase instance).
  `page_size` batches the inserts. The underlying `UNIQUE(document_id,
  chunk_index)` backstops idempotency.

## 4. How to run end-to-end

Prereqs: Lakebase instance with native-password role (same setup as the
reference app), and the `database/lakebase-url` secret stored once:

```bash
python setup_secrets.py        # prompts for the URL, stores base64 secret
```

Local dev: `cp .env.example .env`, paste `LAKEBASE_URL`, then:

```bash
pip install -r requirements.txt
python app.py                  # http://localhost:8000
```

**Step 1 — sync documents (NWS → `weather_documents`):**

```bash
curl -X POST localhost:8000/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Austin, TX", "41.878,-87.629"], "limit": 50}'
```

Creates both tables on first run (`lakebase.ensure_weather_tables()`), then
fetches alerts + daily forecast + hourly forecast per location and upserts a
flat document per item. `limit` caps documents per location (mirrors the
per-ticker news limit).

**Step 2 — embed (docs → vectors), psycopg2 pipeline:**

```bash
python notebooks/ingest_weather_embeddings.py --chunk_size 800 --chunk_overlap 100
```

Reads unembedded docs (`WHERE NOT EXISTS` in `weather_embeddings`), chunks,
embeds with all-MiniLM-L6-v2 in batches, writes via `execute_values`.
Idempotent — re-runs only touch new documents. In Databricks, import the
file as a notebook (cells use the same `dbutils.widgets` config) or schedule it
via the bundled Asset Bundle (`databricks bundle deploy -t dev`).

**Step 3 — search (query → top-k vectors):**

```bash
curl -X POST localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "risk of flooding near rivers", "top_k": 5}'
```

The query is embedded with the same model (loaded **once** at module level,
cached, not per request), then:

```sql
SELECT d.id, d.location, d.headline, d.narrative_text, e.chunk_text,
       1 - (e.embedding <=> %s::vector) AS similarity
FROM weather_embeddings e
JOIN weather_documents d ON d.id = e.document_id
ORDER BY e.embedding <=> %s::vector
LIMIT %s;
```

Edge cases handled: empty `weather_embeddings` returns an informative 200 with
a hint to run sync + ingest first; missing/empty `query` → 400; `top_k`
clamped to `[1, 20]`.

## 5. Known limitations & things I'd improve

- **Geocoding dependency.** City/state strings go through the free Open-Meteo
  API. A production version could bundle a gazetteer / forward-geocoder, cache
  geocoding results in Lakebase, and accept lat/lon directly (already
  supported).
- **Alert coverage.** I fetch active alerts by `point` (best-effort; NWS
  sometimes returns alerts whose polygon/circle covers the point unevenly).
  `GET /alerts/active?area={state}` would broaden recall at the cost of less
  precise location attribution.
- **Single-threaded embedding.** At NWS scale (hundreds of docs) this is fine;
  for large corpora I'd parallelize the psycopg2 reads/writes with
  `concurrent.futures.ThreadPoolExecutor` or embed in batches (already done
  via `batch_size`) rather than reaching for Spark on the write path.
- **Search only covers embedded docs.** Until Step 2 runs, `/weather/search`
  correctly returns an empty result rather than erroring — surfaced via
  `message`.
- **No RAG/LLM step yet.** This homework stops at retrieval; a follow-up could
  feed the top-k `chunk_text` context into an LLM call for the flash-flood
  answer.