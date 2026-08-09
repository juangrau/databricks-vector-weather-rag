# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook/script is the **embedding pipeline** for the Weather
# MAGIC Intelligence homework. It:
# MAGIC
# MAGIC 1. Reads **unembedded** rows from the `weather_documents` table in
# MAGIC    Lakebase (i.e. documents that do not have any rows in
# MAGIC    `weather_embeddings` yet).
# MAGIC 2. Chunks `narrative_text` with a sliding window
# MAGIC    (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`). Most NWS text is short enough
# MAGIC    to stay a single chunk; long combined alert description+instruction
# MAGIC    strings get split.
# MAGIC 3. Embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`
# MAGIC    (384-dim) - the SAME model used by the existing ticker-news
# MAGIC    pipeline, so both datasets stay queryable with the same pgvector
# MAGIC    distance operators.
# MAGIC 4. Writes embeddings into `weather_embeddings` via **psycopg2 +
# MAGIC    `execute_values`** (no Spark JDBC - Spark's JDBC writer does not work
# MAGIC    reliably against this Lakebase instance). Embeddings are cast with
# MAGIC    `%s::vector`; psycopg2's adapter handles a Python list directly.
# MAGIC
# MAGIC It reuses the SAME Lakebase connection helper (`lakebase.get_connection()`,
# MAGIC secret scope `database` / key `lakebase-url`) as `app.py`.
# MAGIC
# MAGIC **Run it as a Databricks notebook** (import into your Git folder) or as a
# MAGIC **plain script**:
# MAGIC
# MAGIC ```bash
# MAGIC python notebooks/ingest_weather_embeddings.py --chunk_size 800 --chunk_overlap 100
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Install all required packages (Databricks only)
# MAGIC %pip install -q 'databricks-sdk>=0.30.0' psycopg2-binary python-dotenv sentence-transformers

# COMMAND ----------

# DBTITLE 1,Config: widgets (Databricks) or CLI args / env vars (plain script)
import os
import sys

try:
    dbutils  # noqa: F821 -- defined by Databricks runtime (cluster/notebook)
    _HAS_DBUTILS = True
except NameError:
    _HAS_DBUTILS = False

_DEFAULTS = {
    "documents_table_name": "weather_documents",
    "embeddings_table_name": "weather_embeddings",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 800,
    "chunk_overlap": 100,
    "batch_size": 32,
}

_CONFIG = {}
if _HAS_DBUTILS:
    for key, default in _DEFAULTS.items():
        dbutils.widgets.text(key, str(default), key)
    _CONFIG = {key: dbutils.widgets.get(key) for key in _DEFAULTS}
else:
    import argparse

    parser = argparse.ArgumentParser(
        description="Embed unembedded weather_documents into weather_embeddings"
    )
    for key, default in _DEFAULTS.items():
        parser.add_argument(f"--{key}", default=os.environ.get(key.upper(), str(default)))
    args = parser.parse_args()
    _CONFIG = {key: getattr(args, key) for key in _DEFAULTS}

DOCUMENTS_TABLE_NAME = _CONFIG["documents_table_name"]
EMBEDDINGS_TABLE_NAME = _CONFIG["embeddings_table_name"]
EMBEDDING_MODEL_NAME = _CONFIG["embedding_model"]
CHUNK_SIZE = int(_CONFIG["chunk_size"])
CHUNK_OVERLAP = int(_CONFIG["chunk_overlap"])
BATCH_SIZE = int(_CONFIG["batch_size"])

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type VECTOR(N) must match exactly. Same dimension map as the
# ticker-news pipeline; default model is 384-dim.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")
print(f"Chunking: CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")

# COMMAND ----------

# DBTITLE 1,Connection via the shared lakebase helper
# Reuse lakebase.get_connection() (same secret + decoding as app.py). In a
# Databricks notebook the repo root isn't on sys.path by default, so add it.
if globals().get("__file__"):
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

try:
    import lakebase as _lakebase

    print("Using lakebase.get_connection() reader (scope=database/lakebase-url)")
except Exception as exc:  # pragma: no cover - fallback for standalone use
    print(f"lakebase import failed ({exc}); falling back to LAKEBASE_URL env var")
    import base64

    raw = os.environ["LAKEBASE_URL"]
    if not raw.startswith(("postgresql://", "postgres://")):
        raw = base64.b64decode(raw).decode("utf-8")
    _LAKEBASE_DSN = raw

    import psycopg2
    from psycopg2.extras import RealDictCursor

    class _lakebase:  # noqa: N801 - minimal stand-in for the shared helper
        @staticmethod
        def get_connection():
            return psycopg2.connect(_LAKEBASE_DSN, cursor_factory=RealDictCursor)

        @staticmethod
        def ensure_weather_embeddings(
            table_name, documents_table, embedding_dim=EMBEDDING_DIM,
        ):
            with _lakebase.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    cur.execute(
                        f"""CREATE TABLE IF NOT EXISTS {table_name} (
                            id            TEXT PRIMARY KEY,
                            document_id   TEXT NOT NULL,
                            chunk_index   INT NOT NULL,
                            chunk_text    TEXT NOT NULL,
                            embedding     VECTOR({int(embedding_dim)}) NOT NULL,
                            model_name    TEXT NOT NULL,
                            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                            UNIQUE (document_id, chunk_index)
                        )"""
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding "
                        f"ON {table_name} USING hnsw (embedding vector_cosine_ops)"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_document_id "
                        f"ON {table_name} (document_id)"
                    )
                    conn.commit()

# COMMAND ----------

# DBTITLE 1,Read unembedded weather documents
from psycopg2.extras import execute_values

# Documents that have NO row in weather_embeddings yet (idempotent re-runs:
# already-embedded docs are skipped). Also skips empty narrative_text.
_SELECT_UNEMBEDDED = f"""
    SELECT d.id, d.location, d.source_type, d.headline,
           d.narrative_text, d.effective_at
    FROM {DOCUMENTS_TABLE_NAME} d
    WHERE NOT EXISTS (
        SELECT 1 FROM {EMBEDDINGS_TABLE_NAME} e
        WHERE e.document_id = d.id
    )
      AND d.narrative_text IS NOT NULL
      AND trim(d.narrative_text) <> ''
"""

with _lakebase.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(_SELECT_UNEMBEDDED)
        docs = cur.fetchall()

print(f"Loaded {len(docs)} unembedded documents from {DOCUMENTS_TABLE_NAME}")
if docs:
    for r in docs[:5]:
        print("  -", r["id"], "|", r["location"], "|", r["source_type"], "|",
              str(r["narrative_text"])[:60], "...")

# COMMAND ----------

# DBTITLE 1,Chunk narrative_text (sliding window)
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split long text into overlapping chunks of `chunk_size` chars, stepping
    by `chunk_size - chunk_overlap` (the classic sliding-window pattern)."""
    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if not piece:
            continue
        chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks or ([text.strip()] if text.strip() else [])


chunk_rows = []  # (document_id, chunk_index, chunk_text)
for doc in docs:
    text = doc["narrative_text"] or ""
    for i, piece in enumerate(chunk_text(text)):
        chunk_rows.append((doc["id"], i, piece))

n_multi = sum(1 for doc in docs if len(chunk_text(doc["narrative_text"] or "")) > 1)
print(f"Produced {len(chunk_rows)} chunks from {len(docs)} documents")
print(f"  {n_multi} document(s) were long enough to need chunking")

# COMMAND ----------

# DBTITLE 1,Load embedding model and embed chunks in batches
# EMBEDDING_DIM may be referenced by the fallback connection class, which is
# defined earlier in the file; keep the model-loading cell minimal.
from sentence_transformers import SentenceTransformer

print(f"Loading embedding model {EMBEDDING_MODEL_NAME} ...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Model loaded.")

emb_rows = []  # (id, document_id, chunk_index, chunk_text, embedding, model_name)
for start in range(0, len(chunk_rows), BATCH_SIZE):
    batch = chunk_rows[start : start + BATCH_SIZE]
    texts = [r[2] for r in batch]
    vectors = model.encode(texts, show_progress_bar=False, batch_size=min(len(texts), 256))
    for (doc_id, chunk_index, chunk_text), vec in zip(batch, vectors):
        emb_id = f"{doc_id}_{chunk_index}"
        emb_rows.append(
            (emb_id, doc_id, chunk_index, chunk_text, vec.tolist(), EMBEDDING_MODEL_NAME)
        )
    print(f"  embedded {min(start + BATCH_SIZE, len(chunk_rows))}/{len(chunk_rows)} chunks")

print(f"Computed {len(emb_rows)} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# DBTITLE 1,Upsert embeddings into weather_embeddings (psycopg2 execute_values)
if not emb_rows:
    print("No chunks to embed - nothing to write.")
else:
    # Ensure the pgvector destination table exists (idempotent). Requires the
    # vector extension, already enabled on this Lakebase instance.
    _lakebase.ensure_weather_embeddings(
        table_name=EMBEDDINGS_TABLE_NAME,
        documents_table=DOCUMENTS_TABLE_NAME,
        embedding_dim=EMBEDDING_DIM,
    )

    _INSERT_CHUNKS = f"""
        INSERT INTO {EMBEDDINGS_TABLE_NAME} (
            id, document_id, chunk_index, chunk_text, embedding, model_name
        )
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """
    # Template casts column 5 to pgvector's vector type. Passing the embedding
    # as a plain Python list is enough - psycopg2 + pgvector handle the cast.
    _TEMPLATE = "(%s, %s, %s, %s, %s::vector, %s)"

    with _lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur, _INSERT_CHUNKS, emb_rows, template=_TEMPLATE, page_size=200
            )
            conn.commit()
            inserted = cur.rowcount

    print(f"OK: wrote/attempted {len(emb_rows)} chunk embeddings into "
          f"{EMBEDDINGS_TABLE_NAME} (duplicates skipped via ON CONFLICT DO NOTHING)")
    print(f"Total rows now in {EMBEDDINGS_TABLE_NAME}:")

    with _lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM {EMBEDDINGS_TABLE_NAME}")
            total = cur.fetchone()["n"]
    print(f"  {total}")

print("\nDone. You can now hit POST /weather/search against the Flask app.")