"""
Lakebase (Databricks-managed Postgres) connection helper + weather schema.

Connects using a single LAKEBASE_URL (a standard Postgres connection URL,
e.g. postgresql://role:password@host:5432/databricks_postgres?sslmode=require)
pointing at a native Postgres role with a static, non-expiring password.
This keeps setup to a single secret instead of five separate env vars.

On Databricks Apps the URL comes from a secret scope (set in app.yaml).
Locally, you can set LAKEBASE_URL in a .env file instead.

This module also owns the DDL/migrations for the weather pipeline tables
(weather_documents, weather_embeddings) so app.py and the ingestion notebook
share a single source of truth for the schema.
"""

import base64
import os
from contextlib import contextmanager

import psycopg2
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

# Table names stay configurable but default to these.
WEATHER_DOCUMENTS_TABLE = os.environ.get(
    "WEATHER_DOCUMENTS_TABLE", "weather_documents"
)
WEATHER_EMBEDDINGS_TABLE = os.environ.get(
    "WEATHER_EMBEDDINGS_TABLE", "weather_embeddings"
)

# Embedding model + output dimension used by the whole weather pipeline.
# We reuse the SAME model as the existing ticker-news pipeline
# (sentence-transformers/all-MiniLM-L6-v2 -> 384-dim) so both datasets stay
# compatible with the same pgvector distance operator conventions.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384


def _lakebase_url() -> str:
    """Resolve the Lakebase connection URL from env var or Databricks secret scope.

    Handles raw, single- and double-base64-encoded secret values.
    """
    env_url = os.environ.get("LAKEBASE_URL")
    if env_url:
        return env_url

    secret = WorkspaceClient().secrets.get_secret(scope=_SCOPE, key=_KEY)
    raw_value = secret.value

    # Already a valid Postgres URL (not encoded)
    if raw_value.startswith(("postgresql://", "postgres://")):
        return raw_value

    # Try first decode
    try:
        first_decode = base64.b64decode(raw_value).decode("utf-8")

        # If it's a valid Postgres URL after first decode, return it
        if first_decode.startswith(("postgresql://", "postgres://")):
            return first_decode

        # Otherwise, try decoding again (double-encoded case)
        try:
            second_decode = base64.b64decode(first_decode).decode("utf-8")
            if second_decode.startswith(("postgresql://", "postgres://")):
                return second_decode
            else:
                # If second decode doesn't give us a URL, use first decode
                return first_decode
        except Exception:
            # If second decode fails, return first decode
            return first_decode

    except Exception as e:
        # If base64 decode fails, the secret might be plain text.
        raise ValueError(
            f"Failed to decode Lakebase URL from secret. Raw value starts with: "
            f"{raw_value[:20]}... Error: {e}"
        )


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a query against Lakebase and return rows as list[dict].

    Commits after execution so INSERT/UPDATE ... RETURNING statements
    persist. psycopg2 rolls back on close, so without this writes were
    silently discarded.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            conn.commit()
            return rows


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


def ensure_weather_documents(table_name: str | None = None) -> None:
    """DDL for the raw weather_documents table (no vectors here).

    Mirrors the ticker_news_documents pattern: one row per normalized NWS
    document (alert, daily-forecast period, or hourly-forecast period), with
    the free-text body to embed in narrative_text and the raw JSON in payload.
    """
    table = table_name or WEATHER_DOCUMENTS_TABLE
    run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id              TEXT PRIMARY KEY,
            location        TEXT NOT NULL,
            source_type     TEXT NOT NULL,
            headline        TEXT,
            narrative_text  TEXT NOT NULL,
            effective_at    TIMESTAMPTZ,
            payload         JSONB NOT NULL,
            synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    run_write(f"CREATE INDEX IF NOT EXISTS idx_{table}_location "
              f"ON {table} (location)")
    run_write(f"CREATE INDEX IF NOT EXISTS idx_{table}_source_type "
              f"ON {table} (source_type)")


def ensure_weather_embeddings(
    table_name: str | None = None,
    documents_table: str | None = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> None:
    """DDL for the weather_embeddings table (pgvector).

    Requires the pgvector extension (already enabled on this Lakebase
    instance) and creates an HNSW cosine-distance index so the search
    endpoint's `<=>` lookups stay fast.
    """
    table = table_name or WEATHER_EMBEDDINGS_TABLE
    docs_table = documents_table or WEATHER_DOCUMENTS_TABLE
    run_write("CREATE EXTENSION IF NOT EXISTS vector")
    run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id            TEXT PRIMARY KEY,
            document_id   TEXT NOT NULL REFERENCES {docs_table}(id) ON DELETE CASCADE,
            chunk_index   INT NOT NULL,
            chunk_text    TEXT NOT NULL,
            embedding     VECTOR({int(embedding_dim)}) NOT NULL,
            model_name    TEXT NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_index)
        )
        """
    )
    run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{table}_embedding "
        f"ON {table} USING hnsw (embedding vector_cosine_ops)"
    )
    run_write(f"CREATE INDEX IF NOT EXISTS idx_{table}_document_id "
              f"ON {table} (document_id)")


def ensure_weather_tables(
    documents_table: str | None = None,
    embeddings_table: str | None = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> None:
    """Create both weather tables if they don't exist (documents first, so the
    FK on weather_embeddings.document_id resolves)."""
    ensure_weather_documents(documents_table)
    ensure_weather_embeddings(embeddings_table, documents_table, embedding_dim)
