-- Setup script for weather_embeddings (pgvector)
-- Run manually in your Lakebase Postgres database OR let
-- notebooks/ingest_weather_embeddings.py create it (it calls
-- lakebase.ensure_weather_embeddings() before writing).
-- Replace {{EMBEDDING_DIM}} with your model's dimension (384 for
-- sentence-transformers/all-MiniLM-L6-v2).

-- Enable pgvector extension (already enabled on this Lakebase instance)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_embeddings (
    id            TEXT PRIMARY KEY,                      -- <document_id>_<chunk_index>
    document_id   TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    chunk_text    TEXT NOT NULL,
    embedding     VECTOR({{EMBEDDING_DIM}}) NOT NULL,   -- 384 for all-MiniLM-L6-v2
    model_name    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- HNSW index for fast cosine similarity search (used by the <=> operator)
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
    ON weather_embeddings (document_id);

-- Verify the table was created
SELECT table_name, column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;