# SQL setup files for the weather pipeline

These SQL files let you create the two tables manually from a `psql` session
(connected with your `LAKEBASE_URL`) **or** a Databricks SQL editor. You do
**not** need them if you let the app create the tables automatically:

- `app.py` calls `lakebase.ensure_weather_tables()` on the first
  `POST /weather/sync` (creates `weather_documents` and `weather_embeddings`).
- `notebooks/ingest_weather_embeddings.py` calls
  `lakebase.ensure_weather_embeddings()` before writing vectors.

Order:

1. `01_setup_weather_documents.sql` - raw documents table (no vectors).
2. `02_setup_weather_embeddings.sql` - pgvector table + HNSW index.

In `02_...`, replace `{{EMBEDDING_DIM}}` with your model's output dimension:

| Model                                    | Dim |
|------------------------------------------|-----|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `sentence-transformers/all-mpnet-base-v2`| 768 |
| `BAAI/bge-small-en-v1.5`                 | 384 |
| `BAAI/bge-base-en-v1.5`                  | 768 |

Quick sanity checks:

```sql
SELECT source_type, COUNT(*) FROM weather_documents GROUP BY source_type;
SELECT COUNT(*), COUNT(embedding) FROM weather_embeddings;
```