-- Setup script for weather_documents (raw NWS documents)
-- Run manually in your Lakebase Postgres database OR let the Flask app create
-- it automatically via lakebase.ensure_weather_tables() on the first
-- POST /weather/sync.

CREATE TABLE IF NOT EXISTS weather_documents (
    id              TEXT PRIMARY KEY,   -- alert id or nws:<office>:<x,y>:<type>:<start>
    location        TEXT NOT NULL,      -- city/state label or lat,lon
    source_type     TEXT NOT NULL,      -- 'alert' | 'forecast' | 'forecast_hourly'
    headline        TEXT,               -- event / period name
    narrative_text  TEXT NOT NULL,      -- free-text body to embed
    effective_at    TIMESTAMPTZ,
    payload         JSONB NOT NULL,     -- raw NWS properties, for provenance
    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);
CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);

-- Verify the table was created
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;