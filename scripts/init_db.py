#!/usr/bin/env python3
"""Initialize database schema for prediction logging."""
import asyncio

import asyncpg

DATABASE_URL = "postgresql://ecg_user:ecg_password_dev@localhost:5432/ecg_predictions"

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    ecg_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Patient info
    patient_age INTEGER,
    patient_sex VARCHAR(1),

    -- Model info
    model_version VARCHAR(50) NOT NULL,
    processing_time_ms FLOAT,

    -- Predictions (stored as JSONB for flexibility)
    diagnoses JSONB NOT NULL,
    recommendations JSONB NOT NULL,

    -- Metadata
    request_ip VARCHAR(50),

    -- Indexes
    CONSTRAINT ecg_id_unique UNIQUE (ecg_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_diagnoses ON predictions USING GIN (diagnoses);

-- View for easy querying
CREATE OR REPLACE VIEW prediction_summary AS
SELECT
    date_trunc('hour', created_at) as hour,
    model_version,
    COUNT(*) as total_predictions,
    AVG(processing_time_ms) as avg_latency_ms,
    AVG(jsonb_array_length(diagnoses)) as avg_diagnoses_count
FROM predictions
GROUP BY hour, model_version
ORDER BY hour DESC;
"""

async def init_db():
    """Create tables."""
    print("🔄 Connecting to database...")
    conn = await asyncpg.connect(DATABASE_URL)

    print("📝 Creating tables...")
    await conn.execute(CREATE_TABLES)

    print("✅ Database initialized successfully!")

    # Verify
    count = await conn.fetchval("SELECT COUNT(*) FROM predictions")
    print(f"📊 Current predictions in DB: {count}")

    await conn.close()

if __name__ == "__main__":
    asyncio.run(init_db())
