-- Companion enrichment schema for Immich (same Postgres instance).
-- Designed to be additive and avoid Immich migrations: keep in separate "enrich" schema.
-- Key assets by Immich assetId (UUID) and/or content_sha256 for cross-system joins.
-- Apply manually if DB already initialized: psql -U "$DB_USERNAME" -d "$DB_DATABASE_NAME" -f /path/to/enrich_schema.sql

BEGIN;

CREATE SCHEMA IF NOT EXISTS enrich;

-- Canonical asset identity / linkage
CREATE TABLE IF NOT EXISTS enrich.asset (
    id              BIGSERIAL PRIMARY KEY,
    immich_asset_id UUID,                         -- Immich asset UUID when known
    content_sha256  TEXT,                         -- hash of original bytes for dedupe/cross-source joins
    original_path   TEXT,                         -- path as seen by ingest/enrichment jobs
    library_path    TEXT,                         -- path in Immich library/external library when known
    source_system   TEXT,                         -- e.g. immich, icloudpd, google_takeout, android, local
    source_ref      TEXT,                         -- source-specific id/path/album reference
    mime_type       TEXT,
    byte_size       BIGINT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT enrich_asset_identity_chk CHECK (immich_asset_id IS NOT NULL OR content_sha256 IS NOT NULL OR original_path IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_enrich_asset_immich_asset_id
    ON enrich.asset (immich_asset_id) WHERE immich_asset_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_enrich_asset_content_sha256
    ON enrich.asset (content_sha256) WHERE content_sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_enrich_asset_source
    ON enrich.asset (source_system, source_ref);
CREATE INDEX IF NOT EXISTS ix_enrich_asset_original_path
    ON enrich.asset (original_path);

-- Full raw metadata dump (ExifTool -j -a -G1 -s, source sidecars, etc.)
CREATE TABLE IF NOT EXISTS enrich.asset_metadata_raw (
    id              BIGSERIAL PRIMARY KEY,
    asset_id        BIGINT NOT NULL REFERENCES enrich.asset(id) ON DELETE CASCADE,
    extractor       TEXT NOT NULL,                -- e.g. exiftool, immich_api, xmp_sidecar, takeout_json
    extractor_version TEXT,
    extracted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_path     TEXT,                         -- file/sidecar/json source for this dump
    raw_json        JSONB NOT NULL,               -- complete extracted metadata payload
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS ix_asset_metadata_raw_asset_extracted
    ON enrich.asset_metadata_raw (asset_id, extracted_at DESC);
CREATE INDEX IF NOT EXISTS ix_asset_metadata_raw_extractor
    ON enrich.asset_metadata_raw (extractor);
CREATE INDEX IF NOT EXISTS idx_metadata_gin
    ON enrich.asset_metadata_raw USING GIN (raw_json);

-- Normalized/core fields for fast queries (derived from Immich + raw metadata)
CREATE TABLE IF NOT EXISTS enrich.asset_core (
    asset_id            BIGINT PRIMARY KEY REFERENCES enrich.asset(id) ON DELETE CASCADE,
    captured_at         TIMESTAMPTZ,
    captured_at_src     TEXT,                     -- e.g. exif_datetimeoriginal, quicktime_createdate, filename, takeout
    timezone_offset     TEXT,
    camera_make         TEXT,
    camera_model        TEXT,
    lens_model          TEXT,
    software            TEXT,
    device_id           TEXT,
    width               INTEGER,
    height              INTEGER,
    orientation         INTEGER,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    altitude_m          DOUBLE PRECISION,
    gps_accuracy_m      DOUBLE PRECISION,
    location_src        TEXT,
    duration_ms         BIGINT,                   -- for videos/live photos
    media_type          TEXT,                     -- image/video/live
    is_screenshot       BOOLEAN,
    content_hash_algo   TEXT DEFAULT 'sha256',
    quality_notes       JSONB,                    -- compression/transcode/originality observations
    normalized_json     JSONB,                    -- additional normalized fields
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_asset_core_captured_at ON enrich.asset_core (captured_at);
CREATE INDEX IF NOT EXISTS ix_asset_core_device ON enrich.asset_core (camera_make, camera_model);
CREATE INDEX IF NOT EXISTS ix_asset_core_location ON enrich.asset_core (latitude, longitude);

-- Local VLM/AI enrichment results (captions, objects/scenes, OCR, embeddings refs)
CREATE TABLE IF NOT EXISTS enrich.asset_vlm (
    id                  BIGSERIAL PRIMARY KEY,
    asset_id            BIGINT NOT NULL REFERENCES enrich.asset(id) ON DELETE CASCADE,
    model_name          TEXT NOT NULL,            -- e.g. llava, llava:13b, moondream
    model_runtime       TEXT,                     -- e.g. ollama
    prompt_version      TEXT,
    enriched_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    caption_detailed    TEXT,
    caption_short       TEXT,
    objects             JSONB,                    -- detected objects/entities with optional confidence/boxes
    scenes              JSONB,                    -- scene labels/tags
    activities          JSONB,
    attributes          JSONB,                    -- colors, weather, indoor/outdoor, etc.
    ocr_text            TEXT,
    ocr_blocks          JSONB,                    -- structured OCR output when available
    embedding_model     TEXT,
    embedding           JSONB,                    -- store vector as JSON array if pgvector extension unavailable
    -- embedding_vec    VECTOR(768),             -- optional: enable pgvector/vector extension and use typed column
    safety_flags        JSONB,
    raw_response        JSONB,                    -- full model response for audit/debug
    quality_score       DOUBLE PRECISION,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS ix_asset_vlm_asset_enriched
    ON enrich.asset_vlm (asset_id, enriched_at DESC);
CREATE INDEX IF NOT EXISTS ix_asset_vlm_model
    ON enrich.asset_vlm (model_name, prompt_version);

-- Face enrichment results (prefer Immich faces where available; store supplemental/local results here)
CREATE TABLE IF NOT EXISTS enrich.asset_faces (
    id                  BIGSERIAL PRIMARY KEY,
    asset_id            BIGINT NOT NULL REFERENCES enrich.asset(id) ON DELETE CASCADE,
    provider            TEXT NOT NULL,            -- e.g. immich, insightface, local
    person_label        TEXT,                     -- local label/cluster id; avoid sensitive claims
    confidence          DOUBLE PRECISION,
    bbox                JSONB,                    -- {x,y,w,h} or polygon in image coords
    landmarks           JSONB,
    embedding_model     TEXT,
    embedding           JSONB,
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_response        JSONB
);

CREATE INDEX IF NOT EXISTS ix_asset_faces_asset ON enrich.asset_faces (asset_id);
CREATE INDEX IF NOT EXISTS ix_asset_faces_person ON enrich.asset_faces (provider, person_label);

-- Sidecar/output tracking (XMP/JSON non-destructive writes)
CREATE TABLE IF NOT EXISTS enrich.asset_sidecar (
    id                  BIGSERIAL PRIMARY KEY,
    asset_id            BIGINT NOT NULL REFERENCES enrich.asset(id) ON DELETE CASCADE,
    sidecar_type        TEXT NOT NULL,            -- e.g. xmp, json
    sidecar_path        TEXT NOT NULL,
    target_original_path TEXT,
    status              TEXT NOT NULL CHECK (status IN ('written', 'skipped_unsafe', 'error')),
    checksum_sha256     TEXT,
    written_at          TIMESTAMPTZ,
    writer              TEXT,                     -- e.g. exiftool, custom_script
    writer_version      TEXT,
    error_message       TEXT,
    details             JSONB
);

CREATE INDEX IF NOT EXISTS ix_asset_sidecar_asset ON enrich.asset_sidecar (asset_id, written_at DESC);
CREATE INDEX IF NOT EXISTS ix_asset_sidecar_status ON enrich.asset_sidecar (status);

-- Convenience view: latest VLM enrichment per asset
CREATE OR REPLACE VIEW enrich.asset_latest_vlm AS
SELECT DISTINCT ON (asset_id)
    id, asset_id, model_name, model_runtime, prompt_version, enriched_at,
    caption_detailed, caption_short, objects, scenes, activities, attributes,
    ocr_text, ocr_blocks, embedding_model, quality_score
FROM enrich.asset_vlm
ORDER BY asset_id, enriched_at DESC, id DESC;

COMMIT;