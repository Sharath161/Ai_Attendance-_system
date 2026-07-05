-- Smart Attendance System — initial PostgreSQL schema
-- Run once on a fresh database, or let AUTO_CREATE_TABLES=true handle it for SQLite dev.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Enum types ────────────────────────────────────────────────────────────────

DO $$ BEGIN CREATE TYPE device_status AS ENUM ('active', 'offline', 'fault');
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN CREATE TYPE student_status AS ENUM ('active', 'pending_embedding', 'embedding_failed');
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN CREATE TYPE job_status AS ENUM ('pending', 'processing', 'complete', 'failed');
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN CREATE TYPE attendance_status AS ENUM (
    'recognized', 'unknown', 'low_confidence',
    'no_face_detected', 'multiple_faces_detected'
); EXCEPTION WHEN duplicate_object THEN null; END $$;

-- ── Tables ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS esp32_devices (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name             VARCHAR(100) NOT NULL,
    device_mac       VARCHAR(17) UNIQUE,
    room_label       VARCHAR(150),
    api_key          VARCHAR(64) UNIQUE NOT NULL,
    firmware_version VARCHAR(20),
    last_seen_at     TIMESTAMPTZ,
    status           device_status NOT NULL DEFAULT 'active',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS students (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id VARCHAR(64) UNIQUE NOT NULL,
    name       VARCHAR(255) NOT NULL,
    class_id   VARCHAR(64),
    status     student_status NOT NULL DEFAULT 'pending_embedding',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS student_embeddings (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id          UUID NOT NULL REFERENCES students(id),
    embedding           DOUBLE PRECISION[] NOT NULL,
    model_version       VARCHAR(100) NOT NULL,
    image_quality_score DOUBLE PRECISION,
    active              BOOLEAN NOT NULL DEFAULT true,
    demographic_group   VARCHAR(50),      -- FairFace race label
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS image_jobs (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_path    TEXT NOT NULL,
    camera_id     VARCHAR(64) NOT NULL,
    device_id     UUID REFERENCES esp32_devices(id),
    class_id      VARCHAR(64),
    captured_at   TIMESTAMPTZ NOT NULL,
    status        job_status NOT NULL DEFAULT 'pending',
    attempts      INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS attendance_events (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id           UUID REFERENCES students(id),
    candidate_student_id UUID REFERENCES students(id),
    job_id               UUID NOT NULL REFERENCES image_jobs(id),
    camera_id            VARCHAR(64) NOT NULL,
    class_id             VARCHAR(64),
    captured_at          TIMESTAMPTZ NOT NULL,
    confidence           DOUBLE PRECISION,
    status               attendance_status NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fairness_metrics (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_date      TIMESTAMPTZ NOT NULL,
    dataset        VARCHAR(64) NOT NULL,
    race_group     VARCHAR(50) NOT NULL,
    sample_size    INTEGER NOT NULL,
    detection_rate DOUBLE PRECISION,
    precision      DOUBLE PRECISION,
    recall         DOUBLE PRECISION,
    f1             DOUBLE PRECISION,
    threshold_used DOUBLE PRECISION NOT NULL,
    notes          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_students_student_id            ON students(student_id);
CREATE INDEX IF NOT EXISTS idx_students_class_id              ON students(class_id);
CREATE INDEX IF NOT EXISTS idx_student_embeddings_student_id  ON student_embeddings(student_id);
CREATE INDEX IF NOT EXISTS idx_student_embeddings_model_ver   ON student_embeddings(model_version);
CREATE INDEX IF NOT EXISTS idx_student_embeddings_demographic ON student_embeddings(demographic_group);
CREATE INDEX IF NOT EXISTS idx_image_jobs_status_created      ON image_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_image_jobs_device_id           ON image_jobs(device_id);
CREATE INDEX IF NOT EXISTS idx_attendance_events_captured_at  ON attendance_events(captured_at);
CREATE INDEX IF NOT EXISTS idx_attendance_events_student_id   ON attendance_events(student_id);
CREATE INDEX IF NOT EXISTS idx_fairness_metrics_test_date     ON fairness_metrics(test_date);
CREATE INDEX IF NOT EXISTS idx_fairness_metrics_race_group    ON fairness_metrics(race_group);
