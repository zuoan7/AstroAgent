CREATE TABLE IF NOT EXISTS memory_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT,
    event_type TEXT NOT NULL,
    source_type TEXT,
    source_id TEXT,
    payload_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    created_by TEXT,
    is_deleted INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memory_event_session_time
    ON memory_event(session_id, created_at, id);

CREATE INDEX IF NOT EXISTS idx_memory_event_type
    ON memory_event(event_type, created_at);

CREATE TABLE IF NOT EXISTS tool_artifact (
    artifact_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    storage_uri TEXT NOT NULL,
    content_type TEXT,
    encoding TEXT,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    compression_codec TEXT,
    schema_name TEXT,
    schema_version INTEGER,
    preview_text TEXT,
    raw_content TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL,
    is_deleted INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tool_artifact_session
    ON tool_artifact(session_id, tool_call_id);

CREATE INDEX IF NOT EXISTS idx_tool_artifact_hash
    ON tool_artifact(tenant_id, session_id, tool_call_id, sha256);
