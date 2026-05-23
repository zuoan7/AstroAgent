-- 短期记忆摘要快照、删除任务和审计表。
-- summary_snapshot 压缩事件区间，deletion_job/audit_log 记录 tombstone 删除。

CREATE TABLE IF NOT EXISTS summary_snapshot (
    snapshot_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    snapshot_type TEXT NOT NULL,
    covered_from_event_id TEXT,
    covered_to_event_id TEXT,
    summary_text TEXT NOT NULL,
    summary_level TEXT NOT NULL DEFAULT 'working',
    quality_score REAL,
    source_count INTEGER NOT NULL DEFAULT 0,
    created_by_model TEXT,
    created_at REAL NOT NULL,
    superseded_by TEXT,
    is_deleted INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_summary_snapshot_session
    ON summary_snapshot(session_id, created_at, is_deleted);

CREATE TABLE IF NOT EXISTS deletion_job (
    job_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    session_id TEXT,
    delete_scope TEXT NOT NULL,
    selector_json TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_by TEXT,
    requested_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    result_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_deletion_job_session
    ON deletion_job(session_id, requested_at);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    session_id TEXT,
    action_type TEXT NOT NULL,
    target_type TEXT,
    target_id TEXT,
    actor_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_log_session
    ON audit_log(session_id, created_at);
