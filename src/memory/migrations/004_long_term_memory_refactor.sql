-- Long-term memory P0 refactor migration.
-- SQLite does not support ADD COLUMN IF NOT EXISTS on older versions, so the
-- repository keeps idempotent runtime guards. This file documents the target
-- schema changes for explicit deployments.

ALTER TABLE memories ADD COLUMN deleted_at TEXT;

ALTER TABLE memory_candidates ADD COLUMN status TEXT NOT NULL DEFAULT 'candidate';
ALTER TABLE memory_candidates ADD COLUMN promoted_memory_id TEXT;
ALTER TABLE memory_candidates ADD COLUMN updated_at TEXT NOT NULL DEFAULT '';
UPDATE memory_candidates SET updated_at = last_seen_at WHERE updated_at = '';

CREATE TABLE IF NOT EXISTS ltm_deletion_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    target_id TEXT,
    reason TEXT NOT NULL DEFAULT '',
    requested_by TEXT NOT NULL DEFAULT 'system',
    deleted_memories INTEGER NOT NULL DEFAULT 0,
    deleted_candidates INTEGER NOT NULL DEFAULT 0,
    deleted_profiles INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ltm_deletion_audit_user_time
    ON ltm_deletion_audit(user_id, created_at DESC);
