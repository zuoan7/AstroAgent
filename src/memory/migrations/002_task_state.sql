-- 短期记忆任务状态投影表。
-- 每个会话保留一份当前任务目标、约束、进度和阻塞的结构化状态。

CREATE TABLE IF NOT EXISTS task_state (
    task_state_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    session_id TEXT NOT NULL UNIQUE,
    current_goal TEXT,
    active_constraints TEXT NOT NULL DEFAULT '[]',
    completed_steps TEXT NOT NULL DEFAULT '[]',
    pending_steps TEXT NOT NULL DEFAULT '[]',
    open_questions TEXT NOT NULL DEFAULT '[]',
    assumptions TEXT NOT NULL DEFAULT '[]',
    blockers TEXT NOT NULL DEFAULT '[]',
    next_action TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL,
    version INTEGER NOT NULL DEFAULT 1,
    updated_from_event_id TEXT,
    updated_at REAL NOT NULL,
    is_deleted INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_task_state_session
    ON task_state(session_id, is_deleted);
