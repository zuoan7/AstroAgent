from typing import Optional

from src.memory.domain.task_state import TaskState
from src.memory.infrastructure.database import SQLiteRepository
from src.memory.infrastructure.utils import _json_dumps, _json_loads


class TaskStateRepository(SQLiteRepository):
    """Repository for the current structured task state projection."""

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
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
                """
            )

    def get(self, session_id: str, include_deleted: bool = False) -> Optional[TaskState]:
        self.initialize()
        condition = "" if include_deleted else " AND is_deleted = 0"
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT task_state_id, tenant_id, session_id, current_goal, active_constraints,
                       completed_steps, pending_steps, open_questions, assumptions, blockers,
                       next_action, status, confidence, version, updated_from_event_id, updated_at, is_deleted
                FROM task_state
                WHERE session_id = ?{condition}
                """,
                (session_id,),
            ).fetchone()
        return self._from_row(row) if row else None

    def save(self, state: TaskState) -> TaskState:
        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_state (
                    task_state_id, tenant_id, session_id, current_goal, active_constraints,
                    completed_steps, pending_steps, open_questions, assumptions, blockers,
                    next_action, status, confidence, version, updated_from_event_id, updated_at, is_deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    current_goal=excluded.current_goal,
                    active_constraints=excluded.active_constraints,
                    completed_steps=excluded.completed_steps,
                    pending_steps=excluded.pending_steps,
                    open_questions=excluded.open_questions,
                    assumptions=excluded.assumptions,
                    blockers=excluded.blockers,
                    next_action=excluded.next_action,
                    status=excluded.status,
                    confidence=excluded.confidence,
                    version=excluded.version,
                    updated_from_event_id=excluded.updated_from_event_id,
                    updated_at=excluded.updated_at,
                    is_deleted=excluded.is_deleted
                """
                ,
                (
                    state.task_state_id,
                    state.tenant_id,
                    state.session_id,
                    state.current_goal,
                    _json_dumps(state.active_constraints),
                    _json_dumps(state.completed_steps),
                    _json_dumps(state.pending_steps),
                    _json_dumps(state.open_questions),
                    _json_dumps(state.assumptions),
                    _json_dumps(state.blockers),
                    state.next_action,
                    state.status,
                    state.confidence,
                    state.version,
                    state.updated_from_event_id,
                    state.updated_at,
                    int(state.is_deleted),
                ),
            )
        return state

    def mark_deleted(self, session_id: str) -> bool:
        self.initialize()
        with self._connect() as conn:
            cursor = conn.execute("UPDATE task_state SET is_deleted = 1 WHERE session_id = ?", (session_id,))
        return cursor.rowcount > 0

    def _from_row(self, row) -> TaskState:
        return TaskState(
            task_state_id=row["task_state_id"],
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            current_goal=row["current_goal"] or "",
            active_constraints=_json_loads(row["active_constraints"], []),
            completed_steps=_json_loads(row["completed_steps"], []),
            pending_steps=_json_loads(row["pending_steps"], []),
            open_questions=_json_loads(row["open_questions"], []),
            assumptions=_json_loads(row["assumptions"], []),
            blockers=_json_loads(row["blockers"], []),
            next_action=row["next_action"] or "",
            status=row["status"] or "active",
            confidence=row["confidence"],
            version=row["version"],
            updated_from_event_id=row["updated_from_event_id"],
            updated_at=row["updated_at"],
            is_deleted=bool(row["is_deleted"]),
        )
