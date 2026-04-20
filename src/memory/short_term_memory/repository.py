import time
from typing import Any, Dict, Optional

from src.memory.core.models import Message, SalientFact, SessionMemoryState, ToolCallRecord
from src.memory.infrastructure.database import SQLiteRepository
from src.memory.infrastructure.utils import _json_dumps, _json_loads, _utcnow_iso


class ShortTermMemoryRepository(SQLiteRepository):
    def initialize(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS stm_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_timestamp REAL NOT NULL DEFAULT 0.0,
                    trimmed_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stm_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 0,
                    importance_reason TEXT NOT NULL DEFAULT '',
                    message_type TEXT NOT NULL DEFAULT 'chat',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    seq INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES stm_sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_stm_messages_session_time
                    ON stm_messages(session_id, timestamp, id);
                CREATE TABLE IF NOT EXISTS stm_tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL DEFAULT '',
                    tool_name TEXT NOT NULL,
                    input_summary TEXT NOT NULL DEFAULT '',
                    output_digest TEXT NOT NULL DEFAULT '',
                    output_summary TEXT NOT NULL DEFAULT '',
                    output_is_summary INTEGER NOT NULL DEFAULT 0,
                    output_is_truncated INTEGER NOT NULL DEFAULT 0,
                    raw_artifact_id TEXT NOT NULL DEFAULT '',
                    raw_size_bytes INTEGER NOT NULL DEFAULT 0,
                    content_type TEXT NOT NULL DEFAULT 'text/plain',
                    tool_input TEXT NOT NULL DEFAULT '',
                    result_summary TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'success',
                    success INTEGER NOT NULL DEFAULT 1,
                    importance INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY (session_id) REFERENCES stm_sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_stm_tool_calls_session_time
                    ON stm_tool_calls(session_id, timestamp, id);
                CREATE TABLE IF NOT EXISTS stm_salient_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    fact_id TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source_type TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES stm_sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_stm_salient_facts_session_time
                    ON stm_salient_facts(session_id, timestamp, id);
                """
            )
            self._ensure_column(conn, "stm_sessions", "summary_timestamp", "REAL NOT NULL DEFAULT 0.0")
            self._ensure_column(conn, "stm_messages", "message_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_messages", "importance_reason", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_messages", "message_type", "TEXT NOT NULL DEFAULT 'chat'")
            self._ensure_column(conn, "stm_messages", "seq", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "stm_tool_calls", "input_summary", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_tool_calls", "output_summary", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_tool_calls", "output_is_summary", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "stm_tool_calls", "output_is_truncated", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "stm_tool_calls", "tool_input", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_tool_calls", "result_summary", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_tool_calls", "status", "TEXT NOT NULL DEFAULT 'success'")
            self._ensure_column(conn, "stm_tool_calls", "success", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(conn, "stm_tool_calls", "importance", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(conn, "stm_salient_facts", "fact_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_salient_facts", "source_type", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_salient_facts", "source_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_tool_calls", "tool_call_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_tool_calls", "output_digest", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_tool_calls", "raw_artifact_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "stm_tool_calls", "raw_size_bytes", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "stm_tool_calls", "content_type", "TEXT NOT NULL DEFAULT 'text/plain'")

    def _ensure_column(self, conn, table: str, column: str, ddl: str):
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def save_session_state(self, state: SessionMemoryState, user_id: str):
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stm_sessions (session_id, user_id, summary, trimmed_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    summary=excluded.summary,
                    trimmed_count=excluded.trimmed_count,
                    updated_at=excluded.updated_at
                """,
                (state.session_id, user_id, state.summary, state.trimmed_count, now, now),
            )

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            session = cursor.execute(
                "SELECT session_id, user_id, summary, trimmed_count, updated_at FROM stm_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not session:
                return None

            messages = [
                Message(
                    message_id=row["message_id"],
                    session_id=session_id,
                    role=row["role"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    importance=row["importance"],
                    importance_reason=row["importance_reason"],
                    message_type=row["message_type"],
                    metadata=_json_loads(row["metadata"], {}),
                )
                for row in cursor.execute(
                    """
                    SELECT message_id, role, content, timestamp, importance, importance_reason, message_type, metadata
                    FROM stm_messages WHERE session_id = ? ORDER BY seq, timestamp, id
                    """,
                    (session_id,),
                ).fetchall()
            ]

            tool_calls = [
                ToolCallRecord(
                    tool_call_id=row["tool_call_id"] or "",
                    tool_name=row["tool_name"],
                    timestamp=row["timestamp"],
                    input_summary=row["input_summary"],
                    output_digest=row["output_digest"],
                    output_summary=row["output_summary"],
                    output_is_summary=bool(row["output_is_summary"]),
                    output_is_truncated=bool(row["output_is_truncated"]),
                    raw_artifact_id=row["raw_artifact_id"],
                    raw_size_bytes=row["raw_size_bytes"],
                    content_type=row["content_type"] or "text/plain",
                    status=row["status"],
                    importance=row["importance"],
                )
                for row in cursor.execute(
                    """
                    SELECT
                        tool_call_id,
                        tool_name,
                        CASE WHEN input_summary != '' THEN input_summary ELSE tool_input END AS input_summary,
                        output_digest,
                        CASE WHEN output_summary != '' THEN output_summary ELSE result_summary END AS output_summary,
                        output_is_summary,
                        output_is_truncated,
                        raw_artifact_id,
                        raw_size_bytes,
                        content_type,
                        timestamp,
                        CASE
                            WHEN status != '' THEN status
                            WHEN success = 1 THEN 'success'
                            ELSE 'error'
                        END AS status,
                        importance
                    FROM stm_tool_calls WHERE session_id = ? ORDER BY timestamp, id
                    """,
                    (session_id,),
                ).fetchall()
            ]

            salient_facts = [
                SalientFact(
                    fact_id=row["fact_id"],
                    fact_type=row["fact_type"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    source_type=row["source_type"],
                    source_id=row["source_id"],
                    source=row["source"],
                )
                for row in cursor.execute(
                    """
                    SELECT fact_id, fact_type, content, timestamp, source_type, source_id, source
                    FROM stm_salient_facts WHERE session_id = ? ORDER BY timestamp, id
                    """,
                    (session_id,),
                ).fetchall()
            ]

            return {
                "user_id": session["user_id"],
                "state": SessionMemoryState(
                    session_id=session["session_id"],
                    summary=session["summary"] or "",
                    trimmed_count=session["trimmed_count"] or 0,
                    updated_at=time.time(),
                ),
                "messages": messages,
                "tool_calls": tool_calls,
                "salient_facts": salient_facts,
            }

    def append_message(self, session_id: str, message: Message):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stm_messages (
                    session_id, message_id, role, content, timestamp, importance,
                    importance_reason, message_type, metadata, seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    message.message_id,
                    message.role,
                    message.content,
                    message.timestamp,
                    message.importance,
                    message.importance_reason,
                    message.message_type,
                    _json_dumps(message.metadata),
                    0,
                ),
            )

    def append_tool_call(self, session_id: str, tool_call: ToolCallRecord):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stm_tool_calls (
                    session_id, tool_call_id, tool_name, input_summary, output_digest, output_summary,
                    output_is_summary, output_is_truncated, raw_artifact_id, raw_size_bytes, content_type,
                    timestamp, status, importance, tool_input, result_summary, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    tool_call.tool_call_id,
                    tool_call.tool_name,
                    tool_call.input_summary,
                    tool_call.output_digest,
                    tool_call.output_summary,
                    int(tool_call.output_is_summary),
                    int(tool_call.output_is_truncated),
                    tool_call.raw_artifact_id,
                    tool_call.raw_size_bytes,
                    tool_call.content_type,
                    tool_call.timestamp,
                    tool_call.status,
                    tool_call.importance,
                    tool_call.input_summary,
                    tool_call.output_summary,
                    int(tool_call.success),
                ),
            )

    def replace_messages(self, session_id: str, messages: list[Message]):
        with self._connect() as conn:
            conn.execute("DELETE FROM stm_messages WHERE session_id = ?", (session_id,))
            for seq, message in enumerate(messages):
                conn.execute(
                    """
                    INSERT INTO stm_messages (
                        session_id, message_id, role, content, timestamp, importance,
                        importance_reason, message_type, metadata, seq
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        message.message_id,
                        message.role,
                        message.content,
                        message.timestamp,
                        message.importance,
                        message.importance_reason,
                        message.message_type,
                        _json_dumps(message.metadata),
                        seq,
                    ),
                )

    def replace_salient_facts(self, session_id: str, facts: list[SalientFact]):
        with self._connect() as conn:
            conn.execute("DELETE FROM stm_salient_facts WHERE session_id = ?", (session_id,))
            for fact in facts:
                conn.execute(
                    """
                    INSERT INTO stm_salient_facts (
                        session_id, fact_id, fact_type, content, timestamp, source_type, source_id, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        fact.fact_id,
                        fact.fact_type,
                        fact.content,
                        fact.timestamp,
                        fact.source_type,
                        fact.source_id,
                        fact.source,
                    ),
                )

    def update_summary(self, session_id: str, summary: str, trimmed_count: int):
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE stm_sessions
                SET summary = ?, trimmed_count = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (summary, trimmed_count, _utcnow_iso(), session_id),
            )

    def clear_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM stm_sessions WHERE session_id = ?", (session_id,))

    def delete_message(self, session_id: str, message_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM stm_messages WHERE session_id = ? AND message_id = ?",
                (session_id, message_id),
            )
        return cursor.rowcount

    def delete_tool_call(self, session_id: str, tool_call_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM stm_tool_calls WHERE session_id = ? AND tool_call_id = ?",
                (session_id, tool_call_id),
            )
        return cursor.rowcount

    def delete_fact(self, session_id: str, fact_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM stm_salient_facts WHERE session_id = ? AND fact_id = ?",
                (session_id, fact_id),
            )
        return cursor.rowcount

    def delete_by_time_range(
        self,
        session_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> int:
        total = 0
        conditions = ["session_id = ?"]
        params: list[object] = [session_id]
        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        where_clause = " AND ".join(conditions)
        with self._connect() as conn:
            for table in ("stm_messages", "stm_tool_calls", "stm_salient_facts"):
                cursor = conn.execute(f"DELETE FROM {table} WHERE {where_clause}", tuple(params))
                total += cursor.rowcount
        return total

    def load_session_user(self, session_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id FROM stm_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["user_id"] if row else None
