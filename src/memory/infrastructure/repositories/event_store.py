from typing import Iterable, Optional

from src.memory.domain.events import MemoryEvent
from src.memory.infrastructure.database import SQLiteRepository
from src.memory.infrastructure.utils import _json_dumps, _json_loads


class EventStore(SQLiteRepository):
    """SQLite append-only store for raw memory events."""

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
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
                """
            )

    def append(self, event: MemoryEvent) -> MemoryEvent:
        """Append an event idempotently by event_id and return the stored event."""

        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_event (
                    event_id, tenant_id, session_id, turn_id, event_type, source_type, source_id,
                    payload_json, schema_version, created_at, created_by, is_deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.tenant_id,
                    event.session_id,
                    event.turn_id,
                    event.event_type,
                    event.source_type,
                    event.source_id,
                    _json_dumps(event.payload),
                    event.schema_version,
                    event.created_at,
                    event.created_by,
                    int(event.is_deleted),
                ),
            )
        stored = self.get(event.event_id)
        return stored or event

    def get(self, event_id: str) -> Optional[MemoryEvent]:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT event_id, tenant_id, session_id, turn_id, event_type, source_type, source_id,
                       payload_json, schema_version, created_at, created_by, is_deleted
                FROM memory_event WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return self._from_row(row) if row else None

    def list_by_session(
        self,
        session_id: str,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        include_deleted: bool = False,
        limit: int = 500,
    ) -> list[MemoryEvent]:
        self.initialize()
        conditions = ["session_id = ?"]
        params: list[object] = [session_id]
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if start_time is not None:
            conditions.append("created_at >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("created_at <= ?")
            params.append(end_time)
        if not include_deleted:
            conditions.append("is_deleted = 0")
        params.append(limit)
        sql = f"""
            SELECT event_id, tenant_id, session_id, turn_id, event_type, source_type, source_id,
                   payload_json, schema_version, created_at, created_by, is_deleted
            FROM memory_event
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at, id
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._from_row(row) for row in rows]

    def list_by_source(
        self,
        session_id: str,
        source_type: str,
        source_id: str,
        include_deleted: bool = False,
    ) -> list[MemoryEvent]:
        self.initialize()
        condition = "" if include_deleted else " AND is_deleted = 0"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT event_id, tenant_id, session_id, turn_id, event_type, source_type, source_id,
                       payload_json, schema_version, created_at, created_by, is_deleted
                FROM memory_event
                WHERE session_id = ? AND source_type = ? AND source_id = ?{condition}
                ORDER BY created_at, id
                """,
                (session_id, source_type, source_id),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def mark_deleted_by_session(
        self,
        session_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> int:
        self.initialize()
        conditions = ["session_id = ?"]
        params: list[object] = [session_id]
        if start_time is not None:
            conditions.append("created_at >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("created_at <= ?")
            params.append(end_time)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE memory_event SET is_deleted = 1 WHERE {' AND '.join(conditions)}",
                tuple(params),
            )
        return cursor.rowcount

    def mark_deleted_by_source(self, session_id: str, source_type: str, source_id: str) -> int:
        self.initialize()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_event
                SET is_deleted = 1
                WHERE session_id = ? AND source_type = ? AND source_id = ?
                """,
                (session_id, source_type, source_id),
            )
        return cursor.rowcount

    def mark_deleted(self, event_ids: Iterable[str]) -> int:
        ids = list(event_ids)
        if not ids:
            return 0
        self.initialize()
        with self._connect() as conn:
            count = 0
            for event_id in ids:
                cursor = conn.execute("UPDATE memory_event SET is_deleted = 1 WHERE event_id = ?", (event_id,))
                count += cursor.rowcount
        return count

    def _from_row(self, row) -> MemoryEvent:
        return MemoryEvent(
            event_id=row["event_id"],
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            event_type=row["event_type"],
            source_type=row["source_type"],
            source_id=row["source_id"],
            payload=_json_loads(row["payload_json"], {}),
            schema_version=row["schema_version"],
            created_at=row["created_at"],
            created_by=row["created_by"],
            is_deleted=bool(row["is_deleted"]),
        )
