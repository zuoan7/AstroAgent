import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from src.core.logger import logger
from src.memory.long_term_memory.models import (
    MemoryEvent,
    EventLogEntry,
    MemoryCandidate,
    MemoryConfirmation,
    MemoryItem,
    MemoryQuery,
    MemoryVersion,
    _json_loads,
    _utcnow_iso,
)


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


class LongTermMemoryRepository:
    SCHEMA_VERSION = 3

    def __init__(self, db_path: str):
        self.db_path = db_path
        _ensure_parent_dir(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    source_type TEXT NOT NULL DEFAULT 'auto',
                    source_conversation_id TEXT,
                    source_content_snippet TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    priority INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    accessed_at TEXT,
                    expires_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    confirmation_count INTEGER NOT NULL DEFAULT 0,
                    confirmed_by_user INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user_type
                    ON memories(user_id, memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_user_status
                    ON memories(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_memories_user_category
                    ON memories(user_id, category);
                CREATE INDEX IF NOT EXISTS idx_memories_user_updated
                    ON memories(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_user_type_key
                    ON memories(user_id, memory_type, key);

                CREATE TABLE IF NOT EXISTS memory_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL,
                    change_reason TEXT,
                    changed_at TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_memory_versions_memory
                    ON memory_versions(memory_id, version DESC);

                CREATE TABLE IF NOT EXISTS memory_candidates (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.3,
                    source_type TEXT NOT NULL DEFAULT 'auto',
                    source_conversation_id TEXT,
                    source_content_snippet TEXT,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'candidate',
                    promoted_memory_id TEXT,
                    updated_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_candidates_user_type
                    ON memory_candidates(user_id, memory_type);
                CREATE INDEX IF NOT EXISTS idx_candidates_user_key
                    ON memory_candidates(user_id, memory_type, key);

                CREATE TABLE IF NOT EXISTS memory_event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_id TEXT,
                    event_type TEXT NOT NULL,
                    event_detail TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_event_log_user_time
                    ON memory_event_log(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_event_log_memory
                    ON memory_event_log(memory_id);

                CREATE TABLE IF NOT EXISTS memory_confirmations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_id TEXT,
                    confirmation_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_confirmations_user_status
                    ON memory_confirmations(user_id, status);

                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL DEFAULT '{}',
                    habits TEXT NOT NULL DEFAULT '{}',
                    constraints TEXT NOT NULL DEFAULT '[]',
                    background TEXT NOT NULL DEFAULT '{}',
                    facts TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

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

                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source_text TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    last_confirmed_at TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_events_user_time
                    ON memory_events(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_events_user_status
                    ON memory_events(user_id, status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_events_user_key
                    ON memory_events(user_id, key, created_at DESC);

                CREATE TABLE IF NOT EXISTS schema_version (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (key, value) VALUES (?, ?)",
                ("version", str(self.SCHEMA_VERSION)),
            )
            self._ensure_legacy_migration(conn)

    def _ensure_legacy_migration(self, conn: sqlite3.Connection):
        if self._table_exists(conn, "user_profiles"):
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(user_profiles)").fetchall()}
            if "background" not in cols:
                conn.execute("ALTER TABLE user_profiles ADD COLUMN background TEXT NOT NULL DEFAULT '{}'")
            if "facts" not in cols:
                conn.execute("ALTER TABLE user_profiles ADD COLUMN facts TEXT NOT NULL DEFAULT '[]'")
        if self._table_exists(conn, "memory_events"):
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(memory_events)").fetchall()}
            required = {
                "key": "TEXT NOT NULL DEFAULT ''",
                "value": "TEXT NOT NULL DEFAULT '\"\"'",
                "source_text": "TEXT NOT NULL DEFAULT ''",
                "status": "TEXT NOT NULL DEFAULT 'active'",
                "confidence": "REAL NOT NULL DEFAULT 0.5",
                "last_confirmed_at": "TEXT",
                "metadata": "TEXT NOT NULL DEFAULT '{}'",
                "updated_at": "TEXT NOT NULL DEFAULT ''",
            }
            for name, ddl in required.items():
                if name not in cols:
                    conn.execute(f"ALTER TABLE memory_events ADD COLUMN {name} {ddl}")
        if self._table_exists(conn, "memories"):
            self._ensure_columns(conn, "memories", {"deleted_at": "TEXT"})
        if self._table_exists(conn, "memory_candidates"):
            self._ensure_columns(
                conn,
                "memory_candidates",
                {
                    "status": "TEXT NOT NULL DEFAULT 'candidate'",
                    "promoted_memory_id": "TEXT",
                    "updated_at": "TEXT NOT NULL DEFAULT ''",
                },
            )
            conn.execute(
                "UPDATE memory_candidates SET updated_at=last_seen_at WHERE updated_at=''"
            )

    def _ensure_columns(self, conn: sqlite3.Connection, table_name: str, columns: Dict[str, str]):
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {ddl}")

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    def add_memory(self, item: MemoryItem) -> MemoryItem:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, user_id, memory_type, category, key, value,
                    confidence, source_type, source_conversation_id,
                    source_content_snippet, status, priority, metadata,
                    created_at, updated_at, accessed_at, expires_at,
                    access_count, confirmation_count, confirmed_by_user, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id, item.user_id, item.memory_type, item.category,
                    item.key, item.to_db_row()["value"],
                    item.confidence, item.source_type, item.source_conversation_id,
                    item.source_content_snippet, item.status, item.priority,
                    item.to_db_row()["metadata"],
                    item.created_at, item.updated_at, item.accessed_at,
                    item.expires_at, item.access_count, item.confirmation_count,
                    int(item.confirmed_by_user), item.deleted_at,
                ),
            )
        return item

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ? AND status != 'deleted'", (memory_id,)
            ).fetchone()
        if not row:
            return None
        return MemoryItem.from_db_row(row)

    def update_memory(self, item: MemoryItem) -> bool:
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories SET
                    memory_type=?, category=?, key=?, value=?,
                    confidence=?, source_type=?, source_conversation_id=?,
                    source_content_snippet=?, status=?, priority=?, metadata=?,
                    updated_at=?, accessed_at=?, expires_at=?,
                    access_count=?, confirmation_count=?, confirmed_by_user=?,
                    deleted_at=?
                WHERE id=? AND user_id=?
                """,
                (
                    item.memory_type, item.category, item.key,
                    item.to_db_row()["value"],
                    item.confidence, item.source_type, item.source_conversation_id,
                    item.source_content_snippet, item.status, item.priority,
                    item.to_db_row()["metadata"],
                    now, item.accessed_at, item.expires_at,
                    item.access_count, item.confirmation_count,
                    int(item.confirmed_by_user), item.deleted_at,
                    item.id, item.user_id,
                ),
            )
            return cursor.rowcount > 0

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET status='deleted', deleted_at=?, updated_at=? WHERE id=? AND user_id=? AND status!='deleted'",
                (_utcnow_iso(), _utcnow_iso(), memory_id, user_id),
            )
            return cursor.rowcount > 0

    def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        where_sql, params, order_clause = query.to_where_clause()
        if not query.status:
            where_sql = f"{where_sql} AND status != 'deleted'"
        sql = f"SELECT * FROM memories WHERE {where_sql} ORDER BY {order_clause} LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [MemoryItem.from_db_row(row) for row in rows]

    def count_memories(self, query: MemoryQuery) -> int:
        where_sql, params, _ = query.to_where_clause()
        sql = f"SELECT COUNT(*) as cnt FROM memories WHERE {where_sql}"
        if not query.status:
            sql = f"SELECT COUNT(*) as cnt FROM memories WHERE {where_sql} AND status != 'deleted'"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return row["cnt"] if row else 0

    def find_memory_by_type_key(
        self, user_id: str, memory_type: str, key: str, status: str = "active"
    ) -> Optional[MemoryItem]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE user_id=? AND memory_type=? AND key=? AND status=? AND status!='deleted' LIMIT 1",
                (user_id, memory_type, key, status),
            ).fetchone()
        if not row:
            return None
        return MemoryItem.from_db_row(row)

    def find_similar_memories(
        self, user_id: str, memory_type: str, category: str, value: str, status: str = "active"
    ) -> List[MemoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id=? AND memory_type=? AND category=? AND status=? AND status!='deleted' AND value LIKE ?",
                (user_id, memory_type, category, status, f"%{value}%"),
            ).fetchall()
        return [MemoryItem.from_db_row(row) for row in rows]

    def increment_access_count(self, memory_id: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET access_count=access_count+1, accessed_at=? WHERE id=?",
                (_utcnow_iso(), memory_id),
            )

    def add_version(self, memory_id: str, version: int, value: Any, confidence: float, change_reason: str):
        value_str = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory_versions (memory_id, version, value, confidence, change_reason, changed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, version, value_str, confidence, change_reason, _utcnow_iso()),
            )

    def get_versions(self, memory_id: str, limit: int = 20) -> List[MemoryVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_versions WHERE memory_id=? ORDER BY version DESC LIMIT ?",
                (memory_id, limit),
            ).fetchall()
        return [
            MemoryVersion(
                id=row["id"],
                memory_id=row["memory_id"],
                version=row["version"],
                value=_json_loads(row["value"]),
                confidence=row["confidence"],
                change_reason=row["change_reason"],
                changed_at=row["changed_at"],
            )
            for row in rows
        ]

    def add_candidate(self, candidate: MemoryCandidate) -> MemoryCandidate:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_candidates (
                    id, user_id, memory_type, category, key, value,
                    confidence, source_type, source_conversation_id,
                    source_content_snippet, occurrence_count,
                    first_seen_at, last_seen_at, metadata, created_at,
                    status, promoted_memory_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id, candidate.user_id, candidate.memory_type,
                    candidate.category, candidate.key, candidate.to_db_row()["value"],
                    candidate.confidence, candidate.source_type,
                    candidate.source_conversation_id, candidate.source_content_snippet,
                    candidate.occurrence_count, candidate.first_seen_at,
                    candidate.last_seen_at, candidate.to_db_row()["metadata"],
                    candidate.created_at, candidate.status,
                    candidate.promoted_memory_id, candidate.updated_at,
                ),
            )
        return candidate

    def get_candidate(self, candidate_id: str) -> Optional[MemoryCandidate]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_candidates WHERE id=? AND status != 'deleted'", (candidate_id,)
            ).fetchone()
        if not row:
            return None
        return MemoryCandidate.from_db_row(row)

    def find_candidate_by_type_key(
        self, user_id: str, memory_type: str, key: str
    ) -> Optional[MemoryCandidate]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_candidates WHERE user_id=? AND memory_type=? AND key=? AND status='candidate' LIMIT 1",
                (user_id, memory_type, key),
            ).fetchone()
        if not row:
            return None
        return MemoryCandidate.from_db_row(row)

    def update_candidate(self, candidate: MemoryCandidate) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_candidates SET
                    value=?, confidence=?, occurrence_count=?,
                    last_seen_at=?, metadata=?, status=?,
                    promoted_memory_id=?, updated_at=?
                WHERE id=?
                """,
                (
                    candidate.to_db_row()["value"],
                    candidate.confidence,
                    candidate.occurrence_count,
                    candidate.last_seen_at,
                    candidate.to_db_row()["metadata"],
                    candidate.status,
                    candidate.promoted_memory_id,
                    candidate.updated_at or _utcnow_iso(),
                    candidate.id,
                ),
            )
            return cursor.rowcount > 0

    def delete_candidate(self, candidate_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memory_candidates SET status='deleted', updated_at=? WHERE id=? AND status!='deleted'",
                (_utcnow_iso(), candidate_id),
            )
            return cursor.rowcount > 0

    def mark_candidate_promoted(self, candidate_id: str, memory_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_candidates
                SET status='promoted', promoted_memory_id=?, updated_at=?
                WHERE id=? AND status!='deleted'
                """,
                (memory_id, _utcnow_iso(), candidate_id),
            )
            return cursor.rowcount > 0

    def list_candidates(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[MemoryCandidate]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_candidates WHERE user_id=? AND status='candidate' ORDER BY last_seen_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            ).fetchall()
        return [MemoryCandidate.from_db_row(row) for row in rows]

    def add_event_log(self, entry: EventLogEntry) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memory_event_log (
                    user_id, memory_id, event_type, event_detail,
                    old_value, new_value, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.user_id, entry.memory_id, entry.event_type,
                    entry.event_detail, entry.old_value, entry.new_value,
                    entry.to_db_row()["metadata"], entry.created_at,
                ),
            )
            return cursor.lastrowid

    def add_event(self, event: MemoryEvent) -> MemoryEvent:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_events (
                    event_id, user_id, event_type, key, value, source_text, status,
                    confidence, last_confirmed_at, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.user_id,
                    event.event_type,
                    event.key,
                    event.to_db_row()["value"],
                    event.source_text,
                    event.status,
                    event.confidence,
                    event.last_confirmed_at,
                    event.to_db_row()["metadata"],
                    event.created_at,
                    event.updated_at,
                ),
            )
        return event

    def add_events(self, events: List[MemoryEvent]) -> List[MemoryEvent]:
        if not events:
            return []
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO memory_events (
                    event_id, user_id, event_type, key, value, source_text, status,
                    confidence, last_confirmed_at, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event.event_id,
                        event.user_id,
                        event.event_type,
                        event.key,
                        event.to_db_row()["value"],
                        event.source_text,
                        event.status,
                        event.confidence,
                        event.last_confirmed_at,
                        event.to_db_row()["metadata"],
                        event.created_at,
                        event.updated_at,
                    )
                    for event in events
                ],
            )
        return events

    def get_recent_events(self, user_id: str, limit: int = 10) -> List[MemoryEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_events
                WHERE user_id=? ORDER BY created_at DESC, updated_at DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [MemoryEvent.from_db_row(row) for row in rows]

    def get_candidate_events(self, user_id: str) -> List[MemoryEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_events
                WHERE user_id=? AND status='candidate'
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [MemoryEvent.from_db_row(row) for row in rows]

    def count_similar_events(self, user_id: str, key: str, value: Any) -> int:
        value_str = json.dumps(value, ensure_ascii=False)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM memory_events
                WHERE user_id=? AND key=? AND value=? AND status IN ('candidate', 'active')
                """,
                (user_id, key, value_str),
            ).fetchone()
        return row["cnt"] if row else 0

    def update_event_status(self, event_id: str, status: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memory_events SET status=?, updated_at=? WHERE event_id=?",
                (status, _utcnow_iso(), event_id),
            )
            return cursor.rowcount > 0

    def update_event_confidence(self, event_id: str, confidence: float) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memory_events SET confidence=?, updated_at=? WHERE event_id=?",
                (confidence, _utcnow_iso(), event_id),
            )
            return cursor.rowcount > 0

    def confirm_event(self, event_id: str) -> bool:
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_events
                SET status='active', last_confirmed_at=?, updated_at=?
                WHERE event_id=?
                """,
                (now, now, event_id),
            )
            return cursor.rowcount > 0

    def get_active_events(self, user_id: str, limit: int = 100) -> List[MemoryEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_events
                WHERE user_id=? AND status='active'
                ORDER BY confidence DESC, COALESCE(last_confirmed_at, created_at) DESC, created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [MemoryEvent.from_db_row(row) for row in rows]

    def get_event_logs(
        self, user_id: str, memory_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[EventLogEntry]:
        if memory_id:
            sql = "SELECT * FROM memory_event_log WHERE user_id=? AND memory_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params: list = [user_id, memory_id, limit, offset]
        else:
            sql = "SELECT * FROM memory_event_log WHERE user_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params = [user_id, limit, offset]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            EventLogEntry(
                id=row["id"],
                user_id=row["user_id"],
                memory_id=row["memory_id"],
                event_type=row["event_type"],
                event_detail=row["event_detail"],
                old_value=row["old_value"],
                new_value=row["new_value"],
                metadata=_json_loads(row["metadata"], {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def add_confirmation(self, confirmation: MemoryConfirmation) -> MemoryConfirmation:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_confirmations (
                    id, user_id, memory_id, confirmation_type,
                    content, status, created_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    confirmation.id, confirmation.user_id, confirmation.memory_id,
                    confirmation.confirmation_type, confirmation.content,
                    confirmation.status, confirmation.created_at, confirmation.resolved_at,
                ),
            )
        return confirmation

    def get_confirmation(self, confirmation_id: str) -> Optional[MemoryConfirmation]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_confirmations WHERE id=?", (confirmation_id,)
            ).fetchone()
        if not row:
            return None
        return MemoryConfirmation.from_db_row(row)

    def update_confirmation_status(
        self, confirmation_id: str, status: str, resolved_at: Optional[str] = None
    ) -> bool:
        resolved = resolved_at or _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memory_confirmations SET status=?, resolved_at=? WHERE id=?",
                (status, resolved, confirmation_id),
            )
            return cursor.rowcount > 0

    def list_pending_confirmations(
        self, user_id: str, limit: int = 20
    ) -> List[MemoryConfirmation]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_confirmations WHERE user_id=? AND status='pending' ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [MemoryConfirmation.from_db_row(row) for row in rows]

    def load_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id=?", (user_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "user_id": row["user_id"],
            "preferences": _json_loads(row["preferences"], {}),
            "habits": _json_loads(row["habits"], {}),
            "constraints": _json_loads(row["constraints"], []),
            "background": _json_loads(row["background"], {}),
            "facts": _json_loads(row["facts"], []),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def save_profile(self, user_id: str, preferences: Dict, habits: Dict, constraints: list, background: Dict, facts: list):
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id, preferences, habits, constraints, background, facts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferences=excluded.preferences,
                    habits=excluded.habits,
                    constraints=excluded.constraints,
                    background=excluded.background,
                    facts=excluded.facts,
                    updated_at=excluded.updated_at
                """,
                (
                    user_id,
                    json.dumps(preferences, ensure_ascii=False),
                    json.dumps(habits, ensure_ascii=False),
                    json.dumps(constraints, ensure_ascii=False),
                    json.dumps(background, ensure_ascii=False),
                    json.dumps(facts, ensure_ascii=False),
                    now, now,
                ),
            )

    def delete_profile(self, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM user_profiles WHERE user_id=?", (user_id,))
            return cursor.rowcount > 0

    def tombstone_user_memories(self, user_id: str, reason: str = "") -> int:
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET status='deleted', deleted_at=?, updated_at=?,
                    metadata=?
                WHERE user_id=? AND status!='deleted'
                """,
                (now, now, json.dumps({"delete_reason": reason}, ensure_ascii=False), user_id),
            )
            return cursor.rowcount

    def tombstone_memory(self, user_id: str, memory_id: str, reason: str = "") -> int:
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET status='deleted', deleted_at=?, updated_at=?,
                    metadata=?
                WHERE user_id=? AND id=? AND status!='deleted'
                """,
                (now, now, json.dumps({"delete_reason": reason}, ensure_ascii=False), user_id, memory_id),
            )
            return cursor.rowcount

    def tombstone_candidate(self, user_id: str, candidate_id: str, reason: str = "") -> int:
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_candidates
                SET status='deleted', updated_at=?,
                    metadata=?
                WHERE user_id=? AND id=? AND status!='deleted'
                """,
                (now, json.dumps({"delete_reason": reason}, ensure_ascii=False), user_id, candidate_id),
            )
            return cursor.rowcount

    def tombstone_user_candidates(self, user_id: str, reason: str = "") -> int:
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_candidates
                SET status='deleted', updated_at=?,
                    metadata=?
                WHERE user_id=? AND status!='deleted'
                """,
                (now, json.dumps({"delete_reason": reason}, ensure_ascii=False), user_id),
            )
            return cursor.rowcount

    def mark_legacy_events_deleted(self, user_id: str, reason: str = "") -> int:
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_events
                SET status='deleted', updated_at=?,
                    metadata=?
                WHERE user_id=? AND status!='deleted'
                """,
                (now, json.dumps({"delete_reason": reason}, ensure_ascii=False), user_id),
            )
            return cursor.rowcount

    def add_deletion_audit(
        self,
        user_id: str,
        scope: str,
        target_id: Optional[str],
        reason: str,
        requested_by: str,
        deleted_memories: int,
        deleted_candidates: int,
        deleted_profiles: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ltm_deletion_audit (
                    user_id, scope, target_id, reason, requested_by,
                    deleted_memories, deleted_candidates, deleted_profiles,
                    metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    scope,
                    target_id,
                    reason,
                    requested_by,
                    deleted_memories,
                    deleted_candidates,
                    deleted_profiles,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    _utcnow_iso(),
                ),
            )
            return cursor.lastrowid

    def list_deletion_audit(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ltm_deletion_audit
                WHERE user_id=? ORDER BY created_at DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "scope": row["scope"],
                "target_id": row["target_id"],
                "reason": row["reason"],
                "requested_by": row["requested_by"],
                "deleted_memories": row["deleted_memories"],
                "deleted_candidates": row["deleted_candidates"],
                "deleted_profiles": row["deleted_profiles"],
                "metadata": _json_loads(row["metadata"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def expire_old_memories(self, user_id: str, before_iso: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET status='expired', updated_at=? WHERE user_id=? AND expires_at IS NOT NULL AND expires_at < ? AND status='active'",
                (_utcnow_iso(), user_id, before_iso),
            )
            return cursor.rowcount

    def archive_unused_memories(self, user_id: str, max_access_count: int = 0, days_unused: int = 90) -> int:
        from datetime import timedelta
        cutoff = (_utcnow_iso()[:10] if len(_utcnow_iso()) >= 10 else _utcnow_iso())
        try:
            from datetime import datetime
            cutoff_dt = datetime.fromisoformat(_utcnow_iso()) - timedelta(days=days_unused)
            cutoff = cutoff_dt.isoformat()
        except Exception:
            pass
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET status='archived', updated_at=? WHERE user_id=? AND access_count<=? AND accessed_at IS NOT NULL AND accessed_at<? AND status='active'",
                (_utcnow_iso(), user_id, max_access_count, cutoff),
            )
            return cursor.rowcount

    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            type_counts = {}
            for row in conn.execute(
                "SELECT memory_type, status, COUNT(*) as cnt FROM memories WHERE user_id=? AND status!='deleted' GROUP BY memory_type, status",
                (user_id,),
            ).fetchall():
                key = f"{row['memory_type']}_{row['status']}"
                type_counts[key] = row["cnt"]

            candidate_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM memory_candidates WHERE user_id=? AND status='candidate'", (user_id,)
            ).fetchone()["cnt"]

            confirmation_pending = conn.execute(
                "SELECT COUNT(*) as cnt FROM memory_confirmations WHERE user_id=? AND status='pending'", (user_id,)
            ).fetchone()["cnt"]

            avg_confidence_row = conn.execute(
                "SELECT AVG(confidence) as avg_conf FROM memories WHERE user_id=? AND status='active'", (user_id,)
            ).fetchone()

        return {
            "type_counts": type_counts,
            "candidate_count": candidate_count,
            "pending_confirmations": confirmation_pending,
            "avg_confidence": round(avg_confidence_row["avg_conf"] or 0, 3),
        }

    def backup_database(self, backup_path: str) -> bool:
        _ensure_parent_dir(backup_path)
        try:
            import shutil
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"数据库备份成功: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            return False

    def restore_from_backup(self, backup_path: str) -> bool:
        try:
            import shutil
            if not os.path.exists(backup_path):
                logger.error(f"备份文件不存在: {backup_path}")
                return False
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"数据库恢复成功: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
            return False
