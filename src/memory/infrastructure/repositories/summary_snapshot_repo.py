"""summary snapshot 仓储。

保存短期记忆摘要快照及其覆盖事件范围，供上下文构建和维护服务读取。
"""

from typing import Optional

from src.memory.domain.summary_snapshot import SummarySnapshot
from src.memory.infrastructure.database import SQLiteRepository


class SummarySnapshotRepository(SQLiteRepository):
    """Repository for summary snapshot projection records."""

    def initialize(self) -> None:
        """创建 summary_snapshot 表和会话索引。"""

        with self._connect() as conn:
            conn.executescript(
                """
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
                """
            )

    def save(self, snapshot: SummarySnapshot) -> SummarySnapshot:
        """插入或更新一条摘要快照。"""

        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO summary_snapshot (
                    snapshot_id, tenant_id, session_id, snapshot_type, covered_from_event_id,
                    covered_to_event_id, summary_text, summary_level, quality_score, source_count,
                    created_by_model, created_at, superseded_by, is_deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    summary_text=excluded.summary_text,
                    quality_score=excluded.quality_score,
                    source_count=excluded.source_count,
                    superseded_by=excluded.superseded_by,
                    is_deleted=excluded.is_deleted
                """
                ,
                (
                    snapshot.snapshot_id,
                    snapshot.tenant_id,
                    snapshot.session_id,
                    snapshot.snapshot_type,
                    snapshot.covered_from_event_id,
                    snapshot.covered_to_event_id,
                    snapshot.summary_text,
                    snapshot.summary_level,
                    snapshot.quality_score,
                    snapshot.source_count,
                    snapshot.created_by_model,
                    snapshot.created_at,
                    snapshot.superseded_by,
                    int(snapshot.is_deleted),
                ),
            )
        return snapshot

    def get_latest(
        self,
        session_id: str,
        snapshot_type: str = "working",
        include_deleted: bool = False,
    ) -> Optional[SummarySnapshot]:
        """读取指定会话和快照类型下最新的未删除快照。"""

        self.initialize()
        condition = "" if include_deleted else " AND is_deleted = 0"
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT snapshot_id, tenant_id, session_id, snapshot_type, covered_from_event_id,
                       covered_to_event_id, summary_text, summary_level, quality_score, source_count,
                       created_by_model, created_at, superseded_by, is_deleted
                FROM summary_snapshot
                WHERE session_id = ? AND snapshot_type = ?{condition}
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id, snapshot_type),
            ).fetchone()
        return self._from_row(row) if row else None

    def mark_superseded(self, old_snapshot_id: str, new_snapshot_id: str) -> bool:
        """标记旧快照已被新快照替代。"""

        self.initialize()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE summary_snapshot SET superseded_by = ? WHERE snapshot_id = ?",
                (new_snapshot_id, old_snapshot_id),
            )
        return cursor.rowcount > 0

    def mark_deleted_by_session(self, session_id: str) -> int:
        """按会话 tombstone 标记所有摘要快照。"""

        self.initialize()
        with self._connect() as conn:
            cursor = conn.execute("UPDATE summary_snapshot SET is_deleted = 1 WHERE session_id = ?", (session_id,))
        return cursor.rowcount

    def _from_row(self, row) -> SummarySnapshot:
        """把 SQLite Row 反序列化为 SummarySnapshot。"""

        return SummarySnapshot(
            snapshot_id=row["snapshot_id"],
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            snapshot_type=row["snapshot_type"],
            covered_from_event_id=row["covered_from_event_id"],
            covered_to_event_id=row["covered_to_event_id"],
            summary_text=row["summary_text"],
            summary_level=row["summary_level"],
            quality_score=row["quality_score"],
            source_count=row["source_count"],
            created_by_model=row["created_by_model"],
            created_at=row["created_at"],
            superseded_by=row["superseded_by"],
            is_deleted=bool(row["is_deleted"]),
        )
