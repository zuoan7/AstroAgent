"""短期记忆工具 artifact 仓储。

工具原始输出保存在 tool_artifact 表中，EventStore 只保存摘要和 artifact_id，
以控制 prompt 大小并保留完整调试材料。
"""

import hashlib
import time
from typing import Optional

from src.memory.domain.artifacts import ToolArtifact
from src.memory.domain.events import new_memory_id
from src.memory.infrastructure.database import SQLiteRepository


class ArtifactStore(SQLiteRepository):
    """SQLite artifact store for raw tool outputs.

    P0 stores raw content in SQLite directly. The storage_uri keeps the interface
    ready for a future object store implementation.
    """

    def initialize(self) -> None:
        """创建工具 artifact 表和会话/哈希索引。"""

        with self._connect() as conn:
            conn.executescript(
                """
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
                """
            )

    def put(
        self,
        tenant_id: str,
        session_id: str,
        tool_call_id: str,
        raw_content: str,
        content_type: str = "text/plain",
        encoding: str = "utf-8",
    ) -> ToolArtifact:
        """Store raw content and return its artifact metadata."""

        self.initialize()
        text = raw_content or ""
        data = text.encode(encoding or "utf-8", errors="replace")
        digest = hashlib.sha256(data).hexdigest()
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT artifact_id, tenant_id, session_id, tool_call_id, storage_uri, content_type, encoding,
                       size_bytes, sha256, compression_codec, schema_name, schema_version, preview_text,
                       created_at, expires_at, is_deleted
                FROM tool_artifact
                WHERE tenant_id = ? AND session_id = ? AND tool_call_id = ? AND sha256 = ? AND is_deleted = 0
                """,
                (tenant_id, session_id, tool_call_id, digest),
            ).fetchone()
            if existing:
                return self._artifact_from_row(existing)

            artifact_id = new_memory_id("art")
            storage_uri = f"sqlite://tool_artifact/{artifact_id}"
            preview = text[:500]
            created_at = time.time()
            conn.execute(
                """
                INSERT INTO tool_artifact (
                    artifact_id, tenant_id, session_id, tool_call_id, storage_uri, content_type, encoding,
                    size_bytes, sha256, preview_text, raw_content, created_at, is_deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    artifact_id,
                    tenant_id,
                    session_id,
                    tool_call_id,
                    storage_uri,
                    content_type,
                    encoding,
                    len(data),
                    digest,
                    preview,
                    text,
                    created_at,
                ),
            )
        stored = self.get(artifact_id)
        if stored is None:
            raise RuntimeError(f"artifact was not stored: {artifact_id}")
        return stored

    def get(self, artifact_id: str, include_deleted: bool = False) -> Optional[ToolArtifact]:
        """读取 artifact 元数据，不返回 raw_content。"""

        self.initialize()
        condition = "" if include_deleted else " AND is_deleted = 0"
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT artifact_id, tenant_id, session_id, tool_call_id, storage_uri, content_type, encoding,
                       size_bytes, sha256, compression_codec, schema_name, schema_version, preview_text,
                       created_at, expires_at, is_deleted
                FROM tool_artifact
                WHERE artifact_id = ?{condition}
                """,
                (artifact_id,),
            ).fetchone()
        return self._artifact_from_row(row) if row else None

    def get_content(self, artifact_id: str, include_deleted: bool = False) -> Optional[str]:
        """读取 artifact 原始内容。"""

        self.initialize()
        condition = "" if include_deleted else " AND is_deleted = 0"
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT raw_content FROM tool_artifact WHERE artifact_id = ?{condition}",
                (artifact_id,),
            ).fetchone()
        return row["raw_content"] if row else None

    def mark_deleted(self, artifact_id: str) -> bool:
        """按 artifact_id tombstone 标记原始输出。"""

        self.initialize()
        with self._connect() as conn:
            cursor = conn.execute("UPDATE tool_artifact SET is_deleted = 1 WHERE artifact_id = ?", (artifact_id,))
        return cursor.rowcount > 0

    def mark_deleted_by_session(self, session_id: str) -> int:
        """按会话 tombstone 标记所有 artifact。"""

        self.initialize()
        with self._connect() as conn:
            cursor = conn.execute("UPDATE tool_artifact SET is_deleted = 1 WHERE session_id = ?", (session_id,))
        return cursor.rowcount

    def mark_deleted_by_tool_call(self, session_id: str, tool_call_id: str) -> int:
        """按工具调用 tombstone 标记对应 artifact。"""

        self.initialize()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE tool_artifact SET is_deleted = 1 WHERE session_id = ? AND tool_call_id = ?",
                (session_id, tool_call_id),
            )
        return cursor.rowcount

    def _artifact_from_row(self, row) -> ToolArtifact:
        """把 SQLite Row 反序列化为 artifact 元数据对象。"""

        return ToolArtifact(
            artifact_id=row["artifact_id"],
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            tool_call_id=row["tool_call_id"],
            storage_uri=row["storage_uri"],
            content_type=row["content_type"] or "text/plain",
            encoding=row["encoding"] or "utf-8",
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            compression_codec=row["compression_codec"],
            schema_name=row["schema_name"],
            schema_version=row["schema_version"],
            preview_text=row["preview_text"] or "",
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            is_deleted=bool(row["is_deleted"]),
        )
