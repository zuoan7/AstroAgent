"""短期记忆删除任务和审计仓储。

删除服务把每次 scoped deletion 记录为 deletion_job，并把操作结果写入
audit_log，方便接口追踪和合规排查。
"""

import time
from typing import Optional

from src.memory.domain.deletion import DeletionJob
from src.memory.infrastructure.database import SQLiteRepository
from src.memory.infrastructure.utils import _json_dumps, _json_loads


class DeletionRepository(SQLiteRepository):
    """Repository for deletion jobs and audit records."""

    def initialize(self) -> None:
        """创建删除任务表和审计表。"""

        with self._connect() as conn:
            conn.executescript(
                """
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
                """
            )

    def create_job(self, job: DeletionJob) -> DeletionJob:
        """创建删除任务初始记录。"""

        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO deletion_job (
                    job_id, tenant_id, session_id, delete_scope, selector_json, status,
                    requested_by, requested_at, started_at, finished_at, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                ,
                (
                    job.job_id,
                    job.tenant_id,
                    job.session_id,
                    job.delete_scope,
                    _json_dumps(job.selector),
                    job.status,
                    job.requested_by,
                    job.requested_at,
                    job.started_at,
                    job.finished_at,
                    _json_dumps(job.result),
                ),
            )
        return job

    def update_job(self, job: DeletionJob) -> DeletionJob:
        """更新删除任务状态、时间和结果。"""

        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE deletion_job
                SET status = ?, started_at = ?, finished_at = ?, result_json = ?
                WHERE job_id = ?
                """
                ,
                (job.status, job.started_at, job.finished_at, _json_dumps(job.result), job.job_id),
            )
        return job

    def get_job(self, job_id: str) -> Optional[DeletionJob]:
        """按 job_id 读取删除任务。"""

        self.initialize()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, tenant_id, session_id, delete_scope, selector_json, status,
                       requested_by, requested_at, started_at, finished_at, result_json
                FROM deletion_job WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return DeletionJob(
            job_id=row["job_id"],
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            delete_scope=row["delete_scope"],
            selector=_json_loads(row["selector_json"], {}),
            status=row["status"],
            requested_by=row["requested_by"],
            requested_at=row["requested_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            result=_json_loads(row["result_json"], {}),
        )

    def append_audit(
        self,
        tenant_id: str,
        session_id: Optional[str],
        action_type: str,
        target_type: Optional[str],
        target_id: Optional[str],
        actor_id: Optional[str],
        metadata: dict,
    ) -> None:
        """追加删除审计记录。"""

        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (
                    tenant_id, session_id, action_type, target_type, target_id, actor_id,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                ,
                (
                    tenant_id,
                    session_id,
                    action_type,
                    target_type,
                    target_id,
                    actor_id,
                    _json_dumps(metadata),
                    time.time(),
                ),
            )
