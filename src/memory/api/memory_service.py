"""短期记忆系统对外入口。

MemoryService 聚合事件写入、上下文读取、摘要快照、任务状态和删除服务。
上层 Agent 只依赖这个 facade，不需要直接理解事件表、artifact 表和投影表。
"""

import re
from typing import Any, Dict, Optional

from src.core.logger import logger
from src.memory.api.dto import (
    AppendMessageRequest,
    AppendToolCallRequest,
    BuildContextRequest,
    DeleteMemoryRequest,
)
from src.memory.application.compression_service import CompressionService
from src.memory.application.deletion_service import DeletionService
from src.memory.application.memory_maintenance_service import (
    MemoryMaintenanceService,
    summary_trigger_strategy_overrides_from_settings,
)
from src.memory.application.memory_read_service import MemoryReadService
from src.memory.application.memory_write_service import MemoryWriteService
from src.memory.application.summary_snapshot_manager import SummarySnapshotManager
from src.memory.application.task_state_manager import TaskStateManager
from src.memory.core.models import Message, ToolCallRecord
from src.memory.domain.deletion import DeletionJob
from src.memory.domain.summary_snapshot import SummarySnapshot
from src.memory.domain.task_state import TaskState
from src.memory.infrastructure.repositories.artifact_store import ArtifactStore
from src.memory.infrastructure.repositories.deletion_repo import DeletionRepository
from src.memory.infrastructure.repositories.event_store import EventStore
from src.memory.infrastructure.repositories.summary_snapshot_repo import (
    SummarySnapshotRepository,
)
from src.memory.infrastructure.repositories.task_state_repo import TaskStateRepository
from src.memory.retrieval import RetrievalPlanner
from src.memory.selection_strategy_config import get_memory_selection_strategy_config


class MemoryService:
    """Thin facade over read/write/maintenance short-term memory services."""

    def __init__(
        self,
        db_path: str,
        tenant_id: str = "default",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        event_store: Optional[EventStore] = None,
        artifact_store: Optional[ArtifactStore] = None,
        task_state_manager: Optional[TaskStateManager] = None,
        summary_snapshot_manager: Optional[SummarySnapshotManager] = None,
        compression_service: Optional[CompressionService] = None,
        retrieval_planner: Optional[RetrievalPlanner] = None,
        deletion_service: Optional[DeletionService] = None,
    ):
        """初始化短期记忆依赖，并确保 SQLite 表结构已创建。"""

        self.db_path = db_path
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.user_id = user_id
        self.event_store = event_store or EventStore(db_path)
        self.artifact_store = artifact_store or ArtifactStore(db_path)
        self.task_state_repository = TaskStateRepository(db_path)
        self.summary_snapshot_repository = SummarySnapshotRepository(db_path)
        self.deletion_repository = DeletionRepository(db_path)
        self.strategy_config = get_memory_selection_strategy_config(
            overrides=summary_trigger_strategy_overrides_from_settings()
        )
        self.task_state_manager = task_state_manager or TaskStateManager(
            self.task_state_repository,
            self.event_store,
        )
        self.summary_snapshot_manager = (
            summary_snapshot_manager
            or SummarySnapshotManager(
                self.summary_snapshot_repository,
                self.event_store,
            )
        )
        self.compression_service = compression_service or CompressionService(
            self.summary_snapshot_manager
        )
        self.retrieval_planner = retrieval_planner or RetrievalPlanner(
            self._estimate_tokens,
            strategy_config=self.strategy_config,
        )
        self.deletion_service = deletion_service or DeletionService(
            event_store=self.event_store,
            artifact_store=self.artifact_store,
            deletion_repository=self.deletion_repository,
            task_state_repository=self.task_state_repository,
            summary_snapshot_repository=self.summary_snapshot_repository,
        )
        self.event_store.initialize()
        self.artifact_store.initialize()
        self.task_state_repository.initialize()
        self.summary_snapshot_repository.initialize()
        self.deletion_repository.initialize()

        self.read_service = MemoryReadService(
            event_store=self.event_store,
            task_state_manager=self.task_state_manager,
            summary_snapshot_manager=self.summary_snapshot_manager,
            retrieval_planner=self.retrieval_planner,
        )
        self.write_service = MemoryWriteService(
            tenant_id=self.tenant_id,
            event_store=self.event_store,
            artifact_store=self.artifact_store,
            task_state_manager=self.task_state_manager,
            compression_service=self.compression_service,
            strategy_config=self.strategy_config,
        )
        self.maintenance_service = MemoryMaintenanceService(
            tenant_id=self.tenant_id,
            event_store=self.event_store,
            artifact_store=self.artifact_store,
            summary_snapshot_manager=self.summary_snapshot_manager,
            compression_service=self.compression_service,
            deletion_service=self.deletion_service,
            read_service=self.read_service,
            strategy_config=self.strategy_config,
        )

    def append_message(self, request: AppendMessageRequest) -> Message:
        """写入一条对话消息事件，助手消息写入后按阈值尝试自动摘要。"""

        self._remember_session(request.session_id, request.user_id)
        message = self.write_service.append_message(request)
        if request.role == "assistant":
            self._maybe_auto_summary_snapshot(request.session_id, role=request.role)
        return message

    def append_tool_call(self, request: AppendToolCallRequest) -> ToolCallRecord:
        """保存工具原始输出 artifact，并写入可用于上下文检索的工具事件。"""

        self._remember_session(request.session_id, request.user_id)
        return self.write_service.append_tool_call(request)

    def update_task_state(
        self,
        session_id: str,
        patch: Dict[str, Any],
        tenant_id: Optional[str] = None,
        expected_version: Optional[int] = None,
        created_by: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> TaskState:
        """用浅层 patch 更新当前会话任务状态投影。"""

        return self.write_service.update_task_state(
            session_id=session_id,
            patch=patch,
            tenant_id=tenant_id,
            expected_version=expected_version,
            created_by=created_by,
            turn_id=turn_id,
        )

    def get_task_state(
        self, session_id: Optional[str] = None, tenant_id: Optional[str] = None
    ) -> TaskState:
        """读取指定会话的结构化任务状态；不存在时由读服务创建默认状态。"""

        effective_session_id = session_id or self._require_session_id()
        return self.read_service.get_task_state(
            session_id=effective_session_id,
            tenant_id=tenant_id or self.tenant_id,
        )

    def build_context(self, request: BuildContextRequest) -> Dict[str, Any]:
        """按 query 和 token 预算构建 prompt 侧短期记忆上下文。"""

        self._remember_session(request.session_id)
        context = self.read_service.build_context(request)
        if context.get("summary_needed"):
            self._maybe_auto_summary_snapshot(
                request.session_id,
                role="context",
                context_pressure=context.get("context_pressure"),
                omitted_counts=context.get("omitted_counts"),
            )
        return context

    def create_summary_snapshot(
        self,
        session_id: str,
        tenant_id: Optional[str] = None,
        created_by_model: str = "rule-based",
        snapshot_batch_size: Optional[int] = None,
    ) -> SummarySnapshot:
        """把尚未覆盖的事件压缩为一条新的 summary snapshot。"""

        return self.maintenance_service.create_summary_snapshot(
            session_id=session_id,
            tenant_id=tenant_id,
            created_by_model=created_by_model,
            snapshot_batch_size=snapshot_batch_size,
        )

    def rebase_summary_snapshot(
        self,
        session_id: str,
        tenant_id: Optional[str] = None,
        snapshot_batch_size: Optional[int] = None,
    ) -> SummarySnapshot:
        """基于已有快照和新增事件生成新的工作快照。"""

        return self.maintenance_service.rebase_summary_snapshot(
            session_id=session_id,
            tenant_id=tenant_id,
            snapshot_batch_size=snapshot_batch_size,
        )

    def delete_memory(self, request: DeleteMemoryRequest) -> DeletionJob:
        """按 scope 执行短期记忆 tombstone 删除并返回删除任务。"""

        return self.maintenance_service.delete_memory(request)

    def get_raw_artifact(self, artifact_id: str) -> Optional[str]:
        """读取工具调用的原始 artifact 内容，主要用于调试和管理接口。"""

        return self.maintenance_service.get_raw_artifact(artifact_id)

    def clear(self, session_id: Optional[str] = None) -> None:
        """清空当前或指定会话的短期记忆数据。"""

        effective_session_id = session_id or self._require_session_id()
        self.delete_memory(
            DeleteMemoryRequest(
                tenant_id=self.tenant_id,
                scope="session",
                selector={"session_id": effective_session_id},
            )
        )

    def get_debug_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """返回事件数量、投影数量和任务状态等调试信息。"""

        effective_session_id = session_id or self._require_session_id()
        info = self.read_service.get_debug_info(effective_session_id)
        info["task_state"] = self.get_task_state(effective_session_id).to_dict()
        return info

    def get_context_debug_info(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """返回当前上下文构建结果的截断预览和检索计划。"""

        effective_session_id = session_id or self._require_session_id()
        context = self.build_context(
            BuildContextRequest(
                tenant_id=self.tenant_id, session_id=effective_session_id
            )
        )
        return {
            "context_text_preview": context["context_text"][:500],
            "context_total_tokens": context["total_tokens"],
            "retrieval_plan": context["retrieval_plan"],
            "selected_summary_snapshot": context["selected_summary_snapshot"],
            "selected_task_state": context["selected_task_state"],
        }

    def get_all_messages(
        self, session_id: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """返回指定会话的全部消息投影，兼容旧调试入口。"""

        return self.read_service.get_all_messages(
            session_id or self._require_session_id()
        )

    def get_tool_calls(self, session_id: Optional[str] = None) -> list[Dict[str, Any]]:
        """返回指定会话的工具调用投影。"""

        return self.read_service.get_tool_calls(
            session_id or self._require_session_id()
        )

    def get_salient_facts(
        self, session_id: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """返回指定会话的显著事实投影。"""

        return self.read_service.get_salient_facts(
            session_id or self._require_session_id()
        )

    def get_summary(self, session_id: Optional[str] = None) -> str:
        """返回指定会话的最新摘要文本。"""

        return self.read_service.get_summary(session_id or self._require_session_id())

    def export_memory(self, session_id: str) -> Dict[str, Any]:
        """导出会话短期记忆的原始事件、任务状态和最新摘要快照。"""

        return self.read_service.export_memory(session_id, self.tenant_id)

    def _remember_session(self, session_id: str, user_id: Optional[str] = None) -> None:
        """记录最近一次使用的 session/user，兼容旧调用方的无参读取方法。"""

        self.session_id = session_id
        if user_id:
            self.user_id = user_id

    def _require_session_id(self) -> str:
        """在旧式无参操作缺少 session_id 时抛出清晰错误。"""

        if self.session_id:
            return self.session_id
        raise ValueError("MemoryService requires a session_id for this operation")

    def _maybe_auto_summary_snapshot(
        self,
        session_id: str,
        role: str = "assistant",
        context_pressure: Optional[float] = None,
        omitted_counts: Optional[dict[str, int]] = None,
    ) -> None:
        """根据维护服务的阈值判断自动创建或 rebase summary snapshot。"""

        try:
            decision = self.maintenance_service.should_create_summary_snapshot(
                session_id=session_id,
                context_pressure=context_pressure,
                omitted_counts=omitted_counts,
            )
            if not decision.should_create:
                return
            if decision.mode == "create":
                self.maintenance_service.create_summary_snapshot(
                    session_id=session_id,
                    created_by_model="auto-trigger",
                )
            elif decision.mode == "rebase":
                self.maintenance_service.rebase_summary_snapshot(
                    session_id=session_id,
                )
        except Exception:
            logger.exception(
                "auto summary snapshot failed for session %s (role=%s)",
                session_id,
                role,
            )

    def _estimate_tokens(self, text: str) -> int:
        """估算中英文混合文本 token 数，用于检索规划的预算控制。"""

        if not text:
            return 0
        ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        other_chars = max(
            len(text) - sum(len(word) for word in ascii_words) - len(cjk_chars), 0
        )
        return max(1, len(ascii_words) + len(cjk_chars) + other_chars // 4)
