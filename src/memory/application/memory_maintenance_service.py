"""短期记忆维护服务。

负责 summary snapshot 的创建/rebase、自动摘要触发判断、原始 artifact 读取
以及 scoped deletion 的编排。
"""

from dataclasses import dataclass
from typing import Optional

from src.core.config import settings
from src.memory.api.dto import DeleteMemoryRequest
from src.memory.application.compression_service import CompressionService
from src.memory.application.deletion_service import DeletionService
from src.memory.application.memory_read_service import MemoryReadService
from src.memory.domain.events import MemoryEvent, MemoryEventType
from src.memory.application.summary_snapshot_manager import SummarySnapshotManager
from src.memory.domain.deletion import DeletionJob
from src.memory.domain.summary_snapshot import SummarySnapshot
from src.memory.infrastructure.repositories.artifact_store import ArtifactStore
from src.memory.infrastructure.repositories.event_store import EventStore

DEFAULT_SNAPSHOT_BATCH_SIZE = 200
SNAPSHOTTABLE_EVENT_TYPES = [
    MemoryEventType.MESSAGE_CREATED.value,
    MemoryEventType.TOOL_CALL_FINISHED.value,
    MemoryEventType.TOOL_CALL_FAILED.value,
    MemoryEventType.FACT_EXTRACTED.value,
    MemoryEventType.TASK_STATE_UPDATED.value,
    MemoryEventType.MEMORY_DELETED.value,
]


@dataclass
class SummaryTriggerDecision:
    """Result of checking whether a summary snapshot should be created or rebased."""

    should_create: bool
    mode: str  # "none" | "create" | "rebase"
    reason: str = ""
    uncovered_event_count: int = 0
    estimated_tokens: int = 0


class MemoryMaintenanceService:
    """Creates snapshots, applies deletes, and serves maintenance reads."""

    def __init__(
        self,
        tenant_id: str,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        summary_snapshot_manager: SummarySnapshotManager,
        compression_service: CompressionService,
        deletion_service: DeletionService,
        read_service: MemoryReadService,
    ):
        self.tenant_id = tenant_id
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.summary_snapshot_manager = summary_snapshot_manager
        self.compression_service = compression_service
        self.deletion_service = deletion_service
        self.read_service = read_service

    def create_summary_snapshot(
        self,
        session_id: str,
        tenant_id: Optional[str] = None,
        created_by_model: str = "rule-based",
        snapshot_batch_size: int = DEFAULT_SNAPSHOT_BATCH_SIZE,
    ) -> SummarySnapshot:
        """从最新快照之后的事件批次创建一条新的摘要快照。"""

        latest = self.summary_snapshot_manager.get_latest(session_id)
        events = self._list_snapshot_batch(
            session_id=session_id,
            after_event_id=latest.covered_to_event_id if latest else None,
            snapshot_batch_size=snapshot_batch_size,
        )
        return self.compression_service.create_summary_snapshot(
            tenant_id=tenant_id or self.tenant_id,
            session_id=session_id,
            events=events,
            created_by_model=created_by_model,
        )

    def rebase_summary_snapshot(
        self,
        session_id: str,
        tenant_id: Optional[str] = None,
        snapshot_batch_size: int = DEFAULT_SNAPSHOT_BATCH_SIZE,
    ) -> SummarySnapshot:
        """把已有快照作为种子，与新增事件合并成新的工作快照。"""

        latest = self.summary_snapshot_manager.get_latest(session_id)
        events = self._list_snapshot_batch(
            session_id,
            after_event_id=latest.covered_to_event_id if latest else None,
            snapshot_batch_size=snapshot_batch_size,
        )
        return self.compression_service.rebase_summary(
            tenant_id or self.tenant_id,
            session_id,
            latest,
            events,
        )

    def _list_snapshot_batch(
        self,
        session_id: str,
        after_event_id: Optional[str],
        snapshot_batch_size: int,
    ):
        """按快照覆盖点选择下一批可摘要事件。"""

        batch_size = max(1, int(snapshot_batch_size or DEFAULT_SNAPSHOT_BATCH_SIZE))
        if after_event_id:
            events = self.event_store.list_by_session(
                session_id,
                event_types=SNAPSHOTTABLE_EVENT_TYPES,
                after_event_id=after_event_id,
                limit=batch_size,
            )
        else:
            events = self.event_store.list_by_session(
                session_id,
                event_types=SNAPSHOTTABLE_EVENT_TYPES,
                limit=batch_size,
                descending=True,
            )
        return events

    def delete_memory(self, request: DeleteMemoryRequest) -> DeletionJob:
        """执行短期记忆删除请求，并保留删除任务和审计记录。"""

        return self.deletion_service.delete_memory(
            tenant_id=request.tenant_id or self.tenant_id,
            scope=request.scope,
            selector=request.selector,
            requested_by=request.requested_by,
        )

    def get_raw_artifact(self, artifact_id: str) -> Optional[str]:
        """按 artifact_id 读取工具调用原始输出。"""

        return self.artifact_store.get_content(artifact_id)

    def should_create_summary_snapshot(
        self,
        session_id: str,
        tenant_id: Optional[str] = None,
    ) -> SummaryTriggerDecision:
        """根据未覆盖事件数量和估算 token 数决定是否生成或 rebase 快照。"""

        if not settings.MEMORY_AUTO_SUMMARY_ENABLED:
            return SummaryTriggerDecision(
                should_create=False, mode="none", reason="auto_summary_disabled"
            )

        latest = self.summary_snapshot_manager.get_latest(session_id)
        effective_tenant = tenant_id or self.tenant_id

        uncovered_events: list[MemoryEvent] = []
        if latest is not None:
            # Only count events after the last covered event
            uncovered_events = self.event_store.list_by_session(
                session_id,
                event_types=SNAPSHOTTABLE_EVENT_TYPES,
                after_event_id=latest.covered_to_event_id,
                limit=500,
            )
        else:
            uncovered_events = self.event_store.list_by_session(
                session_id,
                event_types=SNAPSHOTTABLE_EVENT_TYPES,
                limit=500,
            )

        uncovered_count = len(uncovered_events)
        estimated_tokens = _estimate_event_tokens(uncovered_events)

        if latest is None:
            # First snapshot: create when thresholds are met
            trigger_messages = int(getattr(settings, "MEMORY_SUMMARY_TRIGGER_MESSAGES", 10))
            trigger_tokens = int(getattr(settings, "MEMORY_SUMMARY_TRIGGER_TOKENS", 3000))
            if uncovered_count >= trigger_messages:
                return SummaryTriggerDecision(
                    should_create=True,
                    mode="create",
                    reason=f"message_count({uncovered_count}) >= trigger({trigger_messages})",
                    uncovered_event_count=uncovered_count,
                    estimated_tokens=estimated_tokens,
                )
            if estimated_tokens >= trigger_tokens:
                return SummaryTriggerDecision(
                    should_create=True,
                    mode="create",
                    reason=f"estimated_tokens({estimated_tokens}) >= trigger({trigger_tokens})",
                    uncovered_event_count=uncovered_count,
                    estimated_tokens=estimated_tokens,
                )
            return SummaryTriggerDecision(
                should_create=False,
                mode="none",
                reason=f"below_threshold: msgs={uncovered_count}/{trigger_messages} tokens={estimated_tokens}/{trigger_tokens}",
                uncovered_event_count=uncovered_count,
                estimated_tokens=estimated_tokens,
            )
        else:
            # Existing snapshot: rebase when enough new events
            min_new = int(getattr(settings, "MEMORY_SUMMARY_MIN_NEW_EVENTS", 6))
            trigger_tokens = int(getattr(settings, "MEMORY_SUMMARY_TRIGGER_TOKENS", 3000))
            if uncovered_count >= min_new:
                return SummaryTriggerDecision(
                    should_create=True,
                    mode="rebase",
                    reason=f"new_events({uncovered_count}) >= min_new({min_new})",
                    uncovered_event_count=uncovered_count,
                    estimated_tokens=estimated_tokens,
                )
            if estimated_tokens >= trigger_tokens:
                return SummaryTriggerDecision(
                    should_create=True,
                    mode="rebase",
                    reason=f"estimated_tokens({estimated_tokens}) >= trigger({trigger_tokens})",
                    uncovered_event_count=uncovered_count,
                    estimated_tokens=estimated_tokens,
                )
            return SummaryTriggerDecision(
                should_create=False,
                mode="none",
                reason=f"below_rebase_threshold: new={uncovered_count}/{min_new} tokens={estimated_tokens}/{trigger_tokens}",
                uncovered_event_count=uncovered_count,
                estimated_tokens=estimated_tokens,
            )


def _estimate_event_tokens(events: list[MemoryEvent]) -> int:
    """粗略估算一批事件对应的 token 数。字符数 / 2 为近似估算。"""
    total_chars = 0
    for event in events:
        payload = event.payload or {}
        if event.event_type == "message_created":
            total_chars += len(payload.get("content", ""))
        elif event.event_type in ("tool_call_finished", "tool_call_failed"):
            output = payload.get("output_digest") or payload.get("output_summary", "")
            total_chars += len(str(output))
            total_chars += len(payload.get("tool_name", ""))
            total_chars += len(payload.get("tool_input", ""))
        elif event.event_type == "task_state_updated":
            state = payload.get("state", {})
            total_chars += len(str(state.get("current_goal", "")))
            total_chars += len(str(state.get("next_action", "")))
            total_chars += len(str(state.get("active_constraints", "")))
        elif event.event_type == "fact_extracted":
            total_chars += len(payload.get("content", ""))
    return max(1, total_chars // 2)
