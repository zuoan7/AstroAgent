"""短期记忆维护服务。

负责 summary snapshot 的创建/rebase、自动摘要触发判断、原始 artifact 读取
以及 scoped deletion 的编排。
"""

import re
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
FIXED_SUMMARY_EVENT_TRIGGER = 30
FIXED_SUMMARY_TOKEN_TRIGGER = 6000
TOPIC_DRIFT_DISTANCE_TRIGGER = 0.6
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
        """保存维护服务依赖，所有持久化读写由注入仓储完成。"""

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
        """根据未覆盖事件、token 压力、话题漂移和工具链结束决定是否摘要。"""

        if not settings.MEMORY_AUTO_SUMMARY_ENABLED:
            return SummaryTriggerDecision(
                should_create=False, mode="none", reason="auto_summary_disabled"
            )

        latest = self.summary_snapshot_manager.get_latest(session_id)

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
        if uncovered_count == 0:
            return SummaryTriggerDecision(
                should_create=False,
                mode="none",
                reason="no_uncovered_events",
                uncovered_event_count=0,
                estimated_tokens=0,
            )

        mode = "create" if latest is None else "rebase"
        configured_event_threshold = (
            int(getattr(settings, "MEMORY_SUMMARY_TRIGGER_MESSAGES", 10))
            if latest is None
            else int(getattr(settings, "MEMORY_SUMMARY_MIN_NEW_EVENTS", 6))
        )
        event_threshold = _effective_summary_threshold(
            configured_event_threshold,
            FIXED_SUMMARY_EVENT_TRIGGER,
        )
        configured_token_threshold = int(
            getattr(settings, "MEMORY_SUMMARY_TRIGGER_TOKENS", 3000)
        )
        token_threshold = _effective_summary_threshold(
            configured_token_threshold,
            FIXED_SUMMARY_TOKEN_TRIGGER,
        )

        if uncovered_count >= event_threshold:
            return SummaryTriggerDecision(
                should_create=True,
                mode=mode,
                reason=(
                    f"uncovered_events({uncovered_count}) >= "
                    f"trigger({event_threshold})"
                ),
                uncovered_event_count=uncovered_count,
                estimated_tokens=estimated_tokens,
            )

        if estimated_tokens >= token_threshold:
            return SummaryTriggerDecision(
                should_create=True,
                mode=mode,
                reason=(
                    f"estimated_tokens({estimated_tokens}) >= "
                    f"trigger({token_threshold})"
                ),
                uncovered_event_count=uncovered_count,
                estimated_tokens=estimated_tokens,
            )

        if _has_topic_drift(latest, uncovered_events):
            return SummaryTriggerDecision(
                should_create=True,
                mode=mode,
                reason="topic_drift(distance>0.6)",
                uncovered_event_count=uncovered_count,
                estimated_tokens=estimated_tokens,
            )

        if _has_completed_tool_chain(uncovered_events):
            return SummaryTriggerDecision(
                should_create=True,
                mode=mode,
                reason="tool_chain_completed",
                uncovered_event_count=uncovered_count,
                estimated_tokens=estimated_tokens,
            )

        if latest is None:
            return SummaryTriggerDecision(
                should_create=False,
                mode="none",
                reason=(
                    f"below_threshold: events={uncovered_count}/{event_threshold} "
                    f"tokens={estimated_tokens}/{token_threshold}"
                ),
                uncovered_event_count=uncovered_count,
                estimated_tokens=estimated_tokens,
            )
        else:
            return SummaryTriggerDecision(
                should_create=False,
                mode="none",
                reason=(
                    f"below_rebase_threshold: events={uncovered_count}/{event_threshold} "
                    f"tokens={estimated_tokens}/{token_threshold}"
                ),
                uncovered_event_count=uncovered_count,
                estimated_tokens=estimated_tokens,
            )


def _effective_summary_threshold(configured: int, fixed: int) -> int:
    """Return the lower positive trigger; zero remains valid for non-empty batches."""

    if configured < 0:
        return fixed
    return min(configured, fixed)


def _has_topic_drift(
    latest: SummarySnapshot | None,
    uncovered_events: list[MemoryEvent],
) -> bool:
    """比较上一快照实体与新事件实体，判断是否发生明显话题漂移。"""

    if latest is None or not latest.summary_text:
        return False
    previous_entities = _extract_memory_entities(latest.summary_text)
    current_entities = set()
    for event in uncovered_events[-10:]:
        current_entities |= _extract_memory_entities(_event_text_for_entities(event))
    if not previous_entities or not current_entities:
        return False
    union = previous_entities | current_entities
    distance = 1 - (len(previous_entities & current_entities) / len(union))
    return distance > TOPIC_DRIFT_DISTANCE_TRIGGER


def _has_completed_tool_chain(uncovered_events: list[MemoryEvent]) -> bool:
    """判断未覆盖事件中是否存在工具结果后接 assistant 最终回复。"""

    seen_tool_result = False
    for event in uncovered_events:
        if event.event_type in {
            MemoryEventType.TOOL_CALL_FINISHED.value,
            MemoryEventType.TOOL_CALL_FAILED.value,
        }:
            seen_tool_result = True
            continue
        if (
            seen_tool_result
            and event.event_type == MemoryEventType.MESSAGE_CREATED.value
            and (event.payload or {}).get("role") == "assistant"
        ):
            return True
    return False


def _event_text_for_entities(event: MemoryEvent) -> str:
    """从不同事件 payload 中抽取用于实体识别的文本。"""

    payload = event.payload or {}
    if event.event_type == MemoryEventType.MESSAGE_CREATED.value:
        return str(payload.get("content", ""))
    if event.event_type in {
        MemoryEventType.TOOL_CALL_FINISHED.value,
        MemoryEventType.TOOL_CALL_FAILED.value,
    }:
        return " ".join(
            str(payload.get(key, ""))
            for key in [
                "tool_name",
                "tool_input",
                "input_summary",
                "output_digest",
                "output_summary",
                "raw_output",
            ]
        )
    if event.event_type == MemoryEventType.TASK_STATE_UPDATED.value:
        state = payload.get("state", {})
        return " ".join(
            [
                str(state.get("current_goal", "")),
                str(state.get("next_action", "")),
                str(state.get("active_constraints", "")),
            ]
        )
    if event.event_type == MemoryEventType.FACT_EXTRACTED.value:
        return str(payload.get("content", ""))
    return str(payload)


def _extract_memory_entities(text: str) -> set[str]:
    """从 summary 或事件文本中抽取地点、天体和天象实体。"""

    source = text or ""
    entities = {
        location
        for location in [
            "北京",
            "上海",
            "广州",
            "深圳",
            "杭州",
            "苏州",
            "成都",
            "南京",
            "武汉",
            "西安",
        ]
        if location in source
    }
    entities |= {target.upper() for target in re.findall(r"\b[Mm]\d{2,3}\b", source)}
    aliases = {
        "M42": ["猎户座大星云", "猎户座星云", "Orion Nebula"],
        "M31": ["仙女座星系", "仙女座大星系", "Andromeda"],
        "月球": ["月球", "Moon"],
        "火星": ["火星", "Mars"],
        "木星": ["木星", "Jupiter"],
        "土星": ["土星", "Saturn"],
        "英仙座流星雨": ["英仙座流星雨", "Perseids"],
        "双子座流星雨": ["双子座流星雨", "Geminids"],
    }
    for canonical, names in aliases.items():
        if any(name in source for name in names):
            entities.add(canonical)
    return entities


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
