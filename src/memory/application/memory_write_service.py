"""短期记忆写入服务。

该层把上层 DTO 转换为 append-only 事件，同时把大型工具原文保存为 artifact。
投影更新仍交给专门的 manager，保持写入路径职责单一。
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional

from src.memory.api.dto import AppendMessageRequest, AppendToolCallRequest
from src.memory.application.compression_service import CompressionService
from src.memory.application.task_state_manager import TaskStateManager
from src.memory.core.models import Message, ToolCallRecord
from src.memory.domain.events import MemoryEvent, MemoryEventType
from src.memory.domain.task_state import TaskState
from src.memory.infrastructure.repositories.artifact_store import ArtifactStore
from src.memory.infrastructure.repositories.event_store import EventStore


class MemoryWriteService:
    """Writes raw events and refreshes essential short-term projections."""

    def __init__(
        self,
        tenant_id: str,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        task_state_manager: TaskStateManager,
        compression_service: CompressionService,
    ):
        """保存写路径依赖，负责事件、artifact 和任务状态投影写入。"""

        self.tenant_id = tenant_id
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.task_state_manager = task_state_manager
        self.compression_service = compression_service

    def append_message(self, request: AppendMessageRequest) -> Message:
        """把用户/助手消息转成 message_created 事件并持久化。"""

        timestamp = request.timestamp or time.time()
        message = Message(
            role=request.role,
            content=request.content,
            timestamp=timestamp,
            session_id=request.session_id,
            importance=request.importance or 0,
            metadata=request.metadata or {},
        )
        event = MemoryEvent(
            event_id=request.event_id
            or message.message_id.replace("msg_", "evt_msg_", 1),
            tenant_id=request.tenant_id or self.tenant_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            event_type=MemoryEventType.MESSAGE_CREATED.value,
            source_type="message",
            source_id=message.message_id,
            payload=message.to_dict(),
            created_by=request.user_id,
            created_at=timestamp,
        )
        self.event_store.append(event)
        return message

    def append_tool_call(self, request: AppendToolCallRequest) -> ToolCallRecord:
        """保存工具原始输出，写入摘要化的工具成功/失败事件。"""

        timestamp = request.timestamp or time.time()
        tool_call_id = ToolCallRecord(
            tool_name=request.tool_name, timestamp=timestamp
        ).tool_call_id
        artifact = self.artifact_store.put(
            tenant_id=request.tenant_id or self.tenant_id,
            session_id=request.session_id,
            tool_call_id=tool_call_id,
            raw_content=request.raw_output,
            content_type=request.content_type,
        )
        output_digest = self.compression_service.digest_tool_output(request.raw_output)
        metadata = self._build_tool_metadata(
            tool_name=request.tool_name,
            tool_input=request.tool_input,
            timestamp=timestamp,
            metadata=request.metadata or {},
        )
        record = ToolCallRecord(
            tool_call_id=tool_call_id,
            tool_name=request.tool_name,
            timestamp=timestamp,
            input_summary=request.tool_input,
            output_digest=output_digest,
            output_summary=output_digest,
            output_is_summary=True,
            output_is_truncated=len(output_digest) < len(request.raw_output or ""),
            raw_artifact_id=artifact.artifact_id,
            raw_size_bytes=artifact.size_bytes,
            content_type=artifact.content_type,
            status="success" if request.success else "error",
            metadata=metadata,
        )
        event = MemoryEvent(
            event_id=request.event_id
            or record.tool_call_id.replace("tool_", "evt_tool_", 1),
            tenant_id=request.tenant_id or self.tenant_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            event_type=(
                MemoryEventType.TOOL_CALL_FINISHED.value
                if request.success
                else MemoryEventType.TOOL_CALL_FAILED.value
            ),
            source_type="tool_call",
            source_id=record.tool_call_id,
            payload={**record.to_dict(), "raw_artifact_id": artifact.artifact_id},
            created_by=request.user_id,
            created_at=timestamp,
        )
        self.event_store.append(event)
        return record

    def _build_tool_metadata(
        self,
        tool_name: str,
        tool_input: str,
        timestamp: float,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attach deterministic evidence metadata without changing storage schema."""

        enriched = dict(metadata or {})
        enriched.setdefault("params_hash", self._params_hash(tool_input))
        enriched.setdefault("produced_at", timestamp)
        enriched.setdefault(
            "effective_until",
            self._infer_effective_until(tool_name=tool_name, produced_at=timestamp),
        )
        return enriched

    def _params_hash(self, tool_input: str) -> str:
        """为工具输入生成稳定短哈希，作为 evidence 参数链标识。"""

        normalized = self._normalize_tool_input(tool_input)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _normalize_tool_input(self, tool_input: str) -> str:
        """规范化工具输入，JSON 输入按键排序保证哈希稳定。"""

        raw = (tool_input or "").strip()
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
        except Exception:
            return raw
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _infer_effective_until(self, tool_name: str, produced_at: float) -> float:
        """按工具类型推断证据默认有效期截止时间。"""

        name = (tool_name or "").lower()
        if any(token in name for token in ["weather", "天气"]):
            return produced_at + 6 * 60 * 60
        if any(token in name for token in ["neo", "asteroid", "小行星", "近地"]):
            return produced_at + 12 * 60 * 60
        if any(token in name for token in ["event", "forecast", "calendar", "meteor"]):
            return produced_at + 24 * 60 * 60
        if any(token in name for token in ["position", "位置", "ephemeris"]):
            return produced_at + 2 * 60 * 60
        return produced_at + 24 * 60 * 60

    def update_task_state(
        self,
        session_id: str,
        patch: Dict[str, Any],
        tenant_id: Optional[str] = None,
        expected_version: Optional[int] = None,
        created_by: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> TaskState:
        """代理到 TaskStateManager，保证任务状态和事件日志同步更新。"""

        return self.task_state_manager.patch_state(
            tenant_id=tenant_id or self.tenant_id,
            session_id=session_id,
            patch=patch,
            expected_version=expected_version,
            created_by=created_by,
            turn_id=turn_id,
        )
