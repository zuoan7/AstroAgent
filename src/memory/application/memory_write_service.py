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
        self.tenant_id = tenant_id
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.task_state_manager = task_state_manager
        self.compression_service = compression_service

    def append_message(self, request: AppendMessageRequest) -> Message:
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

    def update_task_state(
        self,
        session_id: str,
        patch: Dict[str, Any],
        tenant_id: Optional[str] = None,
        expected_version: Optional[int] = None,
        created_by: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> TaskState:
        return self.task_state_manager.patch_state(
            tenant_id=tenant_id or self.tenant_id,
            session_id=session_id,
            patch=patch,
            expected_version=expected_version,
            created_by=created_by,
            turn_id=turn_id,
        )
