import re
import time
from typing import Any, Dict, Optional

from src.memory.api.dto import (
    AppendMessageRequest,
    AppendToolCallRequest,
    BuildContextRequest,
    DeleteMemoryRequest,
)
from src.memory.application.compression_service import CompressionService
from src.memory.application.deletion_service import DeletionService
from src.memory.application.summary_snapshot_manager import SummarySnapshotManager
from src.memory.application.task_state_manager import TaskStateManager
from src.memory.core.models import Message, SalientFact, ToolCallRecord
from src.memory.domain.deletion import DeletionJob
from src.memory.domain.events import MemoryEvent, MemoryEventType, new_memory_id
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


class MemoryService:
    """Unified memory facade backed by events, artifacts, summaries, and retrieval."""

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
        self.db_path = db_path
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.user_id = user_id
        self.event_store = event_store or EventStore(db_path)
        self.artifact_store = artifact_store or ArtifactStore(db_path)
        self.task_state_repository = TaskStateRepository(db_path)
        self.summary_snapshot_repository = SummarySnapshotRepository(db_path)
        self.deletion_repository = DeletionRepository(db_path)
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
            self._estimate_tokens
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

    def append_message(self, request: AppendMessageRequest) -> Message:
        """Append a conversational message event."""

        timestamp = request.timestamp or time.time()
        self._remember_session(request.session_id, request.user_id)
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
        """Store raw tool output as an artifact and append a tool-call event."""

        timestamp = request.timestamp or time.time()
        self._remember_session(request.session_id, request.user_id)
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
    ) -> TaskState:
        """Patch structured task state for a session."""

        return self.task_state_manager.patch_state(
            tenant_id=tenant_id or self.tenant_id,
            session_id=session_id,
            patch=patch,
            expected_version=expected_version,
            created_by=created_by,
        )

    def get_task_state(
        self, session_id: str, tenant_id: Optional[str] = None
    ) -> TaskState:
        """Return the current structured task state, creating an empty one if needed."""

        return self.task_state_manager.get_state(
            tenant_id or self.tenant_id, session_id
        )

    def build_context(self, request: BuildContextRequest) -> Dict[str, Any]:
        """Build query-aware context dynamically from events, snapshots, and task state."""

        self._remember_session(request.session_id)
        token_budget = request.max_tokens or 4000
        events = self.event_store.list_by_session(request.session_id)
        messages = self._messages_from_events(events)
        facts = self._facts_from_events(events)
        tool_calls = self._tool_calls_from_events(events)
        return self.retrieval_planner.build_context(
            query=request.query,
            token_budget=token_budget,
            task_state=self.get_task_state(request.session_id, request.tenant_id),
            summary_snapshot=self.summary_snapshot_manager.get_latest(
                request.session_id
            ),
            messages=messages,
            facts=facts,
            tool_calls=tool_calls,
        )

    def create_summary_snapshot(
        self,
        session_id: str,
        tenant_id: Optional[str] = None,
        created_by_model: str = "rule-based",
    ) -> SummarySnapshot:
        """Create a snapshot from current active raw events."""

        effective_tenant = tenant_id or self.tenant_id
        events = self.event_store.list_by_session(session_id)
        return self.compression_service.create_summary_snapshot(
            tenant_id=effective_tenant,
            session_id=session_id,
            events=events,
            created_by_model=created_by_model,
        )

    def rebase_summary_snapshot(
        self, session_id: str, tenant_id: Optional[str] = None
    ) -> SummarySnapshot:
        """Create a new snapshot from latest snapshot plus events after its coverage."""

        effective_tenant = tenant_id or self.tenant_id
        latest = self.summary_snapshot_manager.get_latest(session_id)
        events = self.event_store.list_by_session(session_id)
        if latest and latest.covered_to_event_id:
            seen = False
            uncovered = []
            for event in events:
                if seen:
                    uncovered.append(event)
                if event.event_id == latest.covered_to_event_id:
                    seen = True
            events = uncovered
        return self.compression_service.rebase_summary(
            effective_tenant, session_id, latest, events
        )

    def delete_memory(self, request: DeleteMemoryRequest) -> DeletionJob:
        """Delete memory by supported scope using tombstones."""

        return self.deletion_service.delete_memory(
            tenant_id=request.tenant_id or self.tenant_id,
            scope=request.scope,
            selector=request.selector,
            requested_by=request.requested_by,
        )

    def get_raw_artifact(self, artifact_id: str) -> Optional[str]:
        """Fetch raw artifact content by id."""

        return self.artifact_store.get_content(artifact_id)

    def clear(self, session_id: Optional[str] = None) -> None:
        """Tombstone all active memory for a session."""

        effective_session_id = session_id or self._require_session_id()
        self.delete_memory(
            DeleteMemoryRequest(
                tenant_id=self.tenant_id,
                scope="session",
                selector={"session_id": effective_session_id},
            )
        )

    def get_debug_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        effective_session_id = session_id or self._require_session_id()
        events = self.event_store.list_by_session(effective_session_id)
        return {
            "session_id": effective_session_id,
            "event_count": len(events),
            "message_count": len(self._messages_from_events(events)),
            "tool_call_count": len(self._tool_calls_from_events(events)),
            "fact_count": len(self._facts_from_events(events)),
            "summary_available": self.summary_snapshot_manager.get_latest(
                effective_session_id
            )
            is not None,
        }

    def get_context_debug_info(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
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
        events = self.event_store.list_by_session(
            session_id or self._require_session_id()
        )
        return [msg.to_dict() for msg in self._messages_from_events(events)]

    def get_tool_calls(self, session_id: Optional[str] = None) -> list[Dict[str, Any]]:
        events = self.event_store.list_by_session(
            session_id or self._require_session_id()
        )
        return [item.to_dict() for item in self._tool_calls_from_events(events)]

    def get_salient_facts(
        self, session_id: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        events = self.event_store.list_by_session(
            session_id or self._require_session_id()
        )
        return [item.to_dict() for item in self._facts_from_events(events)]

    def get_summary(self, session_id: Optional[str] = None) -> str:
        latest = self.summary_snapshot_manager.get_latest(
            session_id or self._require_session_id()
        )
        return latest.summary_text if latest else ""

    def export_memory(self, session_id: str) -> Dict[str, Any]:
        """Export P0 raw records for diagnostics and replay preparation."""

        return {
            "session_id": session_id,
            "events": [
                event.to_dict()
                for event in self.event_store.list_by_session(session_id)
            ],
            "task_state": self.get_task_state(session_id).to_dict(),
            "summary_snapshot": (
                self.summary_snapshot_manager.get_latest(session_id).to_dict()
                if self.summary_snapshot_manager.get_latest(session_id)
                else None
            ),
        }

    def _remember_session(self, session_id: str, user_id: Optional[str] = None) -> None:
        self.session_id = session_id
        if user_id:
            self.user_id = user_id

    def _require_session_id(self) -> str:
        if self.session_id:
            return self.session_id
        raise ValueError("MemoryService requires a session_id for this operation")

    def _messages_from_events(self, events: list[MemoryEvent]) -> list[Message]:
        messages = []
        for event in events:
            if event.event_type != MemoryEventType.MESSAGE_CREATED.value:
                continue
            payload = event.payload or {}
            data = {
                "message_id": event.source_id or payload.get("message_id"),
                "session_id": event.session_id,
                "role": payload.get("role", "user"),
                "content": payload.get("content", ""),
                "timestamp": payload.get("timestamp", event.created_at),
                "importance": payload.get("importance", 0),
                "importance_reason": payload.get("importance_reason", ""),
                "message_type": payload.get("message_type", "chat"),
                "metadata": payload.get("metadata", {}) or {},
            }
            messages.append(Message.from_dict(data))
        return messages

    def _tool_calls_from_events(
        self, events: list[MemoryEvent]
    ) -> list[ToolCallRecord]:
        tool_calls = []
        for event in events:
            if event.event_type not in {
                MemoryEventType.TOOL_CALL_FINISHED.value,
                MemoryEventType.TOOL_CALL_FAILED.value,
            }:
                continue
            payload = event.payload or {}
            data = {
                "tool_call_id": event.source_id or payload.get("tool_call_id"),
                "tool_name": payload.get("tool_name", "tool"),
                "timestamp": payload.get("timestamp", event.created_at),
                "input_summary": payload.get(
                    "input_summary", payload.get("tool_input", "")
                ),
                "output_digest": payload.get("output_digest", ""),
                "output_summary": payload.get(
                    "output_summary", payload.get("result_summary", "")
                ),
                "output_is_summary": payload.get("output_is_summary", False),
                "output_is_truncated": payload.get("output_is_truncated", False),
                "raw_artifact_id": payload.get("raw_artifact_id", ""),
                "raw_size_bytes": payload.get("raw_size_bytes", 0),
                "content_type": payload.get("content_type", "text/plain"),
                "status": payload.get(
                    "status",
                    (
                        "success"
                        if event.event_type == MemoryEventType.TOOL_CALL_FINISHED.value
                        else "error"
                    ),
                ),
                "importance": payload.get("importance", 1),
            }
            tool_calls.append(ToolCallRecord.from_dict(data))
        return tool_calls

    def _facts_from_events(self, events: list[MemoryEvent]) -> list[SalientFact]:
        facts = []
        for event in events:
            if event.event_type != MemoryEventType.FACT_EXTRACTED.value:
                continue
            payload = event.payload or {}
            facts.append(
                SalientFact.from_dict(
                    {
                        "fact_id": event.source_id
                        or payload.get("fact_id")
                        or new_memory_id("fact"),
                        "fact_type": payload.get("fact_type", "fact"),
                        "content": payload.get("content", ""),
                        "timestamp": payload.get("timestamp", event.created_at),
                        "source_type": payload.get("source_type", ""),
                        "source_id": payload.get("source_id", ""),
                        "source": payload.get("source", ""),
                    }
                )
            )
        return facts

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        other_chars = max(
            len(text) - sum(len(word) for word in ascii_words) - len(cjk_chars), 0
        )
        return max(1, len(ascii_words) + len(cjk_chars) + other_chars // 4)
