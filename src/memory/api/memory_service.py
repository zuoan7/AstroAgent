import time
from typing import Any, Dict, Optional

from src.memory.api.dto import AppendMessageRequest, AppendToolCallRequest, BuildContextRequest, DeleteMemoryRequest
from src.memory.application.compression_service import CompressionService
from src.memory.application.deletion_service import DeletionService
from src.memory.application.summary_snapshot_manager import SummarySnapshotManager
from src.memory.application.task_state_manager import TaskStateManager
from src.memory.core.models import Message, ToolCallRecord
from src.memory.domain.deletion import DeletionJob
from src.memory.domain.events import MemoryEvent, MemoryEventType
from src.memory.domain.summary_snapshot import SummarySnapshot
from src.memory.domain.task_state import TaskState
from src.memory.infrastructure.repositories.artifact_store import ArtifactStore
from src.memory.infrastructure.repositories.deletion_repo import DeletionRepository
from src.memory.infrastructure.repositories.event_store import EventStore
from src.memory.infrastructure.repositories.summary_snapshot_repo import SummarySnapshotRepository
from src.memory.infrastructure.repositories.task_state_repo import TaskStateRepository
from src.memory.retrieval import RetrievalPlanner
from src.memory.short_term_memory.manager import ShortTermMemory
from src.memory.short_term_memory.repository import ShortTermMemoryRepository


class MemoryService:
    """Unified facade for the first phase of the memory refactor.

    P0 keeps the legacy ShortTermMemory projection in place while adding raw
    event and artifact writes as the new source-of-truth layer.
    """

    def __init__(
        self,
        db_path: str,
        tenant_id: str = "default",
        short_term_memory: Optional[ShortTermMemory] = None,
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
        self.event_store = event_store or EventStore(db_path)
        self.artifact_store = artifact_store or ArtifactStore(db_path)
        self.task_state_repository = TaskStateRepository(db_path)
        self.summary_snapshot_repository = SummarySnapshotRepository(db_path)
        self.deletion_repository = DeletionRepository(db_path)
        self.short_term_repository = ShortTermMemoryRepository(db_path)
        self.task_state_manager = task_state_manager or TaskStateManager(
            self.task_state_repository,
            self.event_store,
        )
        self.summary_snapshot_manager = summary_snapshot_manager or SummarySnapshotManager(
            self.summary_snapshot_repository,
            self.event_store,
        )
        self.compression_service = compression_service or CompressionService(self.summary_snapshot_manager)
        self.retrieval_planner = retrieval_planner or RetrievalPlanner(ShortTermMemory._estimate_tokens)
        self.deletion_service = deletion_service or DeletionService(
            event_store=self.event_store,
            artifact_store=self.artifact_store,
            deletion_repository=self.deletion_repository,
            task_state_repository=self.task_state_repository,
            summary_snapshot_repository=self.summary_snapshot_repository,
            short_term_repository=self.short_term_repository,
        )
        self.short_term_memory = short_term_memory
        self.event_store.initialize()
        self.artifact_store.initialize()
        self.task_state_repository.initialize()
        self.summary_snapshot_repository.initialize()
        self.deletion_repository.initialize()
        self.short_term_repository.initialize()

    def append_message(self, request: AppendMessageRequest) -> Message:
        """Append a message event and update the legacy STM projection."""

        timestamp = request.timestamp or time.time()
        legacy = self._get_or_create_stm(request.session_id, request.user_id)
        legacy.add_message(
            role=request.role,
            content=request.content,
            timestamp=timestamp,
            importance=request.importance,
            metadata=request.metadata,
        )
        message = legacy.messages[-1]
        event = MemoryEvent(
            event_id=request.event_id or message.message_id.replace("msg_", "evt_msg_", 1),
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
        """Store raw tool output as an artifact, append event, and update STM."""

        timestamp = request.timestamp or time.time()
        legacy = self._get_or_create_stm(request.session_id, request.user_id)
        tool_call_id = ToolCallRecord(tool_name=request.tool_name, timestamp=timestamp).tool_call_id
        artifact = self.artifact_store.put(
            tenant_id=request.tenant_id or self.tenant_id,
            session_id=request.session_id,
            tool_call_id=tool_call_id,
            raw_content=request.raw_output,
            content_type=request.content_type,
        )
        output_digest = self.compression_service.digest_tool_output(request.raw_output)
        legacy.add_tool_call(
            tool_name=request.tool_name,
            tool_input=request.tool_input,
            result=request.raw_output,
            timestamp=timestamp,
            success=request.success,
            raw_artifact_id=artifact.artifact_id,
            tool_call_id=tool_call_id,
            raw_size_bytes=artifact.size_bytes,
            content_type=artifact.content_type,
            output_digest=output_digest,
        )
        record = legacy.tool_calls[-1]
        event = MemoryEvent(
            event_id=request.event_id or record.tool_call_id.replace("tool_", "evt_tool_", 1),
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

    def get_task_state(self, session_id: str, tenant_id: Optional[str] = None) -> TaskState:
        """Return the current structured task state, creating an empty one if needed."""

        return self.task_state_manager.get_state(tenant_id or self.tenant_id, session_id)

    def build_context(self, request: BuildContextRequest) -> Dict[str, Any]:
        """Build query-aware context from task state, snapshots, and legacy projections."""

        legacy = self._get_or_create_stm(request.session_id)
        token_budget = request.max_tokens or legacy.config.context_budget
        return self.retrieval_planner.build_context(
            query=request.query,
            token_budget=token_budget,
            task_state=self.get_task_state(request.session_id, request.tenant_id),
            summary_snapshot=self.summary_snapshot_manager.get_latest(request.session_id),
            messages=legacy.messages,
            facts=legacy.salient_facts,
            tool_calls=legacy.tool_calls,
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

    def rebase_summary_snapshot(self, session_id: str, tenant_id: Optional[str] = None) -> SummarySnapshot:
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
        return self.compression_service.rebase_summary(effective_tenant, session_id, latest, events)

    def delete_memory(self, request: DeleteMemoryRequest) -> DeletionJob:
        """Delete memory by supported scope using tombstones and projection cleanup."""

        job = self.deletion_service.delete_memory(
            tenant_id=request.tenant_id or self.tenant_id,
            scope=request.scope,
            selector=request.selector,
            requested_by=request.requested_by,
        )
        self._apply_in_memory_delete(request.scope, request.selector)
        return job

    def get_raw_artifact(self, artifact_id: str) -> Optional[str]:
        """Fetch raw artifact content by id."""

        return self.artifact_store.get_content(artifact_id)

    def export_memory(self, session_id: str) -> Dict[str, Any]:
        """Export P0 raw records for diagnostics and replay preparation."""

        return {
            "session_id": session_id,
            "events": [event.to_dict() for event in self.event_store.list_by_session(session_id)],
            "task_state": self.get_task_state(session_id).to_dict(),
            "summary_snapshot": (
                self.summary_snapshot_manager.get_latest(session_id).to_dict()
                if self.summary_snapshot_manager.get_latest(session_id)
                else None
            ),
        }

    def _get_or_create_stm(self, session_id: str, user_id: Optional[str] = None) -> ShortTermMemory:
        if self.short_term_memory and self.short_term_memory.session_id == session_id:
            return self.short_term_memory
        self.short_term_memory = ShortTermMemory(session_id=session_id, user_id=user_id)
        return self.short_term_memory

    def _apply_in_memory_delete(self, scope: str, selector: Dict[str, Any]) -> None:
        legacy = self.short_term_memory
        if not legacy or legacy.session_id != selector.get("session_id"):
            return
        if scope == "session":
            legacy.messages.clear()
            legacy.tool_calls.clear()
            legacy.salient_facts.clear()
            legacy.summary = ""
            legacy.trimmed_count = 0
            legacy.last_trimmed_content.clear()
        elif scope == "message":
            message_id = selector.get("message_id")
            legacy.messages = [msg for msg in legacy.messages if msg.message_id != message_id]
        elif scope == "tool_call":
            tool_call_id = selector.get("tool_call_id")
            legacy.tool_calls = [call for call in legacy.tool_calls if call.tool_call_id != tool_call_id]
        elif scope == "fact":
            fact_id = selector.get("fact_id")
            legacy.salient_facts = [fact for fact in legacy.salient_facts if fact.fact_id != fact_id]
        elif scope == "time_range":
            start_time = selector.get("start_time")
            end_time = selector.get("end_time")

            def keep(timestamp: float) -> bool:
                if start_time is not None and timestamp < start_time:
                    return True
                if end_time is not None and timestamp > end_time:
                    return True
                return False

            legacy.messages = [msg for msg in legacy.messages if keep(msg.timestamp)]
            legacy.tool_calls = [call for call in legacy.tool_calls if keep(call.timestamp)]
            legacy.salient_facts = [fact for fact in legacy.salient_facts if keep(fact.timestamp)]
