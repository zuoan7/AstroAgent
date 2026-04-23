from typing import Optional

from src.memory.api.dto import DeleteMemoryRequest
from src.memory.application.compression_service import CompressionService
from src.memory.application.deletion_service import DeletionService
from src.memory.application.memory_read_service import MemoryReadService
from src.memory.domain.events import MemoryEventType
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
        return self.deletion_service.delete_memory(
            tenant_id=request.tenant_id or self.tenant_id,
            scope=request.scope,
            selector=request.selector,
            requested_by=request.requested_by,
        )

    def get_raw_artifact(self, artifact_id: str) -> Optional[str]:
        return self.artifact_store.get_content(artifact_id)
