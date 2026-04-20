from typing import Optional, Sequence

from src.memory.domain.events import MemoryEvent, MemoryEventType
from src.memory.domain.summary_snapshot import SummarySnapshot
from src.memory.infrastructure.repositories.event_store import EventStore
from src.memory.infrastructure.repositories.summary_snapshot_repo import SummarySnapshotRepository


class SummarySnapshotManager:
    """Creates and stores summary snapshots as rebuildable projections."""

    def __init__(self, repository: SummarySnapshotRepository, event_store: EventStore):
        self.repository = repository
        self.event_store = event_store

    def create_snapshot(
        self,
        tenant_id: str,
        session_id: str,
        summary_text: str,
        covered_events: Sequence[MemoryEvent],
        snapshot_type: str = "working",
        quality_score: Optional[float] = None,
        created_by_model: Optional[str] = None,
        supersede_latest: bool = True,
    ) -> SummarySnapshot:
        """Persist a new summary snapshot and append its creation event."""

        covered_from = covered_events[0].event_id if covered_events else None
        covered_to = covered_events[-1].event_id if covered_events else None
        snapshot = SummarySnapshot(
            tenant_id=tenant_id,
            session_id=session_id,
            snapshot_type=snapshot_type,
            covered_from_event_id=covered_from,
            covered_to_event_id=covered_to,
            summary_text=summary_text,
            quality_score=quality_score,
            source_count=len(covered_events),
            created_by_model=created_by_model,
        )
        latest = self.repository.get_latest(session_id, snapshot_type=snapshot_type)
        self.repository.save(snapshot)
        if supersede_latest and latest:
            self.repository.mark_superseded(latest.snapshot_id, snapshot.snapshot_id)
        self.event_store.append(
            MemoryEvent(
                tenant_id=tenant_id,
                session_id=session_id,
                event_type=MemoryEventType.SUMMARY_SNAPSHOT_CREATED.value,
                source_type="summary_snapshot",
                source_id=snapshot.snapshot_id,
                payload=snapshot.to_dict(),
            )
        )
        return snapshot

    def get_latest(self, session_id: str, snapshot_type: str = "working") -> Optional[SummarySnapshot]:
        """Return the newest active snapshot."""

        return self.repository.get_latest(session_id, snapshot_type=snapshot_type)
