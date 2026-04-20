import time
from typing import Any, Dict, Optional

from src.memory.domain.deletion import DeletionJob, DeletionScope
from src.memory.domain.events import MemoryEvent, MemoryEventType
from src.memory.infrastructure.repositories.artifact_store import ArtifactStore
from src.memory.infrastructure.repositories.deletion_repo import DeletionRepository
from src.memory.infrastructure.repositories.event_store import EventStore
from src.memory.infrastructure.repositories.summary_snapshot_repo import SummarySnapshotRepository
from src.memory.infrastructure.repositories.task_state_repo import TaskStateRepository
from src.memory.short_term_memory.repository import ShortTermMemoryRepository


class DeletionService:
    """Basic tombstone deletion service for raw records and legacy projections."""

    def __init__(
        self,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        deletion_repository: DeletionRepository,
        task_state_repository: TaskStateRepository,
        summary_snapshot_repository: SummarySnapshotRepository,
        short_term_repository: ShortTermMemoryRepository,
    ):
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.deletion_repository = deletion_repository
        self.task_state_repository = task_state_repository
        self.summary_snapshot_repository = summary_snapshot_repository
        self.short_term_repository = short_term_repository

    def delete_memory(
        self,
        tenant_id: str,
        scope: str,
        selector: Dict[str, Any],
        requested_by: Optional[str] = None,
    ) -> DeletionJob:
        """Apply a tombstone deletion and return the resulting job."""

        normalized_scope = DeletionScope(scope).value
        session_id = selector.get("session_id")
        job = DeletionJob(
            tenant_id=tenant_id,
            session_id=session_id,
            delete_scope=normalized_scope,
            selector=selector,
            requested_by=requested_by,
        )
        self.deletion_repository.create_job(job)
        job.status = "running"
        job.started_at = time.time()
        try:
            result = self._apply_delete(tenant_id, normalized_scope, {**selector, "job_id": job.job_id}, requested_by)
            job.status = "completed"
            job.result = result
        except Exception as exc:
            job.status = "failed"
            job.result = {"error": f"{type(exc).__name__}: {exc}"}
            raise
        finally:
            job.finished_at = time.time()
            self.deletion_repository.update_job(job)
        return job

    def _apply_delete(
        self,
        tenant_id: str,
        scope: str,
        selector: Dict[str, Any],
        requested_by: Optional[str],
    ) -> Dict[str, Any]:
        session_id = selector.get("session_id")
        if not session_id:
            raise ValueError("selector.session_id is required")

        result: Dict[str, Any] = {
            "events_marked": 0,
            "artifacts_marked": 0,
            "legacy_rows_deleted": 0,
            "task_states_marked": 0,
            "snapshots_marked": 0,
        }

        if scope == DeletionScope.SESSION.value:
            result["events_marked"] = self.event_store.mark_deleted_by_session(session_id)
            result["artifacts_marked"] = self.artifact_store.mark_deleted_by_session(session_id)
            result["task_states_marked"] = int(self.task_state_repository.mark_deleted(session_id))
            result["snapshots_marked"] = self.summary_snapshot_repository.mark_deleted_by_session(session_id)
            self.short_term_repository.clear_session(session_id)
        elif scope == DeletionScope.MESSAGE.value:
            message_id = selector.get("message_id")
            if not message_id:
                raise ValueError("selector.message_id is required for message deletion")
            result["events_marked"] = self.event_store.mark_deleted_by_source(session_id, "message", message_id)
            result["legacy_rows_deleted"] = self.short_term_repository.delete_message(session_id, message_id)
        elif scope == DeletionScope.TOOL_CALL.value:
            tool_call_id = selector.get("tool_call_id")
            if not tool_call_id:
                raise ValueError("selector.tool_call_id is required for tool_call deletion")
            result["events_marked"] = self.event_store.mark_deleted_by_source(session_id, "tool_call", tool_call_id)
            result["artifacts_marked"] = self.artifact_store.mark_deleted_by_tool_call(session_id, tool_call_id)
            result["legacy_rows_deleted"] = self.short_term_repository.delete_tool_call(session_id, tool_call_id)
        elif scope == DeletionScope.FACT.value:
            fact_id = selector.get("fact_id")
            if not fact_id:
                raise ValueError("selector.fact_id is required for fact deletion")
            result["events_marked"] = self.event_store.mark_deleted_by_source(session_id, "fact", fact_id)
            result["legacy_rows_deleted"] = self.short_term_repository.delete_fact(session_id, fact_id)
        elif scope == DeletionScope.TIME_RANGE.value:
            start_time = selector.get("start_time")
            end_time = selector.get("end_time")
            result["events_marked"] = self.event_store.mark_deleted_by_session(session_id, start_time, end_time)
            result["legacy_rows_deleted"] = self.short_term_repository.delete_by_time_range(
                session_id, start_time, end_time
            )
        elif scope == DeletionScope.TURN.value:
            turn_id = selector.get("turn_id")
            if not turn_id:
                raise ValueError("selector.turn_id is required for turn deletion")
            events = [event.event_id for event in self.event_store.list_by_session(session_id) if event.turn_id == turn_id]
            result["events_marked"] = self.event_store.mark_deleted(events)
        else:
            raise ValueError(f"unsupported deletion scope: {scope}")

        self.event_store.append(
            MemoryEvent(
                tenant_id=tenant_id,
                session_id=session_id,
                event_type=MemoryEventType.MEMORY_DELETED.value,
                source_type="deletion_job",
                source_id=selector.get("job_id"),
                payload={"scope": scope, "selector": selector, "result": result},
                created_by=requested_by,
            )
        )
        self.deletion_repository.append_audit(
            tenant_id=tenant_id,
            session_id=session_id,
            action_type="memory_deleted",
            target_type=scope,
            target_id=self._target_id(scope, selector),
            actor_id=requested_by,
            metadata=result,
        )
        return result

    def _target_id(self, scope: str, selector: Dict[str, Any]) -> Optional[str]:
        return (
            selector.get("message_id")
            or selector.get("tool_call_id")
            or selector.get("fact_id")
            or selector.get("turn_id")
            or selector.get("session_id")
        )
