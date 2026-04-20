import time
from typing import Any, Dict, Optional

from src.memory.domain.events import MemoryEvent, MemoryEventType
from src.memory.domain.task_state import TaskState, TaskStateConflictError
from src.memory.infrastructure.repositories.event_store import EventStore
from src.memory.infrastructure.repositories.task_state_repo import TaskStateRepository


class TaskStateManager:
    """Coordinates TaskState projection updates with raw events."""

    _LIST_FIELDS = {
        "active_constraints",
        "completed_steps",
        "pending_steps",
        "open_questions",
        "assumptions",
        "blockers",
    }

    _SCALAR_FIELDS = {
        "current_goal",
        "next_action",
        "status",
        "confidence",
    }

    def __init__(self, repository: TaskStateRepository, event_store: EventStore):
        self.repository = repository
        self.event_store = event_store

    def get_state(self, tenant_id: str, session_id: str) -> TaskState:
        state = self.repository.get(session_id)
        if state:
            return state
        state = TaskState(tenant_id=tenant_id, session_id=session_id)
        return self.repository.save(state)

    def patch_state(
        self,
        tenant_id: str,
        session_id: str,
        patch: Dict[str, Any],
        expected_version: Optional[int] = None,
        created_by: Optional[str] = None,
    ) -> TaskState:
        """Apply a shallow patch and record the change as an append-only event."""

        state = self.get_state(tenant_id, session_id)
        if expected_version is not None and state.version != expected_version:
            raise TaskStateConflictError(
                f"task state version conflict: expected {expected_version}, got {state.version}"
            )

        for field in self._SCALAR_FIELDS:
            if field in patch:
                setattr(state, field, patch[field] if patch[field] is not None else "")
        for field in self._LIST_FIELDS:
            if field in patch:
                value = patch[field] or []
                if not isinstance(value, list):
                    raise ValueError(f"{field} must be a list")
                setattr(state, field, value)

        state.version += 1
        state.updated_at = time.time()
        event = MemoryEvent(
            tenant_id=tenant_id,
            session_id=session_id,
            event_type=MemoryEventType.TASK_STATE_UPDATED.value,
            source_type="task_state",
            source_id=state.task_state_id,
            payload={"patch": patch, "state": state.to_dict()},
            created_by=created_by,
        )
        stored_event = self.event_store.append(event)
        state.updated_from_event_id = stored_event.event_id
        return self.repository.save(state)
