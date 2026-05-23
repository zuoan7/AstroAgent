"""任务状态投影管理器。

TaskState 是短期记忆中用于跨轮跟踪目标、约束、待办和阻塞的结构化投影。
本文件负责 patch 归一化、乐观版本检查，以及追加 task_state_updated 事件。
"""

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

    _VALID_STATUSES = {"active", "running", "completed", "awaiting_user", "blocked"}
    _LIST_LIMITS = {
        "active_constraints": 12,
        "completed_steps": 20,
        "pending_steps": 12,
        "open_questions": 10,
        "assumptions": 10,
        "blockers": 10,
    }

    def __init__(self, repository: TaskStateRepository, event_store: EventStore):
        self.repository = repository
        self.event_store = event_store

    def get_state(self, tenant_id: str, session_id: str) -> TaskState:
        """读取任务状态；会话首次访问时创建默认 active 状态。"""

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
        turn_id: Optional[str] = None,
    ) -> TaskState:
        """Apply a shallow patch and record the change as an append-only event."""

        state = self.get_state(tenant_id, session_id)
        if expected_version is not None and state.version != expected_version:
            raise TaskStateConflictError(
                f"task state version conflict: expected {expected_version}, got {state.version}"
            )

        for field in self._SCALAR_FIELDS:
            if field in patch:
                value = patch[field]
                if field == "status":
                    normalized_status = self._normalize_status(value)
                    if normalized_status is None:
                        continue
                    setattr(state, field, normalized_status)
                elif field == "confidence":
                    setattr(state, field, self._normalize_confidence(value))
                else:
                    setattr(state, field, str(value or "").strip())
        for field in self._LIST_FIELDS:
            if field in patch:
                value = patch[field] or []
                if not isinstance(value, list):
                    raise ValueError(f"{field} must be a list")
                setattr(state, field, self._normalize_list(field, value))

        state.version += 1
        state.updated_at = time.time()
        event = MemoryEvent(
            tenant_id=tenant_id,
            session_id=session_id,
            turn_id=turn_id,
            event_type=MemoryEventType.TASK_STATE_UPDATED.value,
            source_type="task_state",
            source_id=state.task_state_id,
            payload={"patch": patch, "state": state.to_dict()},
            created_by=created_by,
        )
        stored_event = self.event_store.append(event)
        state.updated_from_event_id = stored_event.event_id
        return self.repository.save(state)

    def _normalize_list(self, field: str, value: list[Any]) -> list[str]:
        """清洗列表字段，去空、去重并按字段限制最大长度。"""

        limit = self._LIST_LIMITS.get(field, 20)
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
            if len(normalized) >= limit:
                break
        return normalized

    def _normalize_status(self, value: Any) -> Optional[str]:
        """只接受任务状态枚举内的状态值。"""

        text = str(value or "").strip()
        if text in self._VALID_STATUSES:
            return text
        return None

    @staticmethod
    def _normalize_confidence(value: Any) -> Optional[float]:
        """把置信度归一化到 0.0 到 1.0，非法输入返回 None。"""

        if value is None or value == "":
            return None
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        return min(max(confidence, 0.0), 1.0)
