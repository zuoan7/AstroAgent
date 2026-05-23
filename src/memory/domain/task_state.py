"""短期记忆任务状态领域模型。

TaskState 保存当前任务目标、约束、进度、问题和阻塞，是跨轮追问补全和
上下文检索聚焦的主要结构化信号。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.memory.domain.events import new_memory_id


@dataclass
class TaskState:
    """Structured task progress for a session, independent of summaries."""

    tenant_id: str
    session_id: str
    task_state_id: str = field(default_factory=lambda: new_memory_id("task"))
    current_goal: str = ""
    active_constraints: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    next_action: str = ""
    status: str = "active"
    confidence: Optional[float] = None
    version: int = 1
    updated_from_event_id: Optional[str] = None
    updated_at: float = field(default_factory=time.time)
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """序列化任务状态，供 API、prompt context 和持久化使用。"""

        return {
            "task_state_id": self.task_state_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "current_goal": self.current_goal,
            "active_constraints": list(self.active_constraints),
            "completed_steps": list(self.completed_steps),
            "pending_steps": list(self.pending_steps),
            "open_questions": list(self.open_questions),
            "assumptions": list(self.assumptions),
            "blockers": list(self.blockers),
            "next_action": self.next_action,
            "status": self.status,
            "confidence": self.confidence,
            "version": self.version,
            "updated_from_event_id": self.updated_from_event_id,
            "updated_at": self.updated_at,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskState":
        """从字典恢复任务状态，并填充兼容旧数据的默认值。"""

        return cls(
            task_state_id=data.get("task_state_id") or new_memory_id("task"),
            tenant_id=data["tenant_id"],
            session_id=data["session_id"],
            current_goal=data.get("current_goal", "") or "",
            active_constraints=list(data.get("active_constraints") or []),
            completed_steps=list(data.get("completed_steps") or []),
            pending_steps=list(data.get("pending_steps") or []),
            open_questions=list(data.get("open_questions") or []),
            assumptions=list(data.get("assumptions") or []),
            blockers=list(data.get("blockers") or []),
            next_action=data.get("next_action", "") or "",
            status=data.get("status", "active") or "active",
            confidence=data.get("confidence"),
            version=int(data.get("version", 1)),
            updated_from_event_id=data.get("updated_from_event_id"),
            updated_at=float(data.get("updated_at", time.time())),
            is_deleted=bool(data.get("is_deleted", False)),
        )


class TaskStateConflictError(RuntimeError):
    """Raised when a patch uses a stale task-state version."""
