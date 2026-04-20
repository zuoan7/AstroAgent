import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from src.memory.domain.events import new_memory_id


class DeletionScope(str, Enum):
    """Supported memory deletion scopes."""

    SESSION = "session"
    TURN = "turn"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    FACT = "fact"
    TIME_RANGE = "time_range"


@dataclass
class DeletionJob:
    """Deletion request and execution result."""

    tenant_id: str
    delete_scope: str
    selector: Dict[str, Any]
    job_id: str = field(default_factory=lambda: new_memory_id("del"))
    session_id: Optional[str] = None
    status: str = "pending"
    requested_by: Optional[str] = None
    requested_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "delete_scope": self.delete_scope,
            "selector": self.selector,
            "status": self.status,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
        }
