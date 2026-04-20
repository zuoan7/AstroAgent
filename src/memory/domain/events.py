import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


def new_memory_id(prefix: str) -> str:
    """Return a compact id suitable for memory-domain records."""

    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class MemoryEventType(str, Enum):
    """Append-only event types recorded by the memory event store."""

    MESSAGE_CREATED = "message_created"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_FINISHED = "tool_call_finished"
    TOOL_CALL_FAILED = "tool_call_failed"
    FACT_EXTRACTED = "fact_extracted"
    SUMMARY_SNAPSHOT_CREATED = "summary_snapshot_created"
    TASK_STATE_UPDATED = "task_state_updated"
    MEMORY_DELETED = "memory_deleted"


@dataclass
class MemoryEvent:
    """Raw memory event used as the source of truth for projections."""

    tenant_id: str
    session_id: str
    event_type: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: new_memory_id("evt"))
    turn_id: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    schema_version: int = 1
    created_at: float = field(default_factory=time.time)
    created_by: Optional[str] = None
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "event_type": self.event_type,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "payload": self.payload,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEvent":
        return cls(
            event_id=data["event_id"],
            tenant_id=data["tenant_id"],
            session_id=data["session_id"],
            turn_id=data.get("turn_id"),
            event_type=data["event_type"],
            source_type=data.get("source_type"),
            source_id=data.get("source_id"),
            payload=data.get("payload") or {},
            schema_version=int(data.get("schema_version", 1)),
            created_at=float(data.get("created_at", time.time())),
            created_by=data.get("created_by"),
            is_deleted=bool(data.get("is_deleted", False)),
        )
