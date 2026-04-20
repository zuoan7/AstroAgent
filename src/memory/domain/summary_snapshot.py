import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.memory.domain.events import new_memory_id


@dataclass
class SummarySnapshot:
    """Versioned summary snapshot covering a range of raw events."""

    tenant_id: str
    session_id: str
    summary_text: str
    snapshot_id: str = field(default_factory=lambda: new_memory_id("snap"))
    snapshot_type: str = "working"
    covered_from_event_id: Optional[str] = None
    covered_to_event_id: Optional[str] = None
    summary_level: str = "working"
    quality_score: Optional[float] = None
    source_count: int = 0
    created_by_model: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    superseded_by: Optional[str] = None
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "snapshot_type": self.snapshot_type,
            "covered_from_event_id": self.covered_from_event_id,
            "covered_to_event_id": self.covered_to_event_id,
            "summary_text": self.summary_text,
            "summary_level": self.summary_level,
            "quality_score": self.quality_score,
            "source_count": self.source_count,
            "created_by_model": self.created_by_model,
            "created_at": self.created_at,
            "superseded_by": self.superseded_by,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SummarySnapshot":
        return cls(
            snapshot_id=data.get("snapshot_id") or new_memory_id("snap"),
            tenant_id=data["tenant_id"],
            session_id=data["session_id"],
            snapshot_type=data.get("snapshot_type", "working") or "working",
            covered_from_event_id=data.get("covered_from_event_id"),
            covered_to_event_id=data.get("covered_to_event_id"),
            summary_text=data.get("summary_text", "") or "",
            summary_level=data.get("summary_level", "working") or "working",
            quality_score=data.get("quality_score"),
            source_count=int(data.get("source_count", 0) or 0),
            created_by_model=data.get("created_by_model"),
            created_at=float(data.get("created_at", time.time())),
            superseded_by=data.get("superseded_by"),
            is_deleted=bool(data.get("is_deleted", False)),
        )
