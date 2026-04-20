import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Message:
    role: str
    content: str
    timestamp: float
    message_id: str = field(default_factory=lambda: _new_id("msg"))
    session_id: str = ""
    importance: int = 0
    importance_reason: str = ""
    message_type: str = "chat"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "importance_reason": self.importance_reason,
            "message_type": self.message_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            message_id=data.get("message_id") or _new_id("msg"),
            session_id=data.get("session_id", ""),
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"],
            importance=data.get("importance", 0),
            importance_reason=data.get("importance_reason", ""),
            message_type=data.get("message_type", "chat"),
            metadata=data.get("metadata", {}) or {},
        )


@dataclass
class ToolCallRecord:
    tool_name: str
    timestamp: float
    tool_call_id: str = field(default_factory=lambda: _new_id("tool"))
    input_summary: str = ""
    output_digest: str = ""
    output_summary: str = ""
    output_is_summary: bool = False
    output_is_truncated: bool = False
    raw_artifact_id: str = ""
    raw_size_bytes: int = 0
    content_type: str = "text/plain"
    status: str = "success"
    importance: int = 1

    def __post_init__(self):
        if not self.tool_call_id:
            self.tool_call_id = _new_id("tool")
        self.raw_size_bytes = int(self.raw_size_bytes or 0)
        self.content_type = self.content_type or "text/plain"

    @property
    def tool_input(self) -> str:
        return self.input_summary

    @property
    def result_summary(self) -> str:
        return self.output_summary

    @property
    def success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "input_summary": self.input_summary,
            "output_digest": self.output_digest,
            "output_summary": self.output_summary,
            "output_is_summary": self.output_is_summary,
            "output_is_truncated": self.output_is_truncated,
            "raw_artifact_id": self.raw_artifact_id,
            "raw_size_bytes": self.raw_size_bytes,
            "content_type": self.content_type,
            "status": self.status,
            "importance": self.importance,
            "tool_input": self.input_summary,
            "result_summary": self.output_summary,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallRecord":
        status = data.get("status")
        if not status:
            status = "success" if data.get("success", True) else "error"
        return cls(
            tool_call_id=data.get("tool_call_id") or _new_id("tool"),
            tool_name=data["tool_name"],
            timestamp=data["timestamp"],
            input_summary=data.get("input_summary", data.get("tool_input", "")),
            output_digest=data.get("output_digest", ""),
            output_summary=data.get("output_summary", data.get("result_summary", "")),
            output_is_summary=data.get("output_is_summary", False),
            output_is_truncated=data.get("output_is_truncated", False),
            raw_artifact_id=data.get("raw_artifact_id", ""),
            raw_size_bytes=int(data.get("raw_size_bytes", 0) or 0),
            content_type=data.get("content_type", "text/plain") or "text/plain",
            status=status,
            importance=data.get("importance", 1),
        )


@dataclass
class SalientFact:
    fact_type: str
    content: str
    timestamp: float
    fact_id: str = field(default_factory=lambda: _new_id("fact"))
    source_type: str = ""
    source_id: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SalientFact":
        return cls(
            fact_id=data.get("fact_id") or _new_id("fact"),
            fact_type=data["fact_type"],
            content=data["content"],
            timestamp=data["timestamp"],
            source_type=data.get("source_type", ""),
            source_id=data.get("source_id", ""),
            source=data.get("source", ""),
        )


@dataclass
class SessionMemoryState:
    session_id: str
    summary: str = ""
    trimmed_count: int = 0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "summary": self.summary,
            "trimmed_count": self.trimmed_count,
            "updated_at": self.updated_at,
        }
