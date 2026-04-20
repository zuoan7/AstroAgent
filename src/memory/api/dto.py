from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AppendMessageRequest:
    """Input for appending a conversational message."""

    session_id: str
    role: str
    content: str
    tenant_id: str = "default"
    user_id: Optional[str] = None
    turn_id: Optional[str] = None
    timestamp: Optional[float] = None
    importance: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None


@dataclass
class AppendToolCallRequest:
    """Input for appending a tool call with raw output artifact storage."""

    session_id: str
    tool_name: str
    tool_input: str
    raw_output: str
    tenant_id: str = "default"
    user_id: Optional[str] = None
    turn_id: Optional[str] = None
    timestamp: Optional[float] = None
    success: bool = True
    content_type: str = "text/plain"
    event_id: Optional[str] = None


@dataclass
class BuildContextRequest:
    """Input for building memory context."""

    session_id: str
    query: str = ""
    tenant_id: str = "default"
    max_tokens: Optional[int] = None


@dataclass
class DeleteMemoryRequest:
    """Input for scoped memory deletion."""

    scope: str
    selector: Dict[str, Any]
    tenant_id: str = "default"
    requested_by: Optional[str] = None
