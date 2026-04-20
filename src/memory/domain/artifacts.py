import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.memory.domain.events import new_memory_id


@dataclass
class ToolArtifact:
    """Raw tool output metadata and storage pointer."""

    tenant_id: str
    session_id: str
    tool_call_id: str
    storage_uri: str
    sha256: str
    size_bytes: int
    artifact_id: str = field(default_factory=lambda: new_memory_id("art"))
    content_type: str = "text/plain"
    encoding: str = "utf-8"
    compression_codec: Optional[str] = None
    schema_name: Optional[str] = None
    schema_version: Optional[int] = None
    preview_text: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "tool_call_id": self.tool_call_id,
            "storage_uri": self.storage_uri,
            "content_type": self.content_type,
            "encoding": self.encoding,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "compression_codec": self.compression_codec,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "preview_text": self.preview_text,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_deleted": self.is_deleted,
        }
