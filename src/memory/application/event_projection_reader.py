from src.memory.core.models import Message, SalientFact, ToolCallRecord
from src.memory.domain.events import MemoryEvent, MemoryEventType
from src.memory.infrastructure.repositories.event_store import EventStore


class EventProjectionReader:
    """Reads prompt-facing projections from the append-only event store."""

    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    def list_messages(self, session_id: str, limit: int = 200) -> list[Message]:
        events = self.event_store.list_by_session(
            session_id,
            event_type=MemoryEventType.MESSAGE_CREATED.value,
            limit=limit,
            descending=True,
        )
        return [self._message_from_event(event) for event in events]

    def list_tool_calls(self, session_id: str, limit: int = 80) -> list[ToolCallRecord]:
        events = self.event_store.list_by_session(
            session_id,
            event_types=[
                MemoryEventType.TOOL_CALL_FINISHED.value,
                MemoryEventType.TOOL_CALL_FAILED.value,
            ],
            limit=limit,
            descending=True,
        )
        return [self._tool_call_from_event(event) for event in events]

    def list_salient_facts(self, session_id: str, limit: int = 100) -> list[SalientFact]:
        events = self.event_store.list_by_session(
            session_id,
            event_type=MemoryEventType.FACT_EXTRACTED.value,
            limit=limit,
            descending=True,
        )
        return [self._fact_from_event(event) for event in events]

    def _message_from_event(self, event: MemoryEvent) -> Message:
        payload = event.payload or {}
        data = {
            "message_id": event.source_id or payload.get("message_id"),
            "session_id": event.session_id,
            "role": payload.get("role", "user"),
            "content": payload.get("content", ""),
            "timestamp": payload.get("timestamp", event.created_at),
            "importance": payload.get("importance", 0),
            "importance_reason": payload.get("importance_reason", ""),
            "message_type": payload.get("message_type", "chat"),
            "metadata": payload.get("metadata", {}) or {},
        }
        return Message.from_dict(data)

    def _tool_call_from_event(self, event: MemoryEvent) -> ToolCallRecord:
        payload = event.payload or {}
        data = {
            "tool_call_id": event.source_id or payload.get("tool_call_id"),
            "tool_name": payload.get("tool_name", "tool"),
            "timestamp": payload.get("timestamp", event.created_at),
            "input_summary": payload.get("input_summary", payload.get("tool_input", "")),
            "output_digest": payload.get("output_digest", ""),
            "output_summary": payload.get(
                "output_summary", payload.get("result_summary", "")
            ),
            "output_is_summary": payload.get("output_is_summary", False),
            "output_is_truncated": payload.get("output_is_truncated", False),
            "raw_artifact_id": payload.get("raw_artifact_id", ""),
            "raw_size_bytes": payload.get("raw_size_bytes", 0),
            "content_type": payload.get("content_type", "text/plain"),
            "status": payload.get(
                "status",
                (
                    "success"
                    if event.event_type == MemoryEventType.TOOL_CALL_FINISHED.value
                    else "error"
                ),
            ),
            "importance": payload.get("importance", 1),
        }
        return ToolCallRecord.from_dict(data)

    def _fact_from_event(self, event: MemoryEvent) -> SalientFact:
        payload = event.payload or {}
        data = {
            "fact_id": event.source_id or payload.get("fact_id"),
            "fact_type": payload.get("fact_type", "fact"),
            "content": payload.get("content", ""),
            "timestamp": payload.get("timestamp", event.created_at),
            "source_type": payload.get("source_type", event.source_type or ""),
            "source_id": payload.get("source_id", event.source_id or ""),
            "source": payload.get("source", ""),
        }
        return SalientFact.from_dict(data)
