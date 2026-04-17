import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol


TEXTUAL_EVENT_TYPES = {
    "raw_text_delta",
    "thinking_delta",
    "final_answer_delta",
    "reasoning_summary",
    "warning",
    "error",
    "transcription",
}


@dataclass(slots=True)
class StreamEvent:
    type: str
    content: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)
    sequence: Optional[int] = None
    modality: str = "text"
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def validate(self) -> "StreamEvent":
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("event.type must be a non-empty string")
        if not isinstance(self.meta, dict):
            raise ValueError("event.meta must be a dict")
        if self.sequence is not None and (
            not isinstance(self.sequence, int) or self.sequence < 0
        ):
            raise ValueError("event.sequence must be a non-negative int")
        if self.type in TEXTUAL_EVENT_TYPES and self.content is None:
            raise ValueError(f"{self.type} requires textual content")
        if self.type == "image" and not isinstance(self.content, str):
            raise ValueError("image events require a string URL content")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "content": self.content,
            "meta": self.meta,
            "sequence": self.sequence,
            "modality": self.modality,
            "timestamp": self.timestamp,
        }


class StreamEventProcessor(Protocol):
    def process(self, event: StreamEvent) -> Optional[StreamEvent | Iterable[StreamEvent]]:
        ...


class StreamEventAdapter(Protocol):
    def adapt(self, event: StreamEvent) -> Optional[Any]:
        ...


def apply_event_processors(
    event: StreamEvent,
    processors: Iterable[StreamEventProcessor],
) -> list[StreamEvent]:
    queue = [event.validate()]
    for processor in processors:
        next_queue: list[StreamEvent] = []
        for item in queue:
            result = processor.process(item)
            if result is None:
                continue
            if isinstance(result, StreamEvent):
                next_queue.append(result.validate())
            else:
                for child in result:
                    if not isinstance(child, StreamEvent):
                        raise ValueError("event processors must emit StreamEvent instances")
                    next_queue.append(child.validate())
        queue = next_queue
        if not queue:
            break
    return queue


class PlainTextStreamAdapter:
    def adapt(self, event: StreamEvent) -> Optional[str]:
        event.validate()
        if event.type in {"raw_text_delta", "warning", "error"}:
            return "" if event.content is None else str(event.content)
        return None


class FrontendJsonEventAdapter:
    def adapt(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        event.validate()

        if event.type == "thinking_delta":
            return {
                "type": "thinking",
                "content": str(event.content),
                "meta": event.meta,
            }
        if event.type in {"final_answer_delta", "warning", "error"}:
            return {
                "type": "text",
                "content": str(event.content),
                "meta": event.meta,
            }
        if event.type == "image":
            return {
                "type": "image",
                "url": str(event.content),
                "meta": event.meta,
            }
        if event.type == "transcription":
            return {
                "type": "transcription",
                "text": str(event.content),
                "meta": event.meta,
            }
        if event.type in {
            "tool_start",
            "tool_end",
            "plan_update",
            "step_start",
            "step_end",
            "evidence_found",
            "memory_hit",
            "final_answer",
            "reasoning_summary",
        }:
            payload = {
                "type": event.type,
                "meta": event.meta,
                "timestamp": event.timestamp,
                "sequence": event.sequence,
            }
            if isinstance(event.content, dict):
                payload.update(event.content)
            else:
                payload["content"] = event.content
            return payload
        return None


class SSEEventAdapter:
    def __init__(self, inner: Optional[StreamEventAdapter] = None):
        self._inner = inner or FrontendJsonEventAdapter()

    @staticmethod
    def serialize_payload(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def adapt(self, event: StreamEvent) -> Optional[str]:
        payload = self._inner.adapt(event)
        if payload is None:
            return None
        return self.serialize_payload(payload)
