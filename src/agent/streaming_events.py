"""流式事件模型与适配器，统一内部事件校验、处理器链、纯文本/JSON/SSE 输出格式。
"""

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
    """内部标准流式事件结构，承载内容、元信息、序号和模态。"""
    type: str
    content: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)
    sequence: Optional[int] = None
    modality: str = "text"
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def validate(self) -> "StreamEvent":
        """校验流式事件字段是否合法并返回事件自身。"""
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
        """将当前对象转换为 dict 相关的兼容结构。"""
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
    """流式事件处理器协议，允许事件在输出前被过滤或拆分。"""
    def process(self, event: StreamEvent) -> Optional[StreamEvent | Iterable[StreamEvent]]:
        """处理单个内部流式事件并返回过滤或拆分后的事件。"""
        ...


class StreamEventAdapter(Protocol):
    """流式事件适配器协议，将内部事件转换为具体输出格式。"""
    def adapt(self, event: StreamEvent) -> Optional[Any]:
        """把内部流式事件转换为目标输出格式。"""
        ...


def apply_event_processors(
    event: StreamEvent,
    processors: Iterable[StreamEventProcessor],
) -> list[StreamEvent]:
    """按顺序运行事件处理器链并返回最终事件列表。"""
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
    """纯文本适配器，只输出最终答案、警告和错误文本。"""
    def adapt(self, event: StreamEvent) -> Optional[str]:
        """把内部流式事件转换为目标输出格式。"""
        event.validate()
        if event.type in {"final_answer_delta", "warning", "error"}:
            return "" if event.content is None else str(event.content)
        return None


class FrontendJsonEventAdapter:
    """前端 JSON 适配器，把内部事件转换为工作台事件结构。"""
    def adapt(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """把内部流式事件转换为目标输出格式。"""
        event.validate()

        if event.type == "thinking_delta":
            return {
                "type": "thinking",
                "content": str(event.content),
                "meta": event.meta,
            }
        if event.type == "final_answer_delta":
            return {
                "type": "text",
                "content": str(event.content),
                "meta": event.meta,
            }
        if event.type in {"warning", "error"}:
            return {
                "type": event.type,
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
            "route_decision",
            "latency_metrics",
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
    """SSE 适配器，把前端 JSON 事件序列化为 Server-Sent Events 文本。"""
    def __init__(self, inner: Optional[StreamEventAdapter] = None):
        """初始化 SSEEventAdapter 的依赖、配置和内部状态。"""
        self._inner = inner or FrontendJsonEventAdapter()

    @staticmethod
    def serialize_payload(payload: Dict[str, Any]) -> str:
        """把 JSON payload 序列化为 SSE data 文本。"""
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def adapt(self, event: StreamEvent) -> Optional[str]:
        """把内部流式事件转换为目标输出格式。"""
        payload = self._inner.adapt(event)
        if payload is None:
            return None
        return self.serialize_payload(payload)
