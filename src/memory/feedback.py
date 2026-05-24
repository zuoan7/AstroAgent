"""记忆反馈事件模型。

本文件提供轻量的内部反馈记录对象，用于统一表达长期记忆注入后的 shown、
hit、miss、denied 等结果。当前阶段只写回单条记忆 metadata，不引入全局
反馈表。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class MemoryFeedbackRecord:
    """单条记忆在一次注入后的反馈结果。"""

    user_id: str
    memory_id: str
    memory_type: str
    task_type: str
    query: str
    outcome: str
    source: str = "long_term_injection"
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可写入 metadata 或 trace 的稳定字典。"""

        return {
            "user_id": self.user_id,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "task_type": self.task_type,
            "query": self.query,
            "outcome": self.outcome,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MemoryFeedbackRecord":
        """从 event_log metadata 中恢复反馈记录。"""

        return cls(
            user_id=str(data.get("user_id") or ""),
            memory_id=str(data.get("memory_id") or ""),
            memory_type=str(data.get("memory_type") or ""),
            task_type=str(data.get("task_type") or ""),
            query=str(data.get("query") or ""),
            outcome=str(data.get("outcome") or ""),
            source=str(data.get("source") or "long_term_injection"),
            timestamp=str(data.get("timestamp") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


__all__ = ["MemoryFeedbackRecord"]
