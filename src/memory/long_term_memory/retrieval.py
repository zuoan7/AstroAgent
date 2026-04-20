from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.memory.long_term_memory.models import MemoryItem, MemoryQuery, MemoryStatus, MemoryType
from src.memory.long_term_memory.repository import LongTermMemoryRepository


TASK_TYPE_MEMORY_MAP: Dict[str, List[str]] = {
    "qa": [MemoryType.PREFERENCE, MemoryType.BACKGROUND, MemoryType.FACT],
    "creative": [MemoryType.PREFERENCE, MemoryType.BACKGROUND],
    "analysis": [MemoryType.PREFERENCE, MemoryType.BACKGROUND, MemoryType.FACT],
    "observation": [MemoryType.PREFERENCE, MemoryType.HABIT, MemoryType.BACKGROUND, MemoryType.CONSTRAINT],
    "learning": [MemoryType.PREFERENCE, MemoryType.BACKGROUND],
    "general": [MemoryType.PREFERENCE, MemoryType.HABIT, MemoryType.CONSTRAINT, MemoryType.BACKGROUND, MemoryType.FACT],
}

ASTRONOMY_KEYWORDS: Set[str] = {
    "观测", "望远镜", "行星", "恒星", "星系", "星云", "星团",
    "流星", "彗星", "月相", "日食", "月食", "冲日", "合日",
    "拍摄", "摄影", "深空", "赤道仪", "导星", "曝光",
}
OBSERVATION_KEYWORDS: Set[str] = {"今晚", "观测", "天气", "可见", "升起", "落下", "最佳", "推荐", "目标", "望远镜", "拍摄"}
LEARNING_KEYWORDS: Set[str] = {"什么是", "为什么", "怎么", "如何", "原理", "解释", "科普", "入门", "学习", "了解"}


@dataclass
class RetrievalHit:
    item: MemoryItem
    score: float
    reasons: List[str]


class LongTermMemoryRetriever:
    """Query-aware selector for prompt injection and explainability."""

    def __init__(
        self,
        repository: LongTermMemoryRepository,
        relevance_threshold: float = 0.3,
        max_memories: int = 15,
    ):
        self._repo = repository
        self.relevance_threshold = relevance_threshold
        self.max_memories = max_memories
        self.type_weights = {
            MemoryType.PREFERENCE: 1.0,
            MemoryType.HABIT: 0.7,
            MemoryType.CONSTRAINT: 1.2,
            MemoryType.BACKGROUND: 0.8,
            MemoryType.FACT: 0.9,
        }

    def classify_task_type(self, query: str) -> str:
        text = query.lower()
        obs_score = sum(1 for kw in OBSERVATION_KEYWORDS if kw in text)
        learn_score = sum(1 for kw in LEARNING_KEYWORDS if kw in text)
        astro_score = sum(1 for kw in ASTRONOMY_KEYWORDS if kw in text)
        if obs_score >= 2:
            return "observation"
        if learn_score >= 1:
            return "learning"
        if astro_score >= 2:
            return "qa"
        return "general"

    def score(self, item: MemoryItem, query: str, task_type: str) -> Tuple[float, List[str]]:
        reasons = [f"任务类型={task_type}"]
        score = item.confidence * self.type_weights.get(item.memory_type, 1.0)

        if item.source_type == "explicit":
            score *= 1.3
            reasons.append("来自用户显式表达")
        if item.confirmed_by_user:
            score *= 1.2
            reasons.append("已获用户确认")

        if item.memory_type in TASK_TYPE_MEMORY_MAP.get(task_type, []):
            score *= 1.1
            reasons.append("类型适配当前任务")
        else:
            score *= 0.6

        if item.memory_type == MemoryType.CONSTRAINT:
            score *= 1.5
            reasons.append("约束类记忆优先注入")

        tokens = [token for token in query.lower().split() if len(token) > 1]
        key_text = str(item.key).lower()
        value_text = str(item.value).lower() if item.value is not None else ""
        if any(token in value_text for token in tokens):
            score *= 1.2
            reasons.append("值与 query 命中")
        if any(token in key_text for token in tokens):
            score *= 1.1
            reasons.append("key 与 query 命中")

        return min(score, 2.0), reasons

    def retrieve(self, user_id: str, query: str, task_type: Optional[str] = None, limit: Optional[int] = None) -> List[RetrievalHit]:
        resolved_task_type = task_type or self.classify_task_type(query)
        items = self._repo.query_memories(
            MemoryQuery(
                user_id=user_id,
                status=MemoryStatus.ACTIVE,
                min_confidence=self.relevance_threshold,
                limit=100,
            )
        )
        hits: List[RetrievalHit] = []
        for item in items:
            score, reasons = self.score(item, query, resolved_task_type)
            if score >= self.relevance_threshold:
                hits.append(RetrievalHit(item=item, score=score, reasons=reasons))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: limit or self.max_memories]
