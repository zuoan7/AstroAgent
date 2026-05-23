"""长期记忆 query-aware 检索。

本文件负责长期记忆注入前的候选召回和打分：先用 P1 规则评分取 Top100，
再融合已有 embedding 缓存的语义 Top100，并输出可解释组件分。它同时读取
注入反馈统计，按 task_type x memory_type 生成运行时自适应类型先验。
"""

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.memory.long_term_memory.models import (
    MemoryItem,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
)
from src.memory.long_term_memory.embedding import MemoryEmbeddingService
from src.memory.long_term_memory.repository import LongTermMemoryRepository


# Legacy export kept for callers/tests that import the old task map. The new
# scoring path uses TASK_TYPE_TYPE_PRIORS instead of filtering by this map.
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


TASK_SCORING_WEIGHTS: Dict[str, Dict[str, float]] = {
    "observation": {
        "confidence": 0.20,
        "type_weight": 0.18,
        "source_bonus": 0.10,
        "query_relevance": 0.27,
        "recency": 0.08,
        "constraint_bonus": 0.14,
        "stale_penalty": 0.03,
    },
    "learning": {
        "confidence": 0.24,
        "type_weight": 0.20,
        "source_bonus": 0.12,
        "query_relevance": 0.25,
        "recency": 0.08,
        "constraint_bonus": 0.08,
        "stale_penalty": 0.03,
    },
    "qa": {
        "confidence": 0.24,
        "type_weight": 0.20,
        "source_bonus": 0.10,
        "query_relevance": 0.27,
        "recency": 0.08,
        "constraint_bonus": 0.08,
        "stale_penalty": 0.03,
    },
    "general": {
        "confidence": 0.25,
        "type_weight": 0.18,
        "source_bonus": 0.12,
        "query_relevance": 0.25,
        "recency": 0.10,
        "constraint_bonus": 0.07,
        "stale_penalty": 0.03,
    },
}

TASK_TYPE_TYPE_PRIORS: Dict[str, Dict[str, float]] = {
    "observation": {
        MemoryType.CONSTRAINT: 1.0,
        MemoryType.BACKGROUND: 0.90,
        MemoryType.HABIT: 0.85,
        MemoryType.PREFERENCE: 0.75,
        MemoryType.FACT: 0.45,
    },
    "learning": {
        MemoryType.PREFERENCE: 0.90,
        MemoryType.BACKGROUND: 0.85,
        MemoryType.FACT: 0.70,
        MemoryType.CONSTRAINT: 0.65,
        MemoryType.HABIT: 0.45,
    },
    "qa": {
        MemoryType.FACT: 0.90,
        MemoryType.BACKGROUND: 0.85,
        MemoryType.PREFERENCE: 0.70,
        MemoryType.CONSTRAINT: 0.65,
        MemoryType.HABIT: 0.45,
    },
    "general": {
        MemoryType.CONSTRAINT: 0.90,
        MemoryType.PREFERENCE: 0.85,
        MemoryType.BACKGROUND: 0.75,
        MemoryType.FACT: 0.70,
        MemoryType.HABIT: 0.65,
    },
}


@dataclass(frozen=True)
class InjectionPolicy:
    """长期记忆注入评分策略。"""

    task_type: str
    weights: Dict[str, float]
    type_weights: Dict[str, float]


@dataclass
class RetrievalHit:
    """长期记忆检索命中，包含分数和可解释原因。"""

    item: MemoryItem
    score: float
    reasons: List[str]
    components: Dict[str, float] = None
    selected: bool = False
    omitted_reason: Optional[str] = None
    token_estimate: int = 0

    def __post_init__(self):
        """补齐组件分字典，避免调用方处理 None。"""

        if self.components is None:
            self.components = {}

    def to_trace_dict(self) -> Dict[str, Any]:
        """序列化为 prompt 注入 trace 使用的调试结构。"""

        return {
            "memory_id": self.item.id,
            "memory_type": self.item.memory_type,
            "category": self.item.category,
            "key": self.item.key,
            "value": self.item.value,
            "score": round(self.score, 3),
            "components": self.components,
            "selected": self.selected,
            "omitted_reason": self.omitted_reason,
            "token_estimate": self.token_estimate,
            "reasons": self.reasons,
        }


class LongTermMemoryRetriever:
    """Query-aware selector for prompt injection and explainability."""

    def __init__(
        self,
        repository: LongTermMemoryRepository,
        relevance_threshold: float = 0.3,
        max_memories: int = 15,
        embedding_service: Optional[MemoryEmbeddingService] = None,
    ):
        """初始化规则检索参数和可选语义 embedding 服务。"""

        self._repo = repository
        self._embedding_service = embedding_service
        self.relevance_threshold = relevance_threshold
        self.max_memories = max_memories
        self.type_weights = {
            MemoryType.PREFERENCE: 1.0,
            MemoryType.HABIT: 0.7,
            MemoryType.CONSTRAINT: 1.2,
            MemoryType.BACKGROUND: 0.8,
            MemoryType.FACT: 0.9,
        }

    def policy_for_task(self, task_type: Optional[str]) -> InjectionPolicy:
        """解析任务类型对应的静态评分权重和类型先验。"""

        resolved = task_type or "general"
        if resolved not in TASK_SCORING_WEIGHTS:
            resolved = "general"
        return InjectionPolicy(
            task_type=resolved,
            weights=TASK_SCORING_WEIGHTS[resolved],
            type_weights=TASK_TYPE_TYPE_PRIORS.get(
                resolved, TASK_TYPE_TYPE_PRIORS["general"]
            ),
        )

    def classify_task_type(self, query: str) -> str:
        """根据关键词粗分天文问答、观测、学习等任务类型。"""

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
        """计算单条长期记忆与 query/task_type 的相关性分数和原因。"""

        hit = self.score_hit(item, query, task_type)
        return hit.score, hit.reasons

    def score_hit(
        self,
        item: MemoryItem,
        query: str,
        task_type: str,
        semantic_similarity: Optional[float] = None,
        type_weights: Optional[Dict[str, float]] = None,
    ) -> RetrievalHit:
        """返回带组件明细的归一化加法分数。"""

        policy = self.policy_for_task(task_type)
        if type_weights is not None:
            policy = InjectionPolicy(
                task_type=policy.task_type,
                weights=policy.weights,
                type_weights=type_weights,
            )
        semantic_value = self._clamp(float(semantic_similarity or 0.0))
        components = {
            "confidence": self._clamp(float(item.confidence or 0.0)),
            "type_weight": self._type_weight(item, policy),
            "source_bonus": self._source_bonus(item),
            "query_relevance": self._query_relevance(item, query),
            "semantic_similarity": semantic_value,
            "recency": self._recency(item),
            "constraint_bonus": 1.0
            if item.memory_type == MemoryType.CONSTRAINT
            else 0.0,
            "stale_penalty": self._stale_penalty(item),
        }
        weights = dict(policy.weights)
        if semantic_value > 0:
            semantic_weight = min(weights.get("query_relevance", 0.0), 0.08)
            weights["query_relevance"] = max(
                weights.get("query_relevance", 0.0) - semantic_weight,
                0.0,
            )
            weights["semantic_similarity"] = semantic_weight
        score = 0.0
        for name, weight in weights.items():
            if name == "stale_penalty":
                score -= weight * components[name]
            else:
                score += weight * components[name]
        score = self._clamp(score)
        reasons = self._score_reasons(policy.task_type, components)
        return RetrievalHit(
            item=item,
            score=round(score, 3),
            reasons=reasons,
            components={
                **{key: round(value, 3) for key, value in components.items()},
                "policy_score": round(score, 3),
                "rerank_score": 0.0,
            },
        )

    def retrieve(
        self,
        user_id: str,
        query: str,
        task_type: Optional[str] = None,
        limit: Optional[int] = None,
        include_below_threshold: bool = False,
    ) -> List[RetrievalHit]:
        """查询 active memories，过滤低分结果并按相关性返回。"""

        resolved_task_type = task_type or self.classify_task_type(query)
        items = self._repo.list_active_memories(user_id=user_id, limit=1000)
        if not items:
            return []

        adaptive_type_weights = self._adaptive_type_weights(
            items, resolved_task_type
        )
        rule_hits = [
            self.score_hit(
                item,
                query,
                resolved_task_type,
                type_weights=adaptive_type_weights,
            )
            for item in items
        ]
        rule_hits.sort(key=lambda hit: hit.score, reverse=True)
        union_ids = {hit.item.id for hit in rule_hits[:100]}

        semantic_scores, semantic_reason = self._semantic_scores(
            user_id=user_id,
            query=query,
            items=items,
        )
        for memory_id, _score in sorted(
            semantic_scores.items(), key=lambda entry: entry[1], reverse=True
        )[:100]:
            union_ids.add(memory_id)

        item_by_id = {item.id: item for item in items}
        hits: List[RetrievalHit] = []
        for memory_id in union_ids:
            item = item_by_id.get(memory_id)
            if not item:
                continue
            hit = self.score_hit(
                item,
                query,
                resolved_task_type,
                semantic_similarity=semantic_scores.get(item.id, 0.0),
                type_weights=adaptive_type_weights,
            )
            if semantic_scores.get(item.id, 0.0) > 0:
                hit.reasons.append("语义召回命中")
            elif semantic_reason:
                hit.reasons.append(f"语义召回降级: {semantic_reason}")
            if hit.score < self.relevance_threshold:
                hit.omitted_reason = "below_threshold"
            if include_below_threshold or hit.score >= self.relevance_threshold:
                hits.append(hit)
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: limit or self.max_memories]

    def _score_reasons(
        self, task_type: str, components: Dict[str, float]
    ) -> List[str]:
        """根据组件分生成面向 explain trace 的命中原因。"""

        reasons = [f"任务类型={task_type}"]
        if components["type_weight"] >= 0.75:
            reasons.append("类型适配当前任务")
        if components["source_bonus"] >= 0.85:
            reasons.append("来源可靠")
        if components["query_relevance"] >= 0.30:
            reasons.append("与 query 相关")
        if components.get("semantic_similarity", 0.0) >= 0.35:
            reasons.append("语义相似")
        if components["constraint_bonus"] > 0:
            reasons.append("约束类记忆优先注入")
        if components["stale_penalty"] > 0:
            reasons.append("连续未引用降权")
        return reasons

    def _type_weight(self, item: MemoryItem, policy: InjectionPolicy) -> float:
        """读取当前任务下该记忆类型的先验权重。"""

        return self._clamp(policy.type_weights.get(item.memory_type, 0.5))

    def _source_bonus(self, item: MemoryItem) -> float:
        """按来源可信度和人工确认状态计算来源加分。"""

        if item.confirmed_by_user or item.source_type == "confirmed":
            return 1.0
        if item.source_type == "manual":
            return 0.95
        if item.source_type == "explicit":
            return 0.90
        return 0.55

    def _query_relevance(self, item: MemoryItem, query: str) -> float:
        """基于关键词重叠、key/value 直包含计算规则相关性。"""

        query_terms = self._terms(query)
        memory_text = " ".join(
            [
                str(item.key or ""),
                str(item.category or ""),
                self._value_text(item.value),
            ]
        )
        memory_terms = self._terms(memory_text)
        if not query_terms or not memory_terms:
            return 0.0

        overlap = query_terms & memory_terms
        score = 0.0
        if overlap:
            score = 0.20 + 0.70 * min(
                len(overlap) / max(min(len(query_terms), len(memory_terms)), 1),
                1.0,
            )

        query_lower = str(query or "").lower()
        key_lower = str(item.key or "").lower()
        value_lower = self._value_text(item.value).lower()
        if key_lower and key_lower in query_lower:
            score = max(score, 0.55)
        if value_lower and len(value_lower) <= 80 and value_lower in query_lower:
            score = max(score, 0.70)
        return self._clamp(score)

    def _recency(self, item: MemoryItem) -> float:
        """按访问或更新时间做指数衰减的新近度评分。"""

        stamp = item.accessed_at or item.updated_at or item.created_at
        parsed = self._parse_dt(stamp)
        if not parsed:
            return 0.5
        age_days = max((datetime.now() - parsed).total_seconds() / 86400.0, 0.0)
        return self._clamp(math.exp(-age_days / 90.0))

    def _stale_penalty(self, item: MemoryItem) -> float:
        """根据连续注入未引用次数计算 stale 降权。"""

        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        stats = metadata.get("injection_stats") or {}
        try:
            consecutive_miss_count = int(stats.get("consecutive_miss_count", 0) or 0)
        except (TypeError, ValueError):
            consecutive_miss_count = 0
        return self._clamp(consecutive_miss_count / 5.0)

    def _semantic_scores(
        self,
        user_id: str,
        query: str,
        items: List[MemoryItem],
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """用已有 memory embedding 缓存计算 query 与候选的余弦相似度。"""

        if not self._embedding_service:
            return {}, "semantic_service_unavailable"
        cached, stale = self._embedding_service.cached_embeddings_for_items(
            user_id, items
        )
        if stale:
            self._embedding_service.schedule_embeddings(stale, limit=20)
        if not cached:
            return {}, "semantic_cache_miss"
        query_vector, fallback_reason = self._embedding_service.embed_query(query)
        if not query_vector:
            return {}, fallback_reason or "query_embedding_unavailable"
        scores: Dict[str, float] = {}
        for memory_id, vector in cached.items():
            similarity = self._cosine_similarity(query_vector, vector)
            if similarity > 0:
                scores[memory_id] = self._clamp(similarity)
        return scores, None

    def _adaptive_type_weights(
        self, items: List[MemoryItem], task_type: str
    ) -> Dict[str, float]:
        """按注入反馈统计生成 task_type x memory_type 自适应先验。"""

        static = dict(
            TASK_TYPE_TYPE_PRIORS.get(task_type, TASK_TYPE_TYPE_PRIORS["general"])
        )
        aggregates: Dict[str, Dict[str, int]] = {}
        for item in items:
            stats = (item.metadata or {}).get("injection_stats") or {}
            task_stats = None
            by_task_type = stats.get("by_task_type") or {}
            if isinstance(by_task_type, dict):
                task_stats = by_task_type.get(task_type)
            if not task_stats and stats.get("last_task_type") == task_type:
                task_stats = stats
            if not isinstance(task_stats, dict):
                continue
            try:
                shown_count = int(task_stats.get("shown_count", 0) or 0)
                hit_count = int(task_stats.get("hit_count", 0) or 0)
            except (TypeError, ValueError):
                continue
            bucket = aggregates.setdefault(
                item.memory_type, {"shown_count": 0, "hit_count": 0}
            )
            bucket["shown_count"] += shown_count
            bucket["hit_count"] += hit_count

        adjusted = dict(static)
        for memory_type, values in aggregates.items():
            shown = values["shown_count"]
            if shown < 20:
                continue
            empirical_hit_rate = self._clamp(values["hit_count"] / max(shown, 1))
            adjusted[memory_type] = self._clamp(
                0.7 * static.get(memory_type, 0.5) + 0.3 * empirical_hit_rate
            )
        return adjusted

    def _cosine_similarity(self, left: List[float], right: List[float]) -> float:
        """计算两个等长向量的非负余弦相似度。"""

        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return max(0.0, dot / (left_norm * right_norm))

    def _terms(self, text: str) -> Set[str]:
        """抽取中英文关键词，并为连续中文词补充短 n-gram。"""

        lower = str(text or "").lower()
        terms = {
            token
            for token in re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]+", lower)
            if len(token) >= 2
        }
        for keyword in ASTRONOMY_KEYWORDS | OBSERVATION_KEYWORDS | LEARNING_KEYWORDS:
            if keyword.lower() in lower:
                terms.add(keyword.lower())
        for token in list(terms):
            if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
                for size in (2, 3, 4):
                    for index in range(0, max(len(token) - size + 1, 0)):
                        terms.add(token[index : index + size])
        return terms

    def _value_text(self, value: Any) -> str:
        """把列表、字典或标量 value 转为规则匹配文本。"""

        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        if isinstance(value, dict):
            return " ".join(f"{key} {val}" for key, val in value.items())
        return "" if value is None else str(value)

    def _parse_dt(self, value: Any) -> Optional[datetime]:
        """安全解析 ISO 时间字符串，失败时返回 None。"""

        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    def _clamp(self, value: float) -> float:
        """把评分限制在 0 到 1 区间。"""

        return max(0.0, min(float(value), 1.0))
