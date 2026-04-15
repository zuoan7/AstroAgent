from typing import Any, Dict, List, Optional, Set

from src.core.logger import logger
from src.memory.long_term_memory.models import (
    MemoryItem,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
    _utcnow_iso,
)
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
    "视星等", "赤经", "赤纬", "方位角", "高度角",
}

OBSERVATION_KEYWORDS: Set[str] = {
    "今晚", "观测", "天气", "可见", "升起", "落下",
    "最佳", "推荐", "目标", "望远镜", "拍摄",
}

LEARNING_KEYWORDS: Set[str] = {
    "什么是", "为什么", "怎么", "如何", "原理",
    "解释", "科普", "入门", "学习", "了解",
    "是什么", "怎么回事", "讲讲",
}


class PromptInjector:
    def __init__(
        self,
        repository: LongTermMemoryRepository,
        max_prompt_tokens: int = 800,
        max_memories: int = 15,
        relevance_threshold: float = 0.3,
        preference_weight: float = 1.0,
        habit_weight: float = 0.7,
        constraint_weight: float = 1.2,
        background_weight: float = 0.8,
        fact_weight: float = 0.9,
    ):
        self._repo = repository
        self.max_prompt_tokens = max_prompt_tokens
        self.max_memories = max_memories
        self.relevance_threshold = relevance_threshold
        self.type_weights = {
            MemoryType.PREFERENCE: preference_weight,
            MemoryType.HABIT: habit_weight,
            MemoryType.CONSTRAINT: constraint_weight,
            MemoryType.BACKGROUND: background_weight,
            MemoryType.FACT: fact_weight,
        }

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)

    def classify_task_type(self, query: str) -> str:
        query_lower = query.lower()
        obs_score = sum(1 for kw in OBSERVATION_KEYWORDS if kw in query_lower)
        learn_score = sum(1 for kw in LEARNING_KEYWORDS if kw in query_lower)
        astro_score = sum(1 for kw in ASTRONOMY_KEYWORDS if kw in query_lower)

        if obs_score >= 2:
            return "observation"
        if learn_score >= 1:
            return "learning"
        if astro_score >= 2:
            return "qa"
        return "general"

    def compute_relevance(self, item: MemoryItem, query: str, task_type: str) -> float:
        relevance = item.confidence * self.type_weights.get(item.memory_type, 1.0)

        if item.source_type == "explicit":
            relevance *= 1.3
        if item.confirmed_by_user:
            relevance *= 1.2

        relevant_types = TASK_TYPE_MEMORY_MAP.get(task_type, [])
        if item.memory_type in relevant_types:
            relevance *= 1.1
        else:
            relevance *= 0.6

        if item.memory_type == MemoryType.CONSTRAINT:
            relevance *= 1.5

        query_lower = query.lower()
        value_str = str(item.value).lower() if item.value else ""
        key_str = str(item.key).lower()
        if any(kw in value_str for kw in query_lower.split() if len(kw) > 1):
            relevance *= 1.2
        if any(kw in key_str for kw in query_lower.split() if len(kw) > 1):
            relevance *= 1.1

        return min(relevance, 2.0)

    def select_memories(
        self, user_id: str, query: str, task_type: Optional[str] = None
    ) -> List[MemoryItem]:
        if not task_type:
            task_type = self.classify_task_type(query)

        all_memories = self._repo.query_memories(MemoryQuery(
            user_id=user_id,
            status=MemoryStatus.ACTIVE,
            limit=100,
            min_confidence=self.relevance_threshold,
        ))

        if not all_memories:
            return []

        scored = []
        for item in all_memories:
            relevance = self.compute_relevance(item, query, task_type)
            if relevance >= self.relevance_threshold:
                scored.append((item, relevance))

        scored.sort(key=lambda x: x[1], reverse=True)

        selected = []
        total_tokens = 0
        for item, score in scored:
            item_text = self._format_single_memory(item)
            item_tokens = self._estimate_tokens(item_text)
            if total_tokens + item_tokens > self.max_prompt_tokens:
                continue
            if len(selected) >= self.max_memories:
                break
            selected.append(item)
            total_tokens += item_tokens

        for item in selected:
            self._repo.increment_access_count(item.id)

        return selected

    def _format_single_memory(self, item: MemoryItem) -> str:
        type_labels = {
            MemoryType.PREFERENCE: "偏好",
            MemoryType.HABIT: "习惯",
            MemoryType.CONSTRAINT: "约束",
            MemoryType.BACKGROUND: "背景",
            MemoryType.FACT: "事实",
        }
        label = type_labels.get(item.memory_type, item.memory_type)
        value_str = item.value if isinstance(item.value, str) else str(item.value)
        if isinstance(item.value, list):
            value_str = ", ".join(str(v) for v in item.value)
        return f"- [{label}] {item.key}: {value_str}"

    def format_for_prompt(self, user_id: str, query: str, task_type: Optional[str] = None) -> str:
        memories = self.select_memories(user_id, query, task_type)
        if not memories:
            return "暂无用户偏好信息"

        grouped: Dict[str, List[MemoryItem]] = {}
        for item in memories:
            grouped.setdefault(item.memory_type, []).append(item)

        type_labels = {
            MemoryType.PREFERENCE: "用户偏好",
            MemoryType.HABIT: "用户习惯",
            MemoryType.CONSTRAINT: "约束条件",
            MemoryType.BACKGROUND: "用户背景",
            MemoryType.FACT: "稳定事实",
        }
        type_order = [MemoryType.CONSTRAINT, MemoryType.PREFERENCE, MemoryType.BACKGROUND, MemoryType.FACT, MemoryType.HABIT]

        parts = []
        for mem_type in type_order:
            items = grouped.get(mem_type)
            if not items:
                continue
            label = type_labels.get(mem_type, mem_type)
            lines = [self._format_single_memory(item) for item in items]
            parts.append(f"【{label}】\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "暂无用户偏好信息"

    def format_profile_for_prompt(self, user_id: str) -> str:
        profile = self._repo.load_profile(user_id)
        if not profile:
            return "暂无用户偏好信息"

        parts = []
        if profile.get("preferences"):
            lines = [f"- {k}: {v}" for k, v in profile["preferences"].items()]
            parts.append("【用户偏好】\n" + "\n".join(lines))
        if profile.get("habits"):
            lines = []
            for k, v in profile["habits"].items():
                if isinstance(v, list):
                    lines.append(f"- {k}: {', '.join(str(i) for i in v[:12])}")
                else:
                    lines.append(f"- {k}: {v}")
            parts.append("【用户习惯】\n" + "\n".join(lines))
        if profile.get("constraints"):
            lines = [f"- {c}" for c in profile["constraints"]]
            parts.append("【约束条件】\n" + "\n".join(lines))
        if profile.get("background"):
            lines = [f"- {k}: {v}" for k, v in profile["background"].items()]
            parts.append("【用户背景】\n" + "\n".join(lines))
        if profile.get("facts"):
            lines = [f"- {f.get('key', '')}: {f.get('value', '')}" for f in profile["facts"]]
            parts.append("【稳定事实】\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "暂无用户偏好信息"
