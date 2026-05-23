"""长期记忆 prompt 注入器。

按 query 和任务类型选择相关长期记忆，在 token 预算内渲染为模型可读的
用户偏好、约束、背景和事实区块。
"""

from typing import Any, Dict, List, Optional, Set

from src.core.logger import logger
from src.memory.long_term_memory.models import (
    MemoryEvent,
    MemoryItem,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
    _utcnow_iso,
)
from src.memory.long_term_memory.profile_projection import ProfileProjection
from src.memory.long_term_memory.repository import LongTermMemoryRepository
from src.memory.long_term_memory.retrieval import (
    ASTRONOMY_KEYWORDS,
    LEARNING_KEYWORDS,
    OBSERVATION_KEYWORDS,
    TASK_TYPE_MEMORY_MAP,
    LongTermMemoryRetriever,
)


class PromptInjector:
    """选择并格式化可注入 prompt 的长期记忆。"""

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
        self._projection = ProfileProjection(repository)
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
        self._retriever = LongTermMemoryRetriever(
            repository,
            relevance_threshold=relevance_threshold,
            max_memories=max_memories,
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)

    def classify_task_type(self, query: str) -> str:
        """复用长期记忆检索器的任务类型分类。"""

        return self._retriever.classify_task_type(query)

    def compute_relevance(self, item: MemoryItem, query: str, task_type: str) -> float:
        """计算单条记忆对 query 的相关性分数。"""

        score, _ = self._retriever.score(item, query, task_type)
        return score

    def select_memories(
        self, user_id: str, query: str, task_type: Optional[str] = None
    ) -> List[MemoryItem]:
        """在数量和 token 预算内选择最相关的长期记忆。"""

        if not task_type:
            task_type = self.classify_task_type(query)

        hits = self._retriever.retrieve(user_id, query, task_type, limit=100)
        if not hits:
            return []

        selected = []
        total_tokens = 0
        for hit in hits:
            item = hit.item
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
        """把单条长期记忆格式化为一行 prompt 文本。"""

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
        """按类型分组渲染 query-aware 长期记忆上下文。"""

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

    def _format_profile(self, profile: Dict[str, Any]) -> str:
        """把完整用户画像投影渲染为 prompt 文本。"""

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
        return "\n\n".join(parts)

    def _select_events_for_prompt(self, user_id: str, task_type: Optional[str] = None) -> List[MemoryEvent]:
        """选择兼容旧事件模型的高置信记忆事件。"""

        events = self._repo.get_active_events(user_id, limit=self.max_memories)
        return sorted(
            events,
            key=lambda event: (event.confidence, event.last_confirmed_at or event.created_at, event.created_at),
            reverse=True,
        )[: self.max_memories]

    def _format_events(self, events: List[MemoryEvent]) -> str:
        """把旧版 memory_events 渲染为 prompt 附加区块。"""

        if not events:
            return ""
        lines = [f"- {event.event_type}.{event.key}: {event.value}" for event in events]
        return "【近期记忆事件】\n" + "\n".join(lines)

    def format_profile_for_prompt(self, user_id: str, task_type: Optional[str] = None) -> str:
        """渲染完整画像和旧版 active events，主要用于兼容路径。"""

        profile = self._projection.build(user_id)
        if not any(
            profile.get(key)
            for key in ["preferences", "habits", "constraints", "background", "facts"]
        ):
            profile = self._repo.load_profile(user_id)
            if not profile:
                return "暂无用户偏好信息"
        parts = []
        formatted_profile = self._format_profile(profile)
        if formatted_profile:
            parts.append(formatted_profile)
        formatted_events = self._format_events(self._select_events_for_prompt(user_id, task_type=task_type))
        if formatted_events:
            parts.append(formatted_events)
        return "\n\n".join(parts) if parts else "暂无用户偏好信息"
