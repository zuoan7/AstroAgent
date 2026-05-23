"""长期记忆用户画像投影。

正式 active memories 是事实来源；user_profiles 表是为了 prompt 注入和 API
读取而构建的扁平化投影。
"""

from typing import Any, Dict, List, Optional

from src.memory.long_term_memory.models import (
    EventLogEntry,
    EventType,
    MemoryItem,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
)
from src.memory.long_term_memory.repository import LongTermMemoryRepository


class ProfileProjection:
    """Rebuilds user_profile from active long-term memory items.

    The profile table is a prompt-friendly projection. The source of truth remains
    the active rows in memories plus lifecycle tables and event logs.
    """

    def __init__(self, repository: LongTermMemoryRepository):
        self._repo = repository

    def build(self, user_id: str, limit: int = 1000) -> Dict[str, Any]:
        """查询 active memories 并构建用户画像字典。"""

        memories = self._repo.query_memories(
            MemoryQuery(user_id=user_id, status=MemoryStatus.ACTIVE, limit=limit)
        )
        return self.build_from_items(user_id, memories)

    def build_from_items(self, user_id: str, memories: List[MemoryItem]) -> Dict[str, Any]:
        """按记忆类型把正式记忆聚合为 preferences/habits/constraints 等字段。"""

        preferences: Dict[str, Any] = {}
        habits: Dict[str, Any] = {}
        constraints: List[str] = []
        background: Dict[str, Any] = {}
        facts: List[Dict[str, Any]] = []

        for item in sorted(memories, key=lambda m: (m.priority, m.confidence, m.updated_at), reverse=True):
            if item.memory_type == MemoryType.PREFERENCE:
                preferences[item.key] = item.value
            elif item.memory_type == MemoryType.HABIT:
                if item.key == "frequent_topics" and isinstance(item.value, list):
                    existing = habits.get("frequent_topics", [])
                    habits["frequent_topics"] = list(dict.fromkeys(existing + item.value))
                else:
                    habits[item.key] = item.value
            elif item.memory_type == MemoryType.CONSTRAINT:
                value = item.value if isinstance(item.value, str) else item.key
                if value and value not in constraints:
                    constraints.append(value)
            elif item.memory_type == MemoryType.BACKGROUND:
                background[item.key] = item.value
            elif item.memory_type == MemoryType.FACT:
                facts.append(
                    {
                        "id": item.id,
                        "key": item.key,
                        "value": item.value,
                        "category": item.category,
                        "confidence": item.confidence,
                    }
                )

        return {
            "user_id": user_id,
            "preferences": preferences,
            "habits": habits,
            "constraints": constraints,
            "background": background,
            "facts": facts,
        }

    def rebuild(self, user_id: str) -> Dict[str, Any]:
        """重建并持久化用户画像投影，同时写入同步事件日志。"""

        profile = self.build(user_id)
        self._repo.save_profile(
            user_id,
            profile["preferences"],
            profile["habits"],
            profile["constraints"],
            profile["background"],
            profile["facts"],
        )
        self._repo.add_event_log(EventLogEntry(
            user_id=user_id,
            memory_id=None,
            event_type=EventType.PROFILE_SYNCED,
            event_detail="用户画像投影已由 active memories 重建",
            metadata={"projection_source": "memories"},
        ))
        return self._repo.load_profile(user_id) or profile

    def load_or_rebuild(self, user_id: str) -> Optional[Dict[str, Any]]:
        """优先读取画像投影；缺失但有 active memories 时自动重建。"""

        profile = self._repo.load_profile(user_id)
        if profile:
            return profile
        if self._repo.count_memories(MemoryQuery(user_id=user_id, status=MemoryStatus.ACTIVE)) == 0:
            return None
        return self.rebuild(user_id)
