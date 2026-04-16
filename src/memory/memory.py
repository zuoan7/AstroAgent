from typing import Any, Dict, Optional

from src.core.config import settings
from src.memory.core.models import Message, SalientFact, ToolCallRecord
from src.memory.long_term_memory import LongTermMemoryManager
from src.memory.long_term_memory.models import MemoryEvent, UserProfile
from src.memory.short_term_memory.manager import ShortTermMemory


class LongTermMemory:
    """兼容旧接口的长期记忆包装器，内部复用新的 LongTermMemoryManager。"""

    def __init__(self, db_path: Optional[str] = None):
        self._manager = LongTermMemoryManager(db_path=db_path)
        self.config = self._manager.config

    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        data = self._manager.load_profile(user_id)
        return UserProfile(**data) if data else None

    def save_profile(self, profile: UserProfile):
        self._manager._repo.save_profile(
            profile.user_id,
            profile.preferences,
            profile.habits,
            profile.constraints,
            profile.background,
            profile.facts,
        )

    def record_memory_event(
        self,
        user_id: str,
        event_type: str,
        content: str,
        source: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ):
        key = (metadata or {}).get("field") or content
        value = (metadata or {}).get("value", content)
        source_text = source if not content else f"{source}: {content}"
        return self._manager.record_memory_event(
            user_id=user_id,
            event_type=event_type,
            key=str(key),
            value=value,
            source_text=source_text,
            confidence=confidence,
            status="active",
            metadata={**(metadata or {}), "legacy_content": content},
        )

    def get_recent_memory_events(self, user_id: str, limit: int = 10):
        return [event.to_dict() for event in self._manager._repo.get_recent_events(user_id, limit=limit)]

    def extract_from_conversation(self, user_message: str, assistant_message: str) -> Dict[str, Any]:
        return self._manager.extract_from_conversation(user_message, assistant_message)

    def merge_and_update(self, user_id: str, new_info: Dict[str, Any]) -> UserProfile:
        data = self._manager.merge_and_update(user_id, new_info)
        return UserProfile(**data)

    def export_profile_snapshot(self, user_id: str) -> Dict[str, Any]:
        snapshot = self._manager.export_profile_snapshot(user_id)
        profile = snapshot.get("profile")
        if profile:
            snapshot["profile"] = UserProfile(**profile).to_dict()
        return snapshot

    def format_profile_for_prompt(self, user_id: str) -> str:
        return self._manager.format_profile_for_prompt(user_id)

    def delete_profile(self, user_id: str) -> bool:
        return self._manager.delete_profile(user_id)


__all__ = [
    "LongTermMemory",
    "LongTermMemoryManager",
    "MemoryEvent",
    "Message",
    "SalientFact",
    "ShortTermMemory",
    "ToolCallRecord",
    "UserProfile",
    "settings",
]
