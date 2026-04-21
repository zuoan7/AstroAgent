from dataclasses import dataclass
from typing import Optional, Set

from src.core.config import settings as core_settings


def get_memory_settings():
    try:
        from src.memory.config import settings as memory_settings_module

        return getattr(memory_settings_module, "settings", core_settings)
    except Exception:
        return core_settings


def _safe_value(source: object, name: str, default):
    value = getattr(source, name, default)
    try:
        from unittest.mock import MagicMock

        if isinstance(value, MagicMock):
            return default
    except Exception:
        pass
    return value


@dataclass
class ShortTermMemoryConfig:
    max_size: int
    memory_window: int
    max_recent_messages: int
    max_recent_tokens: int
    summary_trigger_messages: int
    summary_trigger_tokens: int
    summary_keep_last_n: int
    summary_max_tokens: int
    context_budget: int
    context_max_tokens: int
    enable_summary: bool
    enable_persistence: bool
    persistence_path: str
    high_importance_roles: Set[str]
    tool_result_max_length: int
    max_tool_records: int
    max_salient_facts: int

    @classmethod
    def from_settings(cls, source: Optional[object] = None) -> "ShortTermMemoryConfig":
        s = source or get_memory_settings()
        context_max_tokens = _safe_value(s, "STM_CONTEXT_MAX_TOKENS", 4000)
        context_budget = _safe_value(s, "STM_CONTEXT_BUDGET", context_max_tokens)
        max_recent_messages = _safe_value(s, "STM_MAX_RECENT_MESSAGES", 6)
        max_tool_records = _safe_value(s, "STM_MAX_TOOL_RECORDS", 5)
        max_salient_facts = _safe_value(s, "STM_MAX_SALIENT_FACTS", 32)
        return cls(
            max_size=_safe_value(s, "MEMORY_SIZE", 15),
            memory_window=_safe_value(s, "MEMORY_WINDOW", 8),
            max_recent_messages=max_recent_messages,
            max_recent_tokens=_safe_value(s, "STM_MAX_RECENT_TOKENS", max(512, context_budget // 2)),
            summary_trigger_messages=_safe_value(s, "STM_SUMMARY_TRIGGER_MESSAGES", 10),
            summary_trigger_tokens=_safe_value(s, "STM_SUMMARY_TRIGGER_TOKENS", 3000),
            summary_keep_last_n=_safe_value(s, "STM_SUMMARY_KEEP_LAST_N", 3),
            summary_max_tokens=_safe_value(s, "STM_SUMMARY_MAX_TOKENS", 500),
            context_budget=context_budget,
            context_max_tokens=context_max_tokens,
            enable_summary=_safe_value(s, "STM_ENABLE_SUMMARY", True),
            enable_persistence=_safe_value(s, "STM_PERSISTENCE_ENABLED", True),
            persistence_path=_safe_value(s, "STM_PERSISTENCE_PATH", "./memory/short_term_memory/sessions.sqlite"),
            high_importance_roles=set(_safe_value(s, "STM_IMPORTANCE_HIGH_ROLES", {"user", "system"})),
            tool_result_max_length=_safe_value(s, "STM_TOOL_RESULT_MAX_LENGTH", 500),
            max_tool_records=max_tool_records,
            max_salient_facts=max_salient_facts,
        )
