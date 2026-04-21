"""
短期记忆配置模块

该模块定义了短期记忆的完整配置体系，包括：
- ShortTermMemoryConfig：短期记忆核心配置类
- get_memory_settings()：安全的配置获取函数
- 配置项的默认值和回退机制

主要功能：
- 统一管理短期记忆的各类参数
- 提供配置项的安全访问和默认值处理
- 支持动态配置更新
"""

from dataclasses import dataclass
from typing import Optional, Set

from src.core.config import settings as core_settings


def _safe_value(source: object, name: str, default):
    """
    安全获取配置值

    从配置对象中获取指定名称的属性值，如果属性不存在或为MagicMock类型，
    则返回默认值。主要用于测试环境和配置缺失时的安全回退。

    参数:
        source: 配置对象
        name: 配置项名称
        default: 默认值

    返回:
        配置项的值或默认值
    """
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
    """
    短期记忆配置类

    定义短期记忆模块的所有配置参数，包括消息存储限制、
    上下文预算、摘要策略、持久化设置等。

    配置项说明：
        max_size: 消息队列最大长度，超过时触发trim操作
        memory_window: 默认的最近消息窗口大小
        max_recent_messages: 保留的最近消息数量
        max_recent_tokens: 最近消息的token预算限制
        summary_trigger_messages: 触发摘要的消息数量阈值
        summary_trigger_tokens: 触发摘要的token数量阈值
        summary_keep_last_n: 摘要时保留的最近消息数量
        summary_max_tokens: 摘要内容的最大token数
        context_budget: 上下文总token预算
        context_max_tokens: 上下文最大token限制
        enable_summary: 是否启用自动摘要功能
        enable_persistence: 是否启用持久化存储
        persistence_path: 持久化数据库文件路径
        high_importance_roles: 高重要性角色集合
        tool_result_max_length: 工具结果的最大长度限制
        max_tool_records: 保存的工具调用记录数量
        max_salient_facts: 保存的重要事实数量上限
    """

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
