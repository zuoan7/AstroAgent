"""记忆系统配置适配层。

本文件把全局 settings 中的短期/长期记忆参数收敛为稳定的数据结构，
供服务初始化和测试覆盖使用，避免业务代码直接散落读取环境配置。
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

from src.core.config import settings


@dataclass(frozen=True)
class ShortTermMemoryConfig:
    default_user_id: str
    memory_size: int
    memory_window: int
    context_max_tokens: int
    summary_max_tokens: int
    summary_trigger_messages: int
    summary_trigger_tokens: int
    persistence_enabled: bool
    persistence_path: str
    tool_result_max_length: int
    task_state_enabled: bool
    task_state_llm_enrich_enabled: bool
    task_state_extract_model_name: str
    task_state_extract_timeout_seconds: float


@dataclass(frozen=True)
class LongTermMemoryConfig:
    db_path: str
    default_confidence: float
    min_confidence_to_store: float
    dedup_similarity_threshold: float
    candidate_occurrence_threshold: int
    candidate_confidence_threshold: float
    candidate_explicit_bypass: bool
    max_prompt_tokens: int
    max_memories_in_prompt: int
    auto_backup_interval_hours: int
    async_extract_workers: int
    semantic_retrieval_enabled: bool
    injection_rerank_enabled: bool
    embed_timeout_seconds: float
    rerank_timeout_seconds: float
    embed_backfill_limit: int


def get_short_term_memory_config() -> ShortTermMemoryConfig:
    """读取短期记忆配置，包含上下文预算、摘要阈值和任务状态开关。"""

    return ShortTermMemoryConfig(
        default_user_id=settings.DEFAULT_USER_ID,
        memory_size=settings.MEMORY_SIZE,
        memory_window=settings.MEMORY_WINDOW,
        context_max_tokens=settings.MEMORY_CONTEXT_MAX_TOKENS,
        summary_max_tokens=settings.MEMORY_SUMMARY_MAX_TOKENS,
        summary_trigger_messages=settings.MEMORY_SUMMARY_TRIGGER_MESSAGES,
        summary_trigger_tokens=settings.MEMORY_SUMMARY_TRIGGER_TOKENS,
        persistence_enabled=settings.MEMORY_PERSISTENCE_ENABLED,
        persistence_path=settings.MEMORY_PERSISTENCE_PATH,
        tool_result_max_length=settings.MEMORY_TOOL_RESULT_MAX_LENGTH,
        task_state_enabled=settings.MEMORY_TASK_STATE_ENABLED,
        task_state_llm_enrich_enabled=settings.MEMORY_TASK_STATE_LLM_ENRICH_ENABLED,
        task_state_extract_model_name=settings.MEMORY_TASK_STATE_EXTRACT_MODEL_NAME,
        task_state_extract_timeout_seconds=settings.MEMORY_TASK_STATE_EXTRACT_TIMEOUT_SECONDS,
    )


def get_long_term_memory_config(
    db_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> LongTermMemoryConfig:
    """读取长期记忆配置，并允许测试或调用方覆盖部分字段。"""

    config = LongTermMemoryConfig(
        db_path=db_path or settings.LONG_TERM_MEMORY_PATH,
        default_confidence=0.5,
        min_confidence_to_store=settings.LTM_MIN_CONFIDENCE_TO_STORE,
        dedup_similarity_threshold=settings.LTM_DEDUP_SIMILARITY_THRESHOLD,
        candidate_occurrence_threshold=settings.LTM_CANDIDATE_OCCURRENCE_THRESHOLD,
        candidate_confidence_threshold=settings.LTM_CANDIDATE_CONFIDENCE_THRESHOLD,
        candidate_explicit_bypass=True,
        max_prompt_tokens=settings.LTM_MAX_PROMPT_TOKENS,
        max_memories_in_prompt=settings.LTM_MAX_MEMORIES_IN_PROMPT,
        auto_backup_interval_hours=settings.LTM_AUTO_BACKUP_INTERVAL_HOURS,
        async_extract_workers=1,
        semantic_retrieval_enabled=settings.LTM_SEMANTIC_RETRIEVAL_ENABLED,
        injection_rerank_enabled=settings.LTM_INJECTION_RERANK_ENABLED,
        embed_timeout_seconds=settings.LTM_EMBED_TIMEOUT_SECONDS,
        rerank_timeout_seconds=settings.LTM_RERANK_TIMEOUT_SECONDS,
        embed_backfill_limit=settings.LTM_EMBED_BACKFILL_LIMIT,
    )
    if not overrides:
        return config
    data = asdict(config)
    known_fields = {field.name for field in fields(LongTermMemoryConfig)}
    data.update({key: value for key, value in overrides.items() if key in known_fields})
    return LongTermMemoryConfig(**data)


__all__ = [
    "LongTermMemoryConfig",
    "ShortTermMemoryConfig",
    "get_long_term_memory_config",
    "get_short_term_memory_config",
    "settings",
]
