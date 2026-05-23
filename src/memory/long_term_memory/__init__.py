"""长期记忆模块导出。

包含正式记忆、候选、确认、删除、事件日志和 LongTermMemoryService facade。
"""

from src.memory.long_term_memory.models import (
    CandidateMemory,
    ConfirmationStatus,
    ConflictInfo,
    ConflictResolution,
    EventLogEntry,
    EventType,
    ExtractionResult,
    LongTermMemoryDeletionRequest,
    LongTermMemoryDeletionResult,
    MemoryCandidate,
    MemoryConfirmation,
    MemoryEvent,
    MemoryItem,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    SourceType,
    UserProfile,
)
from src.memory.long_term_memory.service import LongTermMemoryService

__all__ = [
    "CandidateMemory",
    "ConfirmationStatus",
    "ConflictInfo",
    "ConflictResolution",
    "EventLogEntry",
    "EventType",
    "ExtractionResult",
    "LongTermMemoryService",
    "LongTermMemoryDeletionRequest",
    "LongTermMemoryDeletionResult",
    "MemoryEvent",
    "MemoryCandidate",
    "MemoryConfirmation",
    "MemoryItem",
    "MemoryQuery",
    "MemoryStatus",
    "MemoryType",
    "MemoryVersion",
    "SourceType",
    "UserProfile",
]
