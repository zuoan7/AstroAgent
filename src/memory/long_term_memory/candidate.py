from typing import Any, Dict, List, Optional

from src.core.logger import logger
from src.memory.long_term_memory.models import (
    EventLogEntry,
    EventType,
    MemoryCandidate,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    SourceType,
    _utcnow_iso,
)
from src.memory.long_term_memory.repository import LongTermMemoryRepository


class CandidateManager:
    PROMOTION_OCCURRENCE_THRESHOLD = 2
    PROMOTION_CONFIDENCE_THRESHOLD = 0.6
    PROMOTION_EXPLICIT_BYPASS = True

    def __init__(
        self,
        repository: LongTermMemoryRepository,
        occurrence_threshold: int = 2,
        confidence_threshold: float = 0.6,
        explicit_bypass: bool = True,
    ):
        self._repo = repository
        self.occurrence_threshold = occurrence_threshold
        self.confidence_threshold = confidence_threshold
        self.explicit_bypass = explicit_bypass

    def add_or_update_candidate(
        self,
        user_id: str,
        memory_type: str,
        category: str,
        key: str,
        value: Any,
        confidence: float = 0.3,
        source_type: str = SourceType.AUTO,
        source_conversation_id: Optional[str] = None,
        source_content_snippet: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryCandidate:
        existing = self._repo.find_candidate_by_type_key(user_id, memory_type, key)

        if existing:
            existing.occurrence_count += 1
            existing.last_seen_at = _utcnow_iso()
            existing.updated_at = existing.last_seen_at
            existing.confidence = min(existing.confidence + 0.05, 0.9)
            if source_type == SourceType.EXPLICIT:
                existing.source_type = SourceType.EXPLICIT
                existing.confidence = min(existing.confidence + 0.2, 0.95)
            if source_content_snippet:
                existing.source_content_snippet = source_content_snippet
            if metadata:
                existing.metadata.update(metadata)
            self._repo.update_candidate(existing)
            self._repo.add_event_log(EventLogEntry(
                user_id=user_id,
                memory_id=None,
                event_type=EventType.CANDIDATE_CREATED,
                event_detail=f"候选记忆更新: {memory_type}.{key}",
                new_value=str(value),
                metadata={"candidate_id": existing.id, "occurrence_count": existing.occurrence_count},
            ))
            logger.debug(f"候选记忆更新: {memory_type}.{key} (出现{existing.occurrence_count}次)")
            return existing

        candidate = MemoryCandidate(
            user_id=user_id,
            memory_type=memory_type,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            source_type=source_type,
            source_conversation_id=source_conversation_id,
            source_content_snippet=source_content_snippet,
            metadata=metadata or {},
        )
        self._repo.add_candidate(candidate)
        self._repo.add_event_log(EventLogEntry(
            user_id=user_id,
            memory_id=None,
            event_type=EventType.CANDIDATE_CREATED,
            event_detail=f"候选记忆创建: {memory_type}.{key}",
            new_value=str(value),
            metadata={"candidate_id": candidate.id, "confidence": candidate.confidence},
        ))
        logger.debug(f"新候选记忆: {memory_type}.{key}")
        return candidate

    def should_promote(self, candidate: MemoryCandidate) -> bool:
        if self.explicit_bypass and candidate.source_type == SourceType.EXPLICIT:
            return True
        if candidate.occurrence_count >= self.occurrence_threshold:
            return True
        if candidate.confidence >= self.confidence_threshold:
            return True
        return False

    def promote_candidate(self, candidate_id: str) -> Optional[MemoryItem]:
        candidate = self._repo.get_candidate(candidate_id)
        if not candidate:
            logger.warning(f"候选记忆不存在: {candidate_id}")
            return None

        if not self.should_promote(candidate):
            logger.debug(f"候选记忆未达提升标准: {candidate_id}")
            return None

        memory_item = candidate.to_memory_item()
        self._repo.add_memory(memory_item)
        self._repo.mark_candidate_promoted(candidate_id, memory_item.id)

        self._repo.add_event_log(EventLogEntry(
            user_id=candidate.user_id,
            memory_id=memory_item.id,
            event_type=EventType.CANDIDATE_PROMOTED,
            event_detail=f"候选记忆提升为正式记忆: {candidate.memory_type}.{candidate.key}",
            new_value=str(candidate.value),
            metadata={"candidate_id": candidate_id, "occurrence_count": candidate.occurrence_count},
        ))

        logger.info(f"候选记忆提升: {candidate.memory_type}.{candidate.key} (出现{candidate.occurrence_count}次)")
        return memory_item

    def reject_candidate(self, candidate_id: str, reason: str = "") -> bool:
        candidate = self._repo.get_candidate(candidate_id)
        if not candidate:
            return False

        self._repo.delete_candidate(candidate_id)
        self._repo.add_event_log(EventLogEntry(
            user_id=candidate.user_id,
            memory_id=None,
            event_type=EventType.CANDIDATE_REJECTED,
            event_detail=f"候选记忆被拒绝: {candidate.memory_type}.{candidate.key}",
            metadata={"candidate_id": candidate_id, "reason": reason},
        ))
        logger.info(f"候选记忆拒绝: {candidate_id}, 原因: {reason}")
        return True

    def process_extraction_as_candidate(
        self,
        user_id: str,
        memory_type: str,
        category: str,
        key: str,
        value: Any,
        confidence: float = 0.3,
        source_type: str = SourceType.AUTO,
        is_explicit: bool = False,
        source_conversation_id: Optional[str] = None,
        source_content_snippet: Optional[str] = None,
    ) -> Optional[MemoryItem]:
        candidate = self.add_or_update_candidate(
            user_id=user_id,
            memory_type=memory_type,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            source_type=source_type,
            source_conversation_id=source_conversation_id,
            source_content_snippet=source_content_snippet,
        )

        if self.should_promote(candidate):
            return self.promote_candidate(candidate.id)

        return None

    def list_candidates(self, user_id: str, limit: int = 50, offset: int = 0) -> List[MemoryCandidate]:
        return self._repo.list_candidates(user_id, limit=limit, offset=offset)

    def get_candidate(self, candidate_id: str) -> Optional[MemoryCandidate]:
        return self._repo.get_candidate(candidate_id)

    def promote_all_eligible(self, user_id: str) -> List[MemoryItem]:
        promoted = []
        candidates = self._repo.list_candidates(user_id, limit=1000)
        for candidate in candidates:
            if self.should_promote(candidate):
                item = self.promote_candidate(candidate.id)
                if item:
                    promoted.append(item)
        if promoted:
            logger.info(f"批量提升 {len(promoted)} 条候选记忆 (user_id: {user_id})")
        return promoted
