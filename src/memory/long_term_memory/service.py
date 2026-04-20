from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger
from src.memory.long_term_memory.candidate import CandidateManager
from src.memory.long_term_memory.deletion import LongTermMemoryDeletionService
from src.memory.long_term_memory.event_log import ConfirmationManager, EventLogger
from src.memory.long_term_memory.extractor import MemoryExtractor
from src.memory.long_term_memory.models import (
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
    _utcnow_iso,
)
from src.memory.long_term_memory.profile_projection import ProfileProjection
from src.memory.long_term_memory.prompt_injector import PromptInjector
from src.memory.long_term_memory.quality import QualityAssurance
from src.memory.long_term_memory.repository import LongTermMemoryRepository


class LongTermMemoryService:
    """Unified facade for the long-term memory flow.

    Main flow: extract -> candidate -> promote/confirm -> memory item ->
    profile projection -> query-aware prompt injection.
    """

    def __init__(self, db_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.config = {
            "db_path": db_path or settings.LONG_TERM_MEMORY_PATH,
            "default_confidence": 0.5,
            "min_confidence_to_store": 0.3,
            "dedup_similarity_threshold": 0.85,
            "candidate_occurrence_threshold": 2,
            "candidate_confidence_threshold": 0.6,
            "candidate_explicit_bypass": True,
            "max_prompt_tokens": 800,
            "max_memories_in_prompt": 15,
        }
        if config:
            self.config.update(config)

        self.repository = LongTermMemoryRepository(self.config["db_path"])
        self.repository.initialize()
        self.quality = QualityAssurance(
            dedup_similarity_threshold=self.config["dedup_similarity_threshold"],
            min_confidence_to_store=self.config["min_confidence_to_store"],
        )
        self.extractor = MemoryExtractor()
        self.event_logger = EventLogger(self.repository)
        self.confirmations = ConfirmationManager(self.repository, self.event_logger)
        self.candidates = CandidateManager(
            self.repository,
            occurrence_threshold=self.config["candidate_occurrence_threshold"],
            confidence_threshold=self.config["candidate_confidence_threshold"],
            explicit_bypass=self.config["candidate_explicit_bypass"],
        )
        self.projection = ProfileProjection(self.repository)
        self.prompt_injector = PromptInjector(
            self.repository,
            max_prompt_tokens=self.config["max_prompt_tokens"],
            max_memories=self.config["max_memories_in_prompt"],
        )
        self.deletion = LongTermMemoryDeletionService(self.repository, self.projection)

    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        category: str,
        key: str,
        value: Any,
        confidence: Optional[float] = None,
        source_type: str = SourceType.AUTO,
        is_explicit: bool = False,
        source_conversation_id: Optional[str] = None,
        source_content_snippet: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryItem]:
        resolved_confidence = confidence
        if resolved_confidence is None:
            resolved_confidence = self.quality.confidence_scorer.initial_confidence(source_type, is_explicit)

        if not self.quality.should_store(resolved_confidence, is_explicit):
            logger.debug("长期记忆置信度过低，跳过: %s.%s", memory_type, key)
            return None

        existing = self.repository.find_memory_by_type_key(user_id, memory_type, key)
        if existing:
            return self._update_existing_memory(
                existing,
                value,
                resolved_confidence,
                source_type,
                source_conversation_id,
                source_content_snippet,
                metadata,
            )

        item = MemoryItem.create(
            user_id=user_id,
            memory_type=memory_type,
            category=category,
            key=key,
            value=value,
            confidence=resolved_confidence,
            source_type=SourceType.EXPLICIT if is_explicit else source_type,
            source_conversation_id=source_conversation_id,
            source_content_snippet=source_content_snippet,
            priority=priority,
            metadata=metadata or {},
            expires_at=self.quality.compute_expiry(memory_type, source_type),
        )
        self.repository.add_memory(item)
        self._save_version(item, "initial_create")
        self.event_logger.log_created(user_id, item.id, memory_type, key, value)
        self.projection.rebuild(user_id)
        return item

    def _update_existing_memory(
        self,
        existing: MemoryItem,
        value: Any,
        confidence: float,
        source_type: str,
        source_conversation_id: Optional[str],
        source_content_snippet: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> MemoryItem:
        if existing.value != value:
            self._save_version(existing, "value_update")
            old_value = existing.value
            existing.value = value
            existing.confidence = max(existing.confidence, confidence)
            self.event_logger.log_updated(existing.user_id, existing.id, existing.key, old_value, value, "value_update")
        else:
            existing.confidence = max(existing.confidence, confidence)
            existing.access_count += 1
        if source_type == SourceType.EXPLICIT:
            existing.source_type = SourceType.EXPLICIT
        if source_conversation_id:
            existing.source_conversation_id = source_conversation_id
        if source_content_snippet:
            existing.source_content_snippet = source_content_snippet
        if metadata:
            existing.metadata.update(metadata)
        existing.accessed_at = _utcnow_iso()
        existing.updated_at = existing.accessed_at
        self.repository.update_memory(existing)
        self.projection.rebuild(existing.user_id)
        return existing

    def _save_version(self, item: MemoryItem, reason: str):
        versions = self.repository.get_versions(item.id, limit=1)
        next_version = (versions[0].version + 1) if versions else 1
        self.repository.add_version(item.id, next_version, item.value, item.confidence, reason)

    def extract_and_store(
        self,
        user_message: str,
        assistant_message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        if not self.extractor.should_attempt_extraction(user_message):
            return []
        extracted = self.extractor.extract_from_conversation(user_message, assistant_message, conversation_id)
        return self.store_extractions(user_id, extracted, conversation_id=conversation_id)

    def store_extractions(
        self,
        user_id: str,
        extracted_items: List[ExtractionResult],
        conversation_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        promoted: List[MemoryItem] = []
        for item in extracted_items:
            if not item.should_extract or item.is_temporary:
                continue
            stored = self.candidates.process_extraction_as_candidate(
                user_id=user_id,
                memory_type=item.memory_type,
                category=item.category,
                key=item.key,
                value=item.value,
                confidence=item.confidence,
                source_type=SourceType.EXPLICIT if item.is_explicit else item.source_type,
                is_explicit=item.is_explicit,
                source_conversation_id=conversation_id,
                source_content_snippet=item.raw_content,
            )
            if stored:
                promoted.append(stored)
                self._save_version(stored, "candidate_promoted")
        if promoted:
            self.projection.rebuild(user_id)
        return promoted

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        item = self.repository.get_memory(memory_id)
        if item:
            self.repository.increment_access_count(memory_id)
            self.event_logger.log_accessed(item.user_id, memory_id, item.key)
        return item

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        value: Optional[Any] = None,
        confidence: Optional[float] = None,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        confirmed_by_user: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryItem]:
        item = self.repository.get_memory(memory_id)
        if not item or item.user_id != user_id:
            return None
        if value is not None and value != item.value:
            self._save_version(item, "manual_update")
            old_value = item.value
            item.value = value
            self.event_logger.log_updated(user_id, memory_id, item.key, old_value, value, "manual_update")
        if confidence is not None:
            item.confidence = confidence
        if status is not None:
            item.status = status
        if priority is not None:
            item.priority = priority
        if confirmed_by_user is not None:
            item.confirmed_by_user = confirmed_by_user
            if confirmed_by_user:
                item.confirmation_count += 1
        if metadata:
            item.metadata.update(metadata)
        item.updated_at = _utcnow_iso()
        self.repository.update_memory(item)
        self.projection.rebuild(user_id)
        return item

    def delete_memory(self, memory_id: str, user_id: str, reason: str = "") -> bool:
        result = self.delete(LongTermMemoryDeletionRequest(
            user_id=user_id, scope="memory", target_id=memory_id, reason=reason
        ))
        return result.deleted_memories > 0

    def delete_profile(self, user_id: str, reason: str = "") -> bool:
        result = self.delete(LongTermMemoryDeletionRequest(user_id=user_id, scope="profile", reason=reason))
        return result.deleted_profiles > 0

    def delete(self, request: LongTermMemoryDeletionRequest) -> LongTermMemoryDeletionResult:
        return self.deletion.delete(request)

    def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        return self.repository.query_memories(query)

    def count_memories(self, query: MemoryQuery) -> int:
        return self.repository.count_memories(query)

    def promote_candidate(self, candidate_id: str) -> Optional[MemoryItem]:
        item = self.candidates.promote_candidate(candidate_id)
        if item:
            self._save_version(item, "candidate_promoted")
            self.projection.rebuild(item.user_id)
        return item

    def promote_all_eligible(self, user_id: str) -> List[MemoryItem]:
        items = self.candidates.promote_all_eligible(user_id)
        for item in items:
            self._save_version(item, "candidate_promoted")
        if items:
            self.projection.rebuild(user_id)
        return items

    def rebuild_profile(self, user_id: str) -> Dict[str, Any]:
        return self.projection.rebuild(user_id)

    def load_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.projection.load_or_rebuild(user_id)

    def format_profile_for_prompt(self, user_id: str, task_type: Optional[str] = None) -> str:
        return self.prompt_injector.format_profile_for_prompt(user_id, task_type=task_type)

    def format_smart_prompt(self, user_id: str, query: str) -> str:
        return self.prompt_injector.format_for_prompt(user_id, query)

    def explain_memory_hits(self, user_id: str, query: str, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        resolved_task_type = task_type or self.prompt_injector.classify_task_type(query)
        hits = []
        for item in self.prompt_injector.select_memories(user_id, query, resolved_task_type):
            score, reasons = self.prompt_injector._retriever.score(item, query, resolved_task_type)
            hits.append(
                {
                    "memory_id": item.id,
                    "memory_type": item.memory_type,
                    "category": item.category,
                    "key": item.key,
                    "value": item.value,
                    "confidence": item.confidence,
                    "relevance": round(score, 3),
                    "reason": "；".join(reasons),
                    "source_type": item.source_type,
                    "confirmed_by_user": item.confirmed_by_user,
                    "timestamp": item.updated_at or item.created_at,
                }
            )
        return hits

    def record_memory_event(
        self,
        user_id: str,
        event_type: str,
        key: str,
        value: Any,
        source_text: str = "",
        confidence: Optional[float] = None,
        status: str = "candidate",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        event = MemoryEvent(
            user_id=user_id,
            event_type=event_type,
            key=key,
            value=value,
            source_text=source_text,
            confidence=confidence if confidence is not None else self.config["default_confidence"],
            status=status,
            metadata=metadata or {},
            last_confirmed_at=_utcnow_iso() if status == "active" else None,
        )
        stored = self.repository.add_event(event)
        self.repository.add_event_log(
            EventLogEntry(
                user_id=user_id,
                memory_id=None,
                event_type=EventType.CREATED,
                event_detail=f"兼容 memory_event 已记录: {event_type}.{key}",
                new_value=str(value),
                metadata={"legacy_status": status, "source_text": source_text, **(metadata or {})},
            )
        )
        return stored

    def get_versions(self, memory_id: str, limit: int = 20) -> List[MemoryVersion]:
        return self.repository.get_versions(memory_id, limit=limit)

    def list_candidates(self, user_id: str, limit: int = 50, offset: int = 0) -> List[MemoryCandidate]:
        return self.candidates.list_candidates(user_id, limit=limit, offset=offset)

    def reject_candidate(self, candidate_id: str, reason: str = "") -> bool:
        return self.candidates.reject_candidate(candidate_id, reason)

    def list_pending_confirmations(self, user_id: str, limit: int = 20) -> List[MemoryConfirmation]:
        return self.confirmations.list_pending_confirmations(user_id, limit=limit)

    def resolve_confirmation(self, confirmation_id: str, status: str) -> Optional[MemoryConfirmation]:
        return self.confirmations.resolve_confirmation(confirmation_id, status)
