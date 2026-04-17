import json
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger
from src.memory.long_term_memory.backup import BackupManager
from src.memory.long_term_memory.candidate import CandidateManager
from src.memory.long_term_memory.event_log import ConfirmationManager, EventLogger
from src.memory.long_term_memory.extractor import MemoryExtractor
from src.memory.long_term_memory.models import (
    CandidateMemory,
    ConflictInfo,
    ConflictResolution,
    EventLogEntry,
    EventType,
    ExtractionResult,
    MemoryEvent,
    MemoryCandidate,
    MemoryConfirmation,
    MemoryItem,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    SourceType,
    UserProfile,
    _utcnow_iso,
)
from src.memory.long_term_memory.prompt_injector import PromptInjector
from src.memory.long_term_memory.quality import QualityAssurance
from src.memory.long_term_memory.repository import LongTermMemoryRepository


class LongTermMemoryManager:
    def __init__(self, db_path: Optional[str] = None):
        self.config = {
            "db_path": db_path or settings.LONG_TERM_MEMORY_PATH,
            "recent_events_limit": 10,
            "candidate_promote_threshold": 2,
            "default_confidence": 0.5,
            "high_confidence_threshold": 0.8,
            "enable_candidate_memory": True,
            "min_confidence_to_store": 0.3,
            "dedup_similarity_threshold": 0.85,
            "candidate_occurrence_threshold": 2,
            "candidate_confidence_threshold": 0.6,
            "max_prompt_tokens": 800,
            "max_memories_in_prompt": 15,
            "auto_backup_interval_hours": 24,
        }

        self._repo = LongTermMemoryRepository(self.config["db_path"])
        self._repo.initialize()

        self._quality = QualityAssurance(
            dedup_similarity_threshold=self.config["dedup_similarity_threshold"],
            min_confidence_to_store=self.config["min_confidence_to_store"],
        )
        self._extractor = MemoryExtractor()
        self._event_logger = EventLogger(self._repo)
        self._confirmation_mgr = ConfirmationManager(self._repo, self._event_logger)
        self._candidate_mgr = CandidateManager(
            self._repo,
            occurrence_threshold=self.config["candidate_occurrence_threshold"],
            confidence_threshold=self.config["candidate_confidence_threshold"],
        )
        self._prompt_injector = PromptInjector(
            self._repo,
            max_prompt_tokens=self.config["max_prompt_tokens"],
            max_memories=self.config["max_memories_in_prompt"],
        )
        self._backup_mgr = BackupManager(
            self._repo,
            auto_backup_interval_hours=self.config["auto_backup_interval_hours"],
        )

    def _is_explicit_long_term_signal(self, text: str) -> bool:
        return self._extractor.is_explicit_expression(text)

    def _is_temporary_request(self, text: str) -> bool:
        return self._extractor.is_temporary_request(text)

    def _should_attempt_extraction(self, user_message: str) -> bool:
        return self._extractor.should_attempt_extraction(user_message)

    def _normalize_event_type(self, memory_type: str) -> str:
        return str(memory_type)

    def _create_memory_events(
        self,
        user_id: str,
        extracted_items: List[ExtractionResult],
        source_text: str,
    ) -> List[MemoryEvent]:
        events: List[MemoryEvent] = []
        for item in extracted_items:
            if not item.should_extract or item.is_temporary:
                continue
            status = "active" if self._should_promote_event(item) else "candidate"
            events.append(
                MemoryEvent(
                    user_id=user_id,
                    event_type=self._normalize_event_type(item.memory_type),
                    key=item.key,
                    value=item.value,
                    source_text=source_text[:500],
                    confidence=item.confidence or self.config["default_confidence"],
                    status=status,
                    metadata={
                        "category": item.category,
                        "source_type": item.source_type,
                        "is_explicit": item.is_explicit,
                        "raw_content": item.raw_content[:200],
                    },
                    last_confirmed_at=_utcnow_iso() if status == "active" else None,
                )
            )
        return events

    def _store_candidate_events(self, events: List[MemoryEvent]) -> List[MemoryEvent]:
        return self._repo.add_events(events)

    def _should_promote_event(self, event: ExtractionResult | MemoryEvent) -> bool:
        metadata = getattr(event, "metadata", {}) or {}
        is_explicit = getattr(event, "is_explicit", metadata.get("is_explicit", False))
        confidence = getattr(event, "confidence", self.config["default_confidence"])
        if is_explicit:
            return True
        if confidence >= self.config["high_confidence_threshold"]:
            return True
        user_id = getattr(event, "user_id", None)
        key = getattr(event, "key", "")
        value = getattr(event, "value", None)
        if user_id and key:
            similar_count = self._repo.count_similar_events(user_id, key, value)
            if similar_count >= self.config["candidate_promote_threshold"]:
                return True
        return False

    def _promote_candidate_event(self, event: MemoryEvent) -> bool:
        updated = self._repo.confirm_event(event.event_id)
        if updated:
            event.status = "active"
            event.last_confirmed_at = _utcnow_iso()
        return updated

    def _promote_events_if_needed(self, user_id: str) -> List[MemoryEvent]:
        promoted: List[MemoryEvent] = []
        for event in self._repo.get_candidate_events(user_id):
            if self._should_promote_event(event) and self._promote_candidate_event(event):
                promoted.append(event)
        return promoted

    def _load_profile_model(self, user_id: str) -> Optional[UserProfile]:
        data = self._repo.load_profile(user_id)
        if not data:
            return None
        return UserProfile(**data)

    def _merge_preferences(self, existing: Dict[str, Any], events: List[MemoryEvent]) -> Dict[str, Any]:
        merged = dict(existing)
        existing_conf = {}
        for event in events:
            if event.event_type != MemoryType.PREFERENCE:
                continue
            old_conf = existing_conf.get(event.key, 0.0)
            if event.confidence >= old_conf:
                merged[event.key] = event.value
                existing_conf[event.key] = event.confidence
        return merged

    def _merge_habits(self, existing: Dict[str, Any], events: List[MemoryEvent]) -> Dict[str, Any]:
        merged = dict(existing)
        for event in events:
            if event.event_type != MemoryType.HABIT:
                continue
            current = merged.get(event.key)
            if isinstance(event.value, list):
                current_list = current if isinstance(current, list) else []
                merged[event.key] = list(dict.fromkeys(current_list + event.value))
            elif isinstance(current, list):
                merged[event.key] = list(dict.fromkeys(current + [event.value]))
            else:
                merged[event.key] = event.value
        return merged

    def _merge_constraints(self, existing: List[str], events: List[MemoryEvent]) -> List[str]:
        merged = list(existing)
        for event in events:
            if event.event_type != MemoryType.CONSTRAINT:
                continue
            value = event.value if isinstance(event.value, str) else event.key
            if value not in merged:
                merged.append(value)
        return merged

    def _merge_profile(self, existing: Optional[UserProfile], user_id: str, events: List[MemoryEvent]) -> UserProfile:
        base = existing or UserProfile(user_id=user_id)
        return UserProfile(
            user_id=user_id,
            preferences=self._merge_preferences(base.preferences, events),
            habits=self._merge_habits(base.habits, events),
            constraints=self._merge_constraints(base.constraints, events),
            background=base.background,
            facts=base.facts,
            created_at=base.created_at,
            updated_at=_utcnow_iso(),
        )

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
        if confidence is None:
            confidence = self._quality.confidence_scorer.initial_confidence(source_type, is_explicit)

        if not self._quality.should_store(confidence, is_explicit):
            logger.debug(f"记忆置信度过低，跳过存储: {memory_type}.{key} (conf={confidence:.2f})")
            return None

        existing = self._repo.find_memory_by_type_key(user_id, memory_type, key)
        if existing:
            return self._handle_existing_memory(
                existing, value, confidence, source_type, is_explicit,
                source_conversation_id, source_content_snippet, metadata,
            )

        similar = self._repo.find_similar_memories(
            user_id, memory_type, category, str(value)[:50]
        )
        if similar:
            duplicates = self._quality.check_duplicates(value, similar)
            if duplicates:
                return self._handle_duplicate(
                    duplicates[0][0], value, confidence, source_type,
                    source_conversation_id, source_content_snippet, metadata,
                )

        expires_at = self._quality.compute_expiry(memory_type, source_type)
        item = MemoryItem.create(
            user_id=user_id,
            memory_type=memory_type,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            source_type=source_type,
            source_conversation_id=source_conversation_id,
            source_content_snippet=source_content_snippet,
            priority=priority,
            metadata=metadata or {},
            expires_at=expires_at,
        )
        if is_explicit:
            item.source_type = SourceType.EXPLICIT

        self._repo.add_memory(item)
        self._save_version(item, "initial_create")
        self._event_logger.log_created(user_id, item.id, memory_type, key, value)
        self._sync_profile_from_memories(user_id)
        return item

    def _handle_existing_memory(
        self,
        existing: MemoryItem,
        new_value: Any,
        new_confidence: float,
        source_type: str,
        is_explicit: bool,
        source_conversation_id: Optional[str],
        source_content_snippet: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[MemoryItem]:
        conflicts = self._quality.detect_conflicts(
            existing.memory_type, existing.key, new_value, new_confidence, [existing]
        )

        if conflicts:
            conflict = conflicts[0]
            self._event_logger.log_conflict(
                existing.user_id, existing.id,
                conflict.conflict_type, conflict.resolution,
                existing.value, new_value,
            )

            resolution = self._quality.resolve_conflict(conflict)
            if resolution == ConflictResolution.NEEDS_CONFIRM:
                self._confirmation_mgr.create_confirmation(
                    user_id=existing.user_id,
                    memory_id=existing.id,
                    confirmation_type="update",
                    content=f"检测到冲突: 旧值={existing.value}, 新值={new_value}",
                )
                return existing
            elif resolution == ConflictResolution.OVERWRITE:
                return self._update_memory_value(existing, new_value, new_confidence, source_type, "conflict_overwrite")
            elif resolution == ConflictResolution.UPDATE:
                merged_value, merged_conf = self._quality.merge_duplicate(existing, new_value, new_confidence)
                return self._update_memory_value(existing, merged_value, merged_conf, source_type, "conflict_update")
            else:
                return existing

        if existing.value == new_value:
            existing.confidence = max(existing.confidence, new_confidence)
            existing.access_count += 1
            existing.accessed_at = _utcnow_iso()
            if source_conversation_id:
                existing.source_conversation_id = source_conversation_id
            self._repo.update_memory(existing)
            return existing

        return self._update_memory_value(existing, new_value, new_confidence, source_type, "value_update")

    def _handle_duplicate(
        self,
        existing: MemoryItem,
        new_value: Any,
        new_confidence: float,
        source_type: str,
        source_conversation_id: Optional[str],
        source_content_snippet: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[MemoryItem]:
        merged_value, merged_conf = self._quality.merge_duplicate(existing, new_value, new_confidence)
        self._event_logger.log_deduplicated(
            existing.user_id, existing.id, existing.key,
            f"合并相似记忆 (相似度>={self.config['dedup_similarity_threshold']})",
        )
        return self._update_memory_value(existing, merged_value, merged_conf, source_type, "dedup_merge")

    def _update_memory_value(
        self, item: MemoryItem, new_value: Any, new_confidence: float, source_type: str, reason: str
    ) -> MemoryItem:
        old_value = item.value
        old_confidence = item.confidence

        self._save_version(item, reason)

        item.value = new_value
        item.confidence = new_confidence
        item.updated_at = _utcnow_iso()
        if source_type == SourceType.EXPLICIT:
            item.source_type = SourceType.EXPLICIT
        item.access_count += 1
        item.accessed_at = _utcnow_iso()

        self._repo.update_memory(item)
        self._event_logger.log_updated(item.user_id, item.id, item.key, old_value, new_value, reason)
        return item

    def _save_version(self, item: MemoryItem, reason: str):
        versions = self._repo.get_versions(item.id, limit=1)
        next_version = (versions[0].version + 1) if versions else 1
        self._repo.add_version(item.id, next_version, item.value, item.confidence, reason)

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        item = self._repo.get_memory(memory_id)
        if item:
            self._repo.increment_access_count(memory_id)
            self._event_logger.log_accessed(item.user_id, memory_id, item.key)
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
        item = self._repo.get_memory(memory_id)
        if not item or item.user_id != user_id:
            return None

        if value is not None and value != item.value:
            self._save_version(item, "manual_update")
            old_value = item.value
            item.value = value
            self._event_logger.log_updated(user_id, memory_id, item.key, old_value, value, "manual_update")

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
        if metadata is not None:
            item.metadata.update(metadata)

        item.updated_at = _utcnow_iso()
        self._repo.update_memory(item)
        return item

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        item = self._repo.get_memory(memory_id)
        if not item or item.user_id != user_id:
            return False
        self._event_logger.log_deleted(user_id, memory_id, item.key, item.value)
        return self._repo.delete_memory(memory_id, user_id)

    def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        return self._repo.query_memories(query)

    def count_memories(self, query: MemoryQuery) -> int:
        return self._repo.count_memories(query)

    def get_memory_versions(self, memory_id: str, limit: int = 20) -> List[MemoryVersion]:
        return self._repo.get_versions(memory_id, limit=limit)

    def restore_memory_version(self, memory_id: str, version: int, user_id: str) -> Optional[MemoryItem]:
        item = self._repo.get_memory(memory_id)
        if not item or item.user_id != user_id:
            return None

        versions = self._repo.get_versions(memory_id, limit=50)
        target = None
        for v in versions:
            if v.version == version:
                target = v
                break
        if not target:
            return None

        self._save_version(item, f"restore_to_v{version}")
        old_value = item.value
        item.value = target.value
        item.confidence = target.confidence
        item.updated_at = _utcnow_iso()
        self._repo.update_memory(item)
        self._event_logger.log_updated(user_id, memory_id, item.key, old_value, target.value, f"restore_to_v{version}")
        return item

    def extract_and_store(
        self,
        user_message: str,
        assistant_message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        if not self._should_attempt_extraction(user_message):
            return []
        extractions = self._extractor.extract_from_conversation(
            user_message, assistant_message, conversation_id
        )
        events = self._create_memory_events(user_id, extractions, user_message)
        self._store_candidate_events(events)
        promoted = self._promote_events_if_needed(user_id)
        active_events = [event for event in events if event.status == "active"] + promoted
        if active_events:
            profile = self._merge_profile(self._load_profile_model(user_id), user_id, active_events)
            self._repo.save_profile(
                user_id,
                profile.preferences,
                profile.habits,
                profile.constraints,
                profile.background,
                profile.facts,
            )

        stored: List[MemoryItem] = []
        for ext in extractions:
            if not ext.should_extract or ext.is_temporary:
                continue
            item = self.add_memory(
                user_id=user_id,
                memory_type=ext.memory_type,
                category=ext.category,
                key=ext.key,
                value=ext.value,
                confidence=ext.confidence,
                source_type=SourceType.EXPLICIT if ext.is_explicit else SourceType.AUTO,
                is_explicit=ext.is_explicit,
                source_conversation_id=conversation_id,
                source_content_snippet=ext.raw_content,
            )
            if item:
                stored.append(item)

        if events:
            logger.info(f"提取并存储 {len(events)} 条长期记忆事件 (user_id: {user_id})")
        return stored

    def format_profile_for_prompt(self, user_id: str, task_type: Optional[str] = None) -> str:
        return self._prompt_injector.format_profile_for_prompt(user_id, task_type=task_type)

    def format_smart_prompt(self, user_id: str, query: str) -> str:
        return self._prompt_injector.format_for_prompt(user_id, query)

    def explain_memory_hits(
        self, user_id: str, query: str, task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not task_type:
            task_type = self._prompt_injector.classify_task_type(query)

        hits: List[Dict[str, Any]] = []
        for item in self._prompt_injector.select_memories(user_id, query, task_type):
            relevance = self._prompt_injector.compute_relevance(item, query, task_type)
            reasons = [f"任务类型={task_type}"]
            if item.source_type == SourceType.EXPLICIT:
                reasons.append("来自用户显式表达")
            if item.confirmed_by_user:
                reasons.append("已获用户确认")
            if item.memory_type == MemoryType.CONSTRAINT:
                reasons.append("属于约束条件，优先级更高")
            query_text = query.lower()
            key_text = str(item.key).lower()
            value_text = str(item.value).lower() if item.value is not None else ""
            if any(token in value_text for token in query_text.split() if len(token) > 1):
                reasons.append("与当前问题内容直接相关")
            elif any(token in key_text for token in query_text.split() if len(token) > 1):
                reasons.append("与当前问题关键词匹配")

            hits.append(
                {
                    "memory_id": item.id,
                    "memory_type": item.memory_type,
                    "category": item.category,
                    "key": item.key,
                    "value": item.value,
                    "confidence": item.confidence,
                    "relevance": round(relevance, 3),
                    "reason": "；".join(reasons),
                    "source_type": item.source_type,
                    "confirmed_by_user": item.confirmed_by_user,
                    "timestamp": item.updated_at or item.created_at,
                }
            )
        return hits

    def load_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._repo.load_profile(user_id)

    def delete_profile(self, user_id: str) -> bool:
        return self._repo.delete_profile(user_id)

    def _sync_profile_from_memories(self, user_id: str):
        memories = self._repo.query_memories(MemoryQuery(
            user_id=user_id, status=MemoryStatus.ACTIVE, limit=500
        ))

        preferences: Dict[str, Any] = {}
        habits: Dict[str, Any] = {}
        constraints: List[str] = []
        background: Dict[str, Any] = {}
        facts: List[Dict[str, Any]] = []

        for item in memories:
            if item.memory_type == MemoryType.PREFERENCE:
                preferences[item.key] = item.value
            elif item.memory_type == MemoryType.HABIT:
                if item.key == "frequent_topics" and isinstance(item.value, list):
                    existing = habits.get("frequent_topics", [])
                    merged = list(dict.fromkeys(existing + item.value))
                    habits["frequent_topics"] = merged
                else:
                    habits[item.key] = item.value
            elif item.memory_type == MemoryType.CONSTRAINT:
                if isinstance(item.value, str):
                    constraints.append(item.value)
                elif isinstance(item.value, bool) and item.value:
                    constraints.append(item.key)
            elif item.memory_type == MemoryType.BACKGROUND:
                background[item.key] = item.value
            elif item.memory_type == MemoryType.FACT:
                facts.append({"key": item.key, "value": item.value, "category": item.category, "id": item.id})

        self._repo.save_profile(user_id, preferences, habits, constraints, background, facts)

    def get_event_logs(
        self, user_id: str, memory_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[EventLogEntry]:
        return self._event_logger.get_event_logs(user_id, memory_id, limit=limit, offset=offset)

    def get_memory_trace(self, user_id: str, memory_id: str) -> List[EventLogEntry]:
        return self._event_logger.get_memory_trace(user_id, memory_id)

    def list_candidates(self, user_id: str, limit: int = 50, offset: int = 0) -> List[MemoryCandidate]:
        return self._candidate_mgr.list_candidates(user_id, limit=limit, offset=offset)

    def promote_candidate(self, candidate_id: str) -> Optional[MemoryItem]:
        return self._candidate_mgr.promote_candidate(candidate_id)

    def reject_candidate(self, candidate_id: str, reason: str = "") -> bool:
        return self._candidate_mgr.reject_candidate(candidate_id, reason)

    def promote_all_eligible(self, user_id: str) -> List[MemoryItem]:
        return self._candidate_mgr.promote_all_eligible(user_id)

    def list_pending_confirmations(self, user_id: str, limit: int = 20) -> List[MemoryConfirmation]:
        return self._confirmation_mgr.list_pending_confirmations(user_id, limit=limit)

    def resolve_confirmation(self, confirmation_id: str, status: str) -> Optional[MemoryConfirmation]:
        return self._confirmation_mgr.resolve_confirmation(confirmation_id, status)

    def batch_confirm(self, user_id: str, confirmation_ids: List[str], status: str) -> List[MemoryConfirmation]:
        return self._confirmation_mgr.batch_confirm(user_id, confirmation_ids, status)

    def create_backup(self, tag: Optional[str] = None) -> Optional[str]:
        return self._backup_mgr.create_backup(tag)

    def restore_from_backup(self, backup_path: str) -> bool:
        return self._backup_mgr.restore_from_backup(backup_path)

    def list_backups(self) -> List[dict]:
        return self._backup_mgr.list_backups()

    def run_maintenance(self, user_id: str) -> Dict[str, int]:
        result = {"expired": 0, "archived": 0, "promoted": 0}

        expired = self._repo.expire_old_memories(user_id, _utcnow_iso())
        result["expired"] = expired

        archived = self._repo.archive_unused_memories(user_id)
        result["archived"] = archived

        promoted = self._candidate_mgr.promote_all_eligible(user_id)
        result["promoted"] = len(promoted)

        self._sync_profile_from_memories(user_id)
        self._backup_mgr.maybe_auto_backup()

        if any(result.values()):
            logger.info(f"维护完成 (user_id: {user_id}): {result}")
        return result

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        return self._repo.get_memory_stats(user_id)

    def export_profile_snapshot(self, user_id: str) -> Dict[str, Any]:
        profile = self._repo.load_profile(user_id)
        if not profile:
            return {}
        memories = self._repo.query_memories(MemoryQuery(
            user_id=user_id, status=MemoryStatus.ACTIVE, limit=100
        ))
        return {
            "user_id": user_id,
            "profile": profile,
            "active_memory_count": len(memories),
            "stats": self._repo.get_memory_stats(user_id),
        }

    def merge_and_update(self, user_id: str, new_info: Dict[str, Any]) -> Dict[str, Any]:
        extraction_like: List[ExtractionResult] = []
        for key, value in (new_info.get("preferences") or {}).items():
            extraction_like.append(ExtractionResult(True, MemoryType.PREFERENCE, key, key, value, 0.9, SourceType.EXPLICIT, True, False, str(value)))
        for key, value in (new_info.get("habits") or {}).items():
            extraction_like.append(ExtractionResult(True, MemoryType.HABIT, key, key, value, 0.8, SourceType.AUTO, False, False, str(value)))
        for constraint in (new_info.get("constraints") or []):
            extraction_like.append(ExtractionResult(True, MemoryType.CONSTRAINT, "custom", f"constraint_{abs(hash(constraint)) % 10000}", constraint, 0.9, SourceType.EXPLICIT, True, False, str(constraint)))
        for key, value in (new_info.get("background") or {}).items():
            extraction_like.append(ExtractionResult(True, MemoryType.BACKGROUND, key, key, value, 0.7, SourceType.AUTO, False, False, str(value)))
        for fact in (new_info.get("facts") or []):
            if isinstance(fact, dict):
                extraction_like.append(ExtractionResult(True, MemoryType.FACT, fact.get("category", "basic_info"), fact.get("key", ""), fact.get("value"), 0.7, SourceType.AUTO, False, False, str(fact.get("value"))))

        events = self._create_memory_events(user_id, extraction_like, json.dumps(new_info, ensure_ascii=False))
        self._store_candidate_events(events)
        promoted = self._promote_events_if_needed(user_id)
        active_events = [event for event in events if event.status == "active"] + promoted
        profile = self._merge_profile(self._load_profile_model(user_id), user_id, active_events)
        self._repo.save_profile(
            user_id,
            profile.preferences,
            profile.habits,
            profile.constraints,
            profile.background,
            profile.facts,
        )

        for ext in extraction_like:
            self.add_memory(
                user_id=user_id,
                memory_type=ext.memory_type,
                category=ext.category,
                key=ext.key,
                value=ext.value,
                confidence=ext.confidence,
                source_type=ext.source_type,
                is_explicit=ext.is_explicit,
                source_content_snippet=ext.raw_content,
            )

        return self._repo.load_profile(user_id) or {}

    def extract_from_conversation(self, user_message: str, assistant_message: str) -> Dict[str, Any]:
        return self._extractor.extract_legacy_format(user_message, assistant_message)

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
    ) -> MemoryEvent:
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
        return self._repo.add_event(event)

    def confirm_memory_event(self, event_id: str) -> bool:
        return self._repo.confirm_event(event_id)

    def reject_memory_event(self, event_id: str) -> bool:
        return self._repo.update_event_status(event_id, "stale")

    def mark_event_stale(self, event_id: str) -> bool:
        return self._repo.update_event_status(event_id, "stale")

    def debug_user_memory(self, user_id: str) -> Dict[str, Any]:
        return {
            "profile": self._repo.load_profile(user_id),
            "active_events": [event.to_dict() for event in self._repo.get_active_events(user_id, limit=50)],
            "candidate_events": [event.to_dict() for event in self._repo.get_candidate_events(user_id)],
        }

    def debug_events(self, user_id: str) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self._repo.get_recent_events(user_id, limit=100)]

    def debug_candidate_events(self, user_id: str) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self._repo.get_candidate_events(user_id)]
