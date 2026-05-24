"""长期记忆服务 facade。

本文件编排长期记忆的完整生命周期：抽取用户画像信号、候选提升、冲突/撤回、
画像投影、prompt 注入、语义索引维护和注入反馈自学习。所有模型相关能力都
以可降级方式挂在主链路旁，避免影响回答生成。
"""

import hashlib
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Set

from src.core.logger import logger
from src.memory.config import get_long_term_memory_config
from src.memory.feedback import MemoryFeedbackRecord
from src.memory.long_term_memory.backup import BackupManager
from src.memory.long_term_memory.candidate import CandidateManager
from src.memory.long_term_memory.deletion import LongTermMemoryDeletionService
from src.memory.long_term_memory.embedding import MemoryEmbeddingService
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
from src.memory.selection_strategy_config import get_memory_selection_strategy_config
from src.memory.task_context import coerce_task_context_profile


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested strategy override dictionary."""

    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


class LongTermMemoryService:
    """Unified facade for the long-term memory flow.

    Main flow: extract -> candidate -> promote/confirm -> memory item ->
    profile projection -> query-aware prompt injection.
    """

    def __init__(
        self, db_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ):
        """初始化长期记忆仓储、质量控制、候选、投影和后台抽取线程池。"""

        self.config = get_long_term_memory_config(
            db_path=db_path,
            overrides=config,
        ).__dict__
        self.strategy_config = get_memory_selection_strategy_config(
            overrides=self._strategy_overrides_from_config(config)
        )

        self.repository = LongTermMemoryRepository(self.config["db_path"])
        self.repository.initialize()
        self.quality = QualityAssurance(
            dedup_similarity_threshold=self.config["dedup_similarity_threshold"],
            min_confidence_to_store=self.config["min_confidence_to_store"],
            strategy_config=self.strategy_config,
        )
        self.extractor = MemoryExtractor(strategy_config=self.strategy_config)
        self.event_logger = EventLogger(self.repository)
        self.confirmations = ConfirmationManager(self.repository, self.event_logger)
        self.candidates = CandidateManager(
            self.repository,
            occurrence_threshold=self.config["candidate_occurrence_threshold"],
            confidence_threshold=self.config["candidate_confidence_threshold"],
            explicit_bypass=self.config["candidate_explicit_bypass"],
            strategy_config=self.strategy_config,
        )
        self.projection = ProfileProjection(self.repository)
        self.embedding_service = MemoryEmbeddingService(
            self.repository,
            enabled=self.config["semantic_retrieval_enabled"],
            timeout_seconds=self.config["embed_timeout_seconds"],
            backfill_limit=self.config["embed_backfill_limit"],
        )
        prompt_injector_kwargs: Dict[str, Any] = {
            "embedding_service": self.embedding_service,
            "rerank_enabled": self.config["injection_rerank_enabled"],
            "rerank_timeout_seconds": self.config["rerank_timeout_seconds"],
            "strategy_config": self.strategy_config,
        }
        if isinstance(config, dict):
            if "max_prompt_tokens" in config:
                prompt_injector_kwargs["max_prompt_tokens"] = config[
                    "max_prompt_tokens"
                ]
            if "max_memories_in_prompt" in config:
                prompt_injector_kwargs["max_memories"] = config[
                    "max_memories_in_prompt"
                ]
        self.prompt_injector = PromptInjector(
            self.repository,
            **prompt_injector_kwargs,
        )
        self.deletion = LongTermMemoryDeletionService(self.repository, self.projection)
        self.backups = BackupManager(
            self.repository,
            auto_backup_interval_hours=self.config["auto_backup_interval_hours"],
        )
        self._extract_executor = ThreadPoolExecutor(
            max_workers=max(1, int(self.config["async_extract_workers"])),
            thread_name_prefix="ltm-extract",
        )
        self._pending_extract_futures: Set[Future] = set()
        self._futures_lock = Lock()
        self._shutdown = False

    def _strategy_overrides_from_config(
        self,
        config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Map constructor config into strategy overrides, preserving old keys."""

        overrides: Dict[str, Any] = {}
        if not isinstance(config, dict):
            return overrides
        injection_overrides = {}
        if "max_prompt_tokens" in config:
            injection_overrides["max_prompt_tokens"] = config["max_prompt_tokens"]
        if "max_memories_in_prompt" in config:
            injection_overrides["max_memories"] = config["max_memories_in_prompt"]
        if injection_overrides:
            overrides.setdefault("long_term", {})["injection"] = injection_overrides
        nested = config.get("selection_strategy")
        if isinstance(nested, dict):
            _deep_update(overrides, nested)
        direct = {}
        for key in ["short_term", "long_term"]:
            if isinstance(config.get(key), dict):
                direct[key] = config[key]
        if isinstance(config.get("version"), str):
            direct["version"] = config["version"]
        if direct:
            _deep_update(overrides, direct)
        return overrides

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
        """直接写入或更新一条正式长期记忆，并同步画像投影。"""

        resolved_confidence = confidence
        if resolved_confidence is None:
            resolved_confidence = self.quality.confidence_scorer.initial_confidence(
                source_type, is_explicit
            )

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
        self._schedule_embedding(item)
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
        """合并同 user/type/key 的已有记忆，保留版本和事件日志。"""

        if existing.value != value:
            self._save_version(existing, "value_update")
            old_value = existing.value
            existing.value = value
            existing.confidence = max(existing.confidence, confidence)
            self.event_logger.log_updated(
                existing.user_id,
                existing.id,
                existing.key,
                old_value,
                value,
                "value_update",
            )
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
        self._schedule_embedding(existing)
        return existing

    def _save_version(self, item: MemoryItem, reason: str):
        """保存记忆当前值的版本快照，用于回滚和审计。"""

        versions = self.repository.get_versions(item.id, limit=1)
        next_version = (versions[0].version + 1) if versions else 1
        self.repository.add_version(
            item.id, next_version, item.value, item.confidence, reason
        )

    def _schedule_embedding(self, item: Optional[MemoryItem]) -> None:
        """异步刷新 embedding 缓存，失败不影响长期记忆主链路。"""

        if not item:
            return
        try:
            if item.status == MemoryStatus.ACTIVE:
                self.embedding_service.schedule_embedding(item)
            else:
                self.repository.delete_memory_embedding(item.id)
        except Exception:
            logger.debug("长期记忆 embedding 调度失败: %s", item.id, exc_info=True)

    def extract_and_store(
        self,
        user_message: str,
        assistant_message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        conversation_window: Optional[List[Dict[str, Any]]] = None,
    ) -> List[MemoryItem]:
        """同步抽取用户画像信息，并按候选提升规则写入长期记忆。"""

        if not self._extractor_should_attempt(user_message, conversation_window):
            return []
        extracted = self._extractor_extract_from_conversation(
            user_message,
            assistant_message,
            conversation_id,
            conversation_window=conversation_window,
        )
        return self.store_extractions(
            user_id, extracted, conversation_id=conversation_id
        )

    def extract_and_store_async(
        self,
        user_message: str,
        assistant_message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        conversation_window: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Future]:
        """后台执行长期记忆抽取，避免主回答链路被 LLM 抽取阻塞。"""

        # Main flow only performs lightweight gating. Real extraction/storage runs
        # in a background worker and failures are logged without affecting replies.
        if not self._extractor_should_attempt(user_message, conversation_window):
            return None
        if self._shutdown:
            return None

        def _run() -> List[MemoryItem]:
            """在线程池中执行抽取和存储，异常时返回空结果。"""

            try:
                extracted = self._extractor_extract_from_conversation(
                    user_message,
                    assistant_message,
                    conversation_id,
                    conversation_window=conversation_window,
                )
                return self.store_extractions(
                    user_id, extracted, conversation_id=conversation_id
                )
            except Exception:
                logger.exception("长期记忆异步抽取失败: user_id=%s", user_id)
                return []
            finally:
                with self._futures_lock:
                    self._pending_extract_futures.discard(fut)

        fut = self._extract_executor.submit(_run)
        with self._futures_lock:
            self._pending_extract_futures.add(fut)
        return fut

    def _extractor_should_attempt(
        self,
        user_message: str,
        conversation_window: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """兼容旧测试 monkeypatch 的窗口感知 gating 调用。"""

        try:
            return self.extractor.should_attempt_extraction(
                user_message, conversation_window=conversation_window
            )
        except TypeError:
            return self.extractor.should_attempt_extraction(user_message)

    def _extractor_extract_from_conversation(
        self,
        user_message: str,
        assistant_message: str,
        conversation_id: Optional[str],
        conversation_window: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ExtractionResult]:
        """兼容旧测试 monkeypatch 的窗口感知抽取调用。"""

        try:
            return self.extractor.extract_from_conversation(
                user_message,
                assistant_message,
                conversation_id,
                conversation_window=conversation_window,
            )
        except TypeError:
            return self.extractor.extract_from_conversation(
                user_message,
                assistant_message,
                conversation_id,
            )

    def store_extractions(
        self,
        user_id: str,
        extracted_items: List[ExtractionResult],
        conversation_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        """把抽取结果交给候选管理器处理，返回本次提升为正式记忆的条目。"""

        promoted: List[MemoryItem] = []
        for item in extracted_items:
            if not item.should_extract:
                continue
            action = (item.action or "upsert").lower()
            if action == "revoke":
                self._process_revoke_extraction(
                    user_id=user_id,
                    item=item,
                    conversation_id=conversation_id,
                )
                continue
            if item.is_temporary:
                continue
            metadata = {
                "extraction_grade": item.extraction_grade,
                "gate_reason": item.gate_reason,
                "action": action,
                **(item.metadata or {}),
            }
            item.metadata = metadata
            if self._handle_existing_memory_extraction(
                user_id=user_id,
                item=item,
                conversation_id=conversation_id,
            ):
                continue
            stored = self.candidates.process_extraction_as_candidate(
                user_id=user_id,
                memory_type=item.memory_type,
                category=item.category,
                key=item.key,
                value=item.value,
                confidence=item.confidence,
                source_type=(
                    SourceType.EXPLICIT if item.is_explicit else item.source_type
                ),
                is_explicit=item.is_explicit,
                source_conversation_id=conversation_id,
                source_content_snippet=item.raw_content,
                metadata=metadata,
            )
            if stored:
                promoted.append(stored)
                self._save_version(stored, "candidate_promoted")
                self._schedule_embedding(stored)
        if promoted:
            self.projection.rebuild(user_id)
        return promoted

    def _handle_existing_memory_extraction(
        self,
        user_id: str,
        item: ExtractionResult,
        conversation_id: Optional[str] = None,
    ) -> bool:
        """处理与 active memory 同 type/key 的候选观察。"""

        if not item.memory_type or not item.key:
            return False
        existing = self.repository.find_memory_by_type_key(
            user_id, item.memory_type, item.key
        )
        if not existing:
            return False

        relation_type = self.quality.conflict_detector.classify_value_relation(
            existing.value,
            item.value,
            existing.memory_type,
            existing.key,
        )
        if relation_type == "same":
            existing.confidence = max(existing.confidence, item.confidence)
            existing.access_count += 1
            existing.accessed_at = _utcnow_iso()
            existing.updated_at = existing.accessed_at
            existing.metadata.update(
                {
                    "last_candidate_observed_at": existing.accessed_at,
                    "last_candidate_observation": item.raw_content,
                }
            )
            self.repository.update_memory(existing)
            self._schedule_embedding(existing)
            return True

        conflict_info = {
            "existing_memory_id": existing.id,
            "existing_value": existing.value,
            "new_value": item.value,
            "relation_type": relation_type,
            "detected_at": _utcnow_iso(),
            "source_conversation_id": conversation_id,
        }
        item.metadata["conflict_info"] = conflict_info
        item.metadata["relation_type"] = relation_type

        if relation_type in ("extension", "refinement"):
            self._save_version(existing, relation_type)
            old_value = existing.value
            existing.value = self.quality.conflict_detector.merge_values_for_relation(
                existing.value, item.value, relation_type
            )
            existing.confidence = max(existing.confidence, item.confidence)
            existing.metadata.update(
                {
                    "last_relation_type": relation_type,
                    "last_related_candidate": item.raw_content,
                    "last_related_at": conflict_info["detected_at"],
                    "previous_value": old_value,
                }
            )
            if item.source_type == SourceType.EXPLICIT:
                existing.source_type = SourceType.EXPLICIT
            if conversation_id:
                existing.source_conversation_id = conversation_id
            if item.raw_content:
                existing.source_content_snippet = item.raw_content
            existing.updated_at = conflict_info["detected_at"]
            self.repository.update_memory(existing)
            self.event_logger.log_updated(
                user_id,
                existing.id,
                existing.key,
                old_value,
                existing.value,
                relation_type,
            )
            self.projection.rebuild(user_id)
            self._schedule_embedding(existing)
            return True

        if relation_type == "conflict" and self._is_probation_active(existing):
            self._save_version(existing, "probation_conflict")
            existing.status = MemoryStatus.ARCHIVED
            existing.metadata.update(
                {
                    "probation_status": "conflicted",
                    "probation_conflict": conflict_info,
                    "archived_at": conflict_info["detected_at"],
                }
            )
            existing.updated_at = conflict_info["detected_at"]
            self.repository.update_memory(existing)
            self.repository.delete_memory_embedding(existing.id)
            self.event_logger.log_archived(user_id, existing.id, existing.key)
            self.projection.rebuild(user_id)
        return False

    def _is_probation_active(self, item: MemoryItem) -> bool:
        """判断一条 probation 记忆是否仍处于冲突观察期。"""

        metadata = item.metadata or {}
        if metadata.get("probation_status") != "active":
            return False
        probation_until = metadata.get("probation_until")
        if not probation_until:
            return False
        try:
            return datetime.fromisoformat(str(probation_until)) > datetime.now()
        except ValueError:
            return False

    def _process_revoke_extraction(
        self,
        user_id: str,
        item: ExtractionResult,
        conversation_id: Optional[str] = None,
    ) -> bool:
        """执行撤回动作：精确命中才归档/拒绝，模糊请求只落候选。"""

        revoke_metadata = {
            "is_revoked": True,
            "revoke_reason": (item.metadata or {}).get("revoke_reason")
            or item.raw_content
            or str(item.value),
            "revoked_at": _utcnow_iso(),
            "source_conversation_id": conversation_id,
            "target_text": (item.metadata or {}).get("target_text", item.raw_content),
        }
        target_specs = self._revoke_target_specs(item)
        for target in target_specs:
            memory_type = target.get("memory_type")
            key = target.get("key")
            if not memory_type or not key:
                continue

            memory = self.repository.find_memory_by_type_key(user_id, memory_type, key)
            if memory:
                self._save_version(memory, "revoked")
                memory.status = MemoryStatus.ARCHIVED
                memory.metadata.update(revoke_metadata)
                memory.updated_at = revoke_metadata["revoked_at"]
                self.repository.update_memory(memory)
                self.event_logger.log_archived(user_id, memory.id, memory.key)
                self.projection.rebuild(user_id)
                return True

            candidate = self.repository.find_candidate_by_type_key(
                user_id, memory_type, key
            )
            if candidate:
                candidate.status = MemoryStatus.REJECTED
                candidate.metadata.update(revoke_metadata)
                candidate.updated_at = revoke_metadata["revoked_at"]
                self.repository.update_candidate(candidate)
                self.repository.add_event_log(
                    EventLogEntry(
                        user_id=user_id,
                        memory_id=None,
                        event_type=EventType.CANDIDATE_REJECTED,
                        event_detail=(
                            f"候选记忆因用户撤回被拒绝: "
                            f"{candidate.memory_type}.{candidate.key}"
                        ),
                        metadata={
                            "candidate_id": candidate.id,
                            **revoke_metadata,
                        },
                    )
                )
                return True

        self._store_ambiguous_revoke_candidate(
            user_id=user_id,
            item=item,
            conversation_id=conversation_id,
            revoke_metadata=revoke_metadata,
        )
        return False

    def _revoke_target_specs(self, item: ExtractionResult) -> List[Dict[str, Any]]:
        """从撤回抽取结果中解析主目标和可选备用目标。"""

        specs: List[Dict[str, Any]] = []
        if item.memory_type and item.key:
            specs.append(
                {
                    "memory_type": item.memory_type,
                    "category": item.category,
                    "key": item.key,
                }
            )
        for target in (item.metadata or {}).get("alternate_targets", []) or []:
            if isinstance(target, dict):
                specs.append(target)
        return specs

    def _store_ambiguous_revoke_candidate(
        self,
        user_id: str,
        item: ExtractionResult,
        conversation_id: Optional[str],
        revoke_metadata: Dict[str, Any],
    ) -> None:
        """无法精确定位时，只保存撤回请求候选，避免批量误删。"""

        self.candidates.process_extraction_as_candidate(
            user_id=user_id,
            memory_type=MemoryType.CONSTRAINT,
            category="custom",
            key="revoke_request",
            value=item.value or item.raw_content or "用户撤回请求",
            confidence=min(item.confidence, 0.45),
            source_type=SourceType.AUTO,
            is_explicit=False,
            source_conversation_id=conversation_id,
            source_content_snippet=item.raw_content,
            metadata={
                "action": "revoke",
                "extraction_grade": item.extraction_grade,
                "gate_reason": item.gate_reason,
                **(item.metadata or {}),
                **revoke_metadata,
                "ambiguous_revoke": True,
            },
        )

    def upsert_profile(
        self, user_id: str, profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """兼容旧画像格式，把 profile_data 拆成正式记忆后重建画像。"""

        for key, value in (profile_data.get("preferences") or {}).items():
            self.add_memory(
                user_id=user_id,
                memory_type=MemoryType.PREFERENCE,
                category=key,
                key=key,
                value=value,
                confidence=0.9,
                source_type=SourceType.EXPLICIT,
                is_explicit=True,
            )
        for key, value in (profile_data.get("habits") or {}).items():
            stored_value = value
            existing = self.repository.find_memory_by_type_key(
                user_id, MemoryType.HABIT, key
            )
            if (
                existing
                and isinstance(existing.value, list)
                and isinstance(value, list)
            ):
                stored_value = list(dict.fromkeys(existing.value + value))
            self.add_memory(
                user_id=user_id,
                memory_type=MemoryType.HABIT,
                category=key,
                key=key,
                value=stored_value,
                confidence=0.8,
                source_type=SourceType.AUTO,
            )
        for index, constraint in enumerate(profile_data.get("constraints") or []):
            digest = hashlib.sha1(str(constraint).encode("utf-8")).hexdigest()[:12]
            constraint_key = f"constraint_{index}_{digest}"
            self.add_memory(
                user_id=user_id,
                memory_type=MemoryType.CONSTRAINT,
                category="custom",
                key=constraint_key,
                value=constraint,
                confidence=0.9,
                source_type=SourceType.EXPLICIT,
                is_explicit=True,
            )
        for key, value in (profile_data.get("background") or {}).items():
            self.add_memory(
                user_id=user_id,
                memory_type=MemoryType.BACKGROUND,
                category=key,
                key=key,
                value=value,
                confidence=0.7,
                source_type=SourceType.AUTO,
            )
        for index, fact in enumerate(profile_data.get("facts") or []):
            if not isinstance(fact, dict):
                continue
            key = fact.get("key") or f"fact_{index}"
            self.add_memory(
                user_id=user_id,
                memory_type=MemoryType.FACT,
                category=fact.get("category", "basic_info"),
                key=key,
                value=fact.get("value"),
                confidence=fact.get("confidence", 0.7),
                source_type=SourceType.AUTO,
            )
        return self.rebuild_profile(user_id)

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """读取单条长期记忆，并记录访问日志和访问次数。"""

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
        """手动更新长期记忆字段，必要时保存版本并重建画像。"""

        item = self.repository.get_memory(memory_id)
        if not item or item.user_id != user_id:
            return None
        if value is not None and value != item.value:
            self._save_version(item, "manual_update")
            old_value = item.value
            item.value = value
            self.event_logger.log_updated(
                user_id, memory_id, item.key, old_value, value, "manual_update"
            )
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
        self._schedule_embedding(item)
        return item

    def delete_memory(self, memory_id: str, user_id: str, reason: str = "") -> bool:
        """按 memory_id tombstone 删除一条长期记忆。"""

        result = self.delete(
            LongTermMemoryDeletionRequest(
                user_id=user_id, scope="memory", target_id=memory_id, reason=reason
            )
        )
        return result.deleted_memories > 0

    def delete_profile(self, user_id: str, reason: str = "") -> bool:
        """删除用户画像投影，但不直接删除正式记忆。"""

        result = self.delete(
            LongTermMemoryDeletionRequest(
                user_id=user_id, scope="profile", reason=reason
            )
        )
        return result.deleted_profiles > 0

    def delete(
        self, request: LongTermMemoryDeletionRequest
    ) -> LongTermMemoryDeletionResult:
        """执行长期记忆删除请求并返回审计结果。"""

        return self.deletion.delete(request)

    def query_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """按 MemoryQuery 查询长期记忆列表。"""

        return self.repository.query_memories(query)

    def count_memories(self, query: MemoryQuery) -> int:
        """按 MemoryQuery 统计长期记忆数量。"""

        return self.repository.count_memories(query)

    def promote_candidate(self, candidate_id: str) -> Optional[MemoryItem]:
        """强制把候选记忆提升为正式记忆，并重建画像。"""

        item = self.candidates.promote_candidate(candidate_id, force=True)
        if item:
            self._save_version(item, "candidate_promoted")
            self.projection.rebuild(item.user_id)
            self._schedule_embedding(item)
        return item

    def promote_all_eligible(self, user_id: str) -> List[MemoryItem]:
        """批量提升达到阈值的低风险候选记忆。"""

        items = self.candidates.promote_all_eligible(user_id)
        for item in items:
            self._save_version(item, "candidate_promoted")
            self._schedule_embedding(item)
        if items:
            self.projection.rebuild(user_id)
        return items

    def rebuild_profile(self, user_id: str) -> Dict[str, Any]:
        """从 active memories 重建用户画像投影。"""

        return self.projection.rebuild(user_id)

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """读取画像投影；缺失时如果存在 active memories 则自动重建。"""

        return self.projection.load_or_rebuild(user_id)

    def render_profile_prompt(
        self, user_id: str, task_type: Optional[str] = None
    ) -> str:
        """渲染完整画像 prompt 区块，主要兼容旧调用方。"""

        return self.prompt_injector.format_profile_for_prompt(
            user_id, task_type=task_type
        )

    def build_prompt_context(
        self,
        user_id: str,
        query: str,
        task_type: Optional[str] = None,
        total_context_budget: Optional[int] = None,
        task_context_profile: Optional[Any] = None,
    ) -> str:
        """按 query 选择相关长期记忆并渲染为 prompt 上下文。"""

        profile = coerce_task_context_profile(
            task_context_profile,
            query=query,
            task_type=task_type,
        )
        rendered = self.prompt_injector.format_for_prompt(
            user_id,
            query,
            task_type=task_type,
            total_context_budget=total_context_budget,
            task_context_profile=profile,
        )
        self._record_prompt_shown_feedback(
            user_id=user_id,
            query=query,
            task_type=task_type,
            task_context_profile=profile,
        )
        return rendered

    def _record_prompt_shown_feedback(
        self,
        *,
        user_id: str,
        query: str,
        task_type: Optional[str],
        task_context_profile: Optional[Any],
    ) -> None:
        """对实际注入 prompt 的长期记忆记录 shown 事件。"""

        trace = self.get_last_injection_trace()
        selected_ids = (
            trace.get("selected_memory_ids") if isinstance(trace, dict) else []
        )
        if not selected_ids:
            return
        for memory_id in selected_ids:
            if not isinstance(memory_id, str):
                continue
            self.record_memory_feedback(
                user_id=user_id,
                memory_id=memory_id,
                query=query,
                outcome="shown",
                task_type=task_type,
                task_context_profile=task_context_profile,
                metadata={"trace_version": trace.get("trace_version")},
            )

    def explain_retrieval_hits(
        self,
        user_id: str,
        query: str,
        task_type: Optional[str] = None,
        total_context_budget: Optional[int] = None,
        include_omitted: bool = False,
        task_context_profile: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """返回长期记忆命中及打分原因，供前端和调试面板展示。"""

        profile = coerce_task_context_profile(
            task_context_profile,
            query=query,
            task_type=task_type,
        )
        resolved_task_type = (
            task_type
            or (profile.task_type if profile is not None else "")
            or self.prompt_injector.classify_task_type(query)
        )
        hits = []
        for hit in self.prompt_injector.select_memory_hits(
            user_id,
            query,
            task_type=resolved_task_type,
            total_context_budget=total_context_budget,
            include_omitted=include_omitted,
            record_access=False,
            task_context_profile=profile,
        ):
            item = hit.item
            hits.append(
                {
                    "memory_id": item.id,
                    "memory_type": item.memory_type,
                    "category": item.category,
                    "key": item.key,
                    "value": item.value,
                    "confidence": item.confidence,
                    "relevance": round(hit.score, 3),
                    "reason": "；".join(hit.reasons),
                    "components": hit.components,
                    "selected": hit.selected,
                    "omitted_reason": hit.omitted_reason,
                    "token_estimate": hit.token_estimate,
                    "source_type": item.source_type,
                    "confirmed_by_user": item.confirmed_by_user,
                    "timestamp": item.updated_at or item.created_at,
                }
            )
        return hits

    def get_last_injection_trace(self) -> Dict[str, Any]:
        """返回最近一次长期记忆注入选择 trace。"""

        return self.prompt_injector.get_last_selection_trace()

    def rebuild_semantic_index(
        self, user_id: Optional[str] = None, limit: int = 500
    ) -> Dict[str, int]:
        """手动触发 active 长期记忆 embedding 缓存回填。"""

        return self.embedding_service.rebuild_index(user_id=user_id, limit=limit)

    def record_injection_feedback(
        self,
        user_id: str,
        query: str,
        assistant_message: str,
        selected_memory_ids: List[str],
        task_type: Optional[str] = None,
        task_context_profile: Optional[Any] = None,
    ) -> Dict[str, int]:
        """记录长期记忆注入后是否被回答引用，用于后续降权。"""

        result = {"updated": 0, "hit": 0, "miss": 0}
        now = _utcnow_iso()
        profile = coerce_task_context_profile(
            task_context_profile,
            query=query,
            task_type=task_type,
        )
        resolved_task_type = (
            task_type
            or (profile.task_type if profile is not None else "")
            or self.prompt_injector.classify_task_type(query)
        )
        for memory_id in selected_memory_ids or []:
            item = self.repository.get_memory(memory_id)
            if (
                not item
                or item.user_id != user_id
                or item.status != MemoryStatus.ACTIVE
            ):
                continue

            metadata = dict(item.metadata or {})
            stats = dict(metadata.get("injection_stats") or {})
            was_hit = self._memory_referenced_in_text(item, assistant_message)

            shown_count = int(stats.get("shown_count", 0) or 0) + 1
            hit_count = int(stats.get("hit_count", 0) or 0)
            miss_count = int(stats.get("miss_count", 0) or 0)
            consecutive_miss_count = int(stats.get("consecutive_miss_count", 0) or 0)
            if was_hit:
                hit_count += 1
                consecutive_miss_count = 0
                stats["last_hit_at"] = now
                result["hit"] += 1
                outcome = "hit"
            else:
                miss_count += 1
                consecutive_miss_count += 1
                result["miss"] += 1
                outcome = "miss"

            feedback = MemoryFeedbackRecord(
                user_id=user_id,
                memory_id=item.id,
                memory_type=item.memory_type,
                task_type=resolved_task_type,
                query=query,
                outcome=outcome,
                timestamp=now,
                metadata={"referenced": was_hit},
            )
            self.record_memory_feedback(
                user_id=user_id,
                memory_id=item.id,
                query=query,
                outcome=outcome,
                task_type=resolved_task_type,
                task_context_profile=profile,
                metadata={"referenced": was_hit},
                timestamp=now,
            )

            stats.update(
                {
                    "shown_count": shown_count,
                    "hit_count": hit_count,
                    "miss_count": miss_count,
                    "consecutive_miss_count": consecutive_miss_count,
                    "last_shown_at": now,
                    "last_task_type": resolved_task_type,
                    "last_query": query,
                    "last_feedback": feedback.to_dict(),
                }
            )
            by_task_type = dict(stats.get("by_task_type") or {})
            task_bucket = dict(by_task_type.get(resolved_task_type) or {})
            task_shown = int(task_bucket.get("shown_count", 0) or 0) + 1
            task_hit = int(task_bucket.get("hit_count", 0) or 0)
            task_miss = int(task_bucket.get("miss_count", 0) or 0)
            if was_hit:
                task_hit += 1
            else:
                task_miss += 1
            task_bucket.update(
                {
                    "shown_count": task_shown,
                    "hit_count": task_hit,
                    "miss_count": task_miss,
                    "memory_type": item.memory_type,
                }
            )
            by_task_type[resolved_task_type] = task_bucket
            stats["by_task_type"] = by_task_type
            metadata["injection_stats"] = stats
            item.metadata = metadata
            item.updated_at = now
            self.repository.update_memory(item)
            result["updated"] += 1
        return result

    def record_memory_feedback(
        self,
        user_id: str,
        memory_id: str,
        query: str,
        outcome: str,
        task_type: Optional[str] = None,
        task_context_profile: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """把单条长期记忆反馈写入统一事件日志。"""

        normalized_outcome = str(outcome or "").strip().lower()
        if normalized_outcome not in {"shown", "hit", "miss", "denied"}:
            raise ValueError(f"unsupported memory feedback outcome: {outcome!r}")

        item = self.repository.get_memory(memory_id)
        if not item or item.user_id != user_id:
            return None
        profile = coerce_task_context_profile(
            task_context_profile,
            query=query,
            task_type=task_type,
        )
        resolved_task_type = (
            task_type
            or (profile.task_type if profile is not None else "")
            or self.prompt_injector.classify_task_type(query)
        )
        record = MemoryFeedbackRecord(
            user_id=user_id,
            memory_id=item.id,
            memory_type=item.memory_type,
            task_type=resolved_task_type,
            query=query,
            outcome=normalized_outcome,
            timestamp=timestamp or _utcnow_iso(),
            metadata={
                **(metadata or {}),
                **(
                    {"task_context_profile": profile.to_dict()}
                    if profile is not None
                    else {}
                ),
            },
        )
        self.repository.add_event_log(
            EventLogEntry(
                user_id=user_id,
                memory_id=item.id,
                event_type=EventType.FEEDBACK_RECORDED,
                event_detail=f"长期记忆反馈: {normalized_outcome}",
                metadata={"feedback": record.to_dict()},
                created_at=record.timestamp,
            )
        )
        return record.to_dict()

    def get_feedback_records(
        self,
        user_id: str,
        memory_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """从 memory_event_log 查询统一长期记忆反馈记录。"""

        logs = self.get_event_logs(
            user_id,
            memory_id=memory_id,
            limit=limit,
            offset=offset,
            event_type=EventType.FEEDBACK_RECORDED,
        )
        records: List[Dict[str, Any]] = []
        for log in logs:
            payload = (log.metadata or {}).get("feedback")
            if not isinstance(payload, dict):
                continue
            record = MemoryFeedbackRecord.from_mapping(payload).to_dict()
            record["event_log_id"] = log.id
            record["created_at"] = log.created_at
            records.append(record)
        return records

    def _memory_referenced_in_text(self, item: MemoryItem, text: str) -> bool:
        """判断回答文本是否引用了某条已注入记忆的 key 或 value。"""

        haystack = str(text or "").lower()
        if not haystack:
            return False

        needles = [str(item.key or "")]
        if isinstance(item.value, list):
            needles.extend(str(value) for value in item.value)
        elif isinstance(item.value, dict):
            for key, value in item.value.items():
                needles.extend([str(key), str(value)])
        elif item.value is not None:
            needles.append(str(item.value))

        for needle in needles:
            normalized = needle.strip().lower()
            if len(normalized) >= 2 and normalized in haystack:
                return True
        return False

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
        """兼容旧 memory_event 格式的记录入口。"""

        event = MemoryEvent(
            user_id=user_id,
            event_type=event_type,
            key=key,
            value=value,
            source_text=source_text,
            confidence=(
                confidence
                if confidence is not None
                else self.config["default_confidence"]
            ),
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
                metadata={
                    "legacy_status": status,
                    "source_text": source_text,
                    **(metadata or {}),
                },
            )
        )
        return stored

    def get_versions(self, memory_id: str, limit: int = 20) -> List[MemoryVersion]:
        """读取指定长期记忆的版本历史。"""

        return self.repository.get_versions(memory_id, limit=limit)

    def restore_version(
        self, memory_id: str, version: int, user_id: str
    ) -> Optional[MemoryItem]:
        """把一条长期记忆恢复到指定历史版本。"""

        item = self.repository.get_memory(memory_id)
        if not item or item.user_id != user_id:
            return None

        target = next(
            (
                stored
                for stored in self.repository.get_versions(memory_id, limit=50)
                if stored.version == version
            ),
            None,
        )
        if not target:
            return None

        self._save_version(item, f"restore_to_v{version}")
        old_value = item.value
        item.value = target.value
        item.confidence = target.confidence
        item.updated_at = _utcnow_iso()
        self.repository.update_memory(item)
        self.event_logger.log_updated(
            user_id,
            memory_id,
            item.key,
            old_value,
            target.value,
            f"restore_to_v{version}",
        )
        self.projection.rebuild(user_id)
        return item

    def get_event_logs(
        self,
        user_id: str,
        memory_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        event_type: Optional[str] = None,
    ) -> List[EventLogEntry]:
        """读取用户长期记忆事件日志，可按 memory_id 过滤。"""

        return self.event_logger.get_event_logs(
            user_id,
            memory_id=memory_id,
            limit=limit,
            offset=offset,
            event_type=event_type,
        )

    def get_memory_trace(self, user_id: str, memory_id: str) -> List[EventLogEntry]:
        """读取单条长期记忆的完整生命周期事件轨迹。"""

        return self.event_logger.get_memory_trace(user_id, memory_id)

    def list_candidates(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[MemoryCandidate]:
        """列出用户候选记忆，可按候选状态过滤。"""

        return self.candidates.list_candidates(
            user_id, limit=limit, offset=offset, status=status
        )

    def reject_candidate(self, candidate_id: str, reason: str = "") -> bool:
        """拒绝候选记忆并写入事件日志。"""

        return self.candidates.reject_candidate(candidate_id, reason)

    def list_pending_confirmations(
        self, user_id: str, limit: int = 20
    ) -> List[MemoryConfirmation]:
        """列出用户待确认的长期记忆确认请求。"""

        return self.confirmations.list_pending_confirmations(user_id, limit=limit)

    def resolve_confirmation(
        self, confirmation_id: str, status: str
    ) -> Optional[MemoryConfirmation]:
        """按确认结果更新单条确认请求状态。"""

        return self.confirmations.resolve_confirmation(confirmation_id, status)

    def batch_confirm(
        self, user_id: str, confirmation_ids: List[str], status: str
    ) -> List[MemoryConfirmation]:
        """批量处理用户的长期记忆确认请求。"""

        return self.confirmations.batch_confirm(user_id, confirmation_ids, status)

    def run_maintenance(self, user_id: str) -> Dict[str, int]:
        """执行长期记忆维护：过期、归档、候选提升、画像重建和自动备份。"""

        expired = self.repository.expire_old_memories(user_id, _utcnow_iso())
        archived = self.repository.archive_unused_memories(user_id)
        promoted = self.promote_all_eligible(user_id)
        semantic = self.embedding_service.backfill_missing_or_stale(
            user_id=user_id,
            limit=self.config["embed_backfill_limit"],
        )

        self.projection.rebuild(user_id)
        self.backups.maybe_auto_backup()

        result = {
            "expired": expired,
            "archived": archived,
            "promoted": len(promoted),
            "semantic_created": semantic.get("created", 0),
            "semantic_updated": semantic.get("updated", 0),
            "semantic_skipped": semantic.get("skipped", 0),
            "semantic_failed": semantic.get("failed", 0),
        }
        if any(result.values()):
            logger.info("长期记忆维护完成 (user_id: %s): %s", user_id, result)
        return result

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        """Shut down the background extraction executor.

        Args:
            wait: If True, wait for running futures to finish.
            cancel_futures: If True, cancel pending futures.
        """
        if self._shutdown:
            return
        self._shutdown = True
        try:
            self._extract_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        except TypeError:
            # Python < 3.9 does not support cancel_futures
            self._extract_executor.shutdown(wait=wait)
        self.embedding_service.shutdown(wait=wait, cancel_futures=cancel_futures)
        with self._futures_lock:
            self._pending_extract_futures.clear()

    def flush_extractions(self, timeout: float = 5.0) -> None:
        """Wait for all pending async extraction futures to complete.

        Args:
            timeout: Maximum total wait time in seconds.
        """
        import time

        deadline = time.monotonic() + timeout
        with self._futures_lock:
            pending = list(self._pending_extract_futures)
        for fut in pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "flush_extractions timeout: %d futures still pending",
                    len([f for f in pending if not f.done()]),
                )
                break
            try:
                fut.result(timeout=max(0.1, remaining))
            except Exception:
                pass
        # Clean up completed futures
        with self._futures_lock:
            self._pending_extract_futures = {
                f for f in self._pending_extract_futures if not f.done()
            }
        self.embedding_service.flush(timeout=timeout)

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """返回长期记忆统计信息。"""

        return self.repository.get_memory_stats(user_id)

    def export_profile_snapshot(self, user_id: str) -> Dict[str, Any]:
        """导出画像、active 记忆数量和统计信息的快照。"""

        profile = self.repository.load_profile(user_id)
        if not profile:
            return {}
        memories = self.repository.query_memories(
            MemoryQuery(user_id=user_id, status=MemoryStatus.ACTIVE, limit=100)
        )
        return {
            "user_id": user_id,
            "profile": profile,
            "active_memory_count": len(memories),
            "stats": self.repository.get_memory_stats(user_id),
        }

    def create_backup(self, tag: Optional[str] = None) -> Optional[str]:
        """创建长期记忆数据库备份。"""

        return self.backups.create_backup(tag)

    def list_backups(self) -> List[dict]:
        """列出长期记忆数据库备份文件。"""

        return self.backups.list_backups()

    def restore_from_backup(self, backup_path: str) -> bool:
        """从指定备份恢复长期记忆数据库。"""

        return self.backups.restore_from_backup(backup_path)
