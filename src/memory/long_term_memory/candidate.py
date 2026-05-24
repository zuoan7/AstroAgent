"""长期记忆候选管理。

抽取结果先进入候选表，候选按稳定性、一致性、来源和类型阈值评分后提升；
一般事实不自动转正，背景类只允许可校验类别在高一致性下自动转正。
"""

from datetime import datetime, timedelta
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
from src.memory.long_term_memory.quality import CandidatePromotionEvaluator
from src.memory.long_term_memory.repository import LongTermMemoryRepository
from src.memory.selection_strategy_config import (
    MemorySelectionStrategyConfig,
    get_memory_selection_strategy_config,
)


class CandidateManager:
    """管理候选记忆的创建、更新、提升和拒绝。"""

    PROMOTION_OCCURRENCE_THRESHOLD = 2
    PROMOTION_CONFIDENCE_THRESHOLD = 0.6
    PROMOTION_EXPLICIT_BYPASS = True
    HIGH_RISK_MEMORY_TYPES = {MemoryType.BACKGROUND, MemoryType.FACT}
    LOW_RISK_MEMORY_TYPES = {
        MemoryType.PREFERENCE,
        MemoryType.HABIT,
        MemoryType.CONSTRAINT,
    }

    def __init__(
        self,
        repository: LongTermMemoryRepository,
        occurrence_threshold: int = 2,
        confidence_threshold: float = 0.6,
        explicit_bypass: bool = True,
        strategy_config: MemorySelectionStrategyConfig | None = None,
    ):
        """初始化候选仓储、旧阈值配置和新转正评分器。"""

        self._repo = repository
        self._strategy_config = (
            strategy_config or get_memory_selection_strategy_config()
        )
        self.occurrence_threshold = occurrence_threshold
        self.confidence_threshold = confidence_threshold
        self.explicit_bypass = explicit_bypass
        self.promotion_evaluator = CandidatePromotionEvaluator(
            strategy_config=self._strategy_config
        )

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
        """新增候选或提升已有候选的出现次数和置信度。"""

        existing = self._repo.find_candidate_by_type_key(user_id, memory_type, key)
        now = _utcnow_iso()
        occurrence = self._build_occurrence_record(
            value=value,
            confidence=confidence,
            source_type=source_type,
            source_conversation_id=source_conversation_id,
            metadata=metadata,
            seen_at=now,
        )

        if existing:
            existing.occurrence_count += 1
            existing.last_seen_at = now
            existing.updated_at = existing.last_seen_at
            existing.confidence = min(max(existing.confidence, confidence) + 0.05, 0.9)
            if source_type == SourceType.EXPLICIT:
                existing.source_type = SourceType.EXPLICIT
                existing.confidence = min(existing.confidence + 0.2, 0.95)
            existing.value = self._merge_candidate_value(existing, value)
            if source_content_snippet:
                existing.source_content_snippet = source_content_snippet
            if metadata:
                existing.metadata.update(metadata)
            existing.metadata["occurrence_history"] = self._append_occurrence_history(
                existing.metadata.get("occurrence_history"),
                occurrence,
            )
            self._repo.update_candidate(existing)
            self._repo.add_event_log(
                EventLogEntry(
                    user_id=user_id,
                    memory_id=None,
                    event_type=EventType.CANDIDATE_CREATED,
                    event_detail=f"候选记忆更新: {memory_type}.{key}",
                    new_value=str(value),
                    metadata={
                        "candidate_id": existing.id,
                        "occurrence_count": existing.occurrence_count,
                    },
                )
            )
            logger.debug(
                f"候选记忆更新: {memory_type}.{key} (出现{existing.occurrence_count}次)"
            )
            return existing

        candidate_metadata = dict(metadata or {})
        candidate_metadata["occurrence_history"] = self._append_occurrence_history(
            candidate_metadata.get("occurrence_history"),
            occurrence,
        )
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
            metadata=candidate_metadata,
        )
        self._repo.add_candidate(candidate)
        self._repo.add_event_log(
            EventLogEntry(
                user_id=user_id,
                memory_id=None,
                event_type=EventType.CANDIDATE_CREATED,
                event_detail=f"候选记忆创建: {memory_type}.{key}",
                new_value=str(value),
                metadata={
                    "candidate_id": candidate.id,
                    "confidence": candidate.confidence,
                },
            )
        )
        logger.debug(f"新候选记忆: {memory_type}.{key}")
        return candidate

    def is_high_risk_candidate(self, candidate: MemoryCandidate) -> bool:
        """判断候选是否属于需要确认的高风险画像类型。"""

        return candidate.memory_type in self.HIGH_RISK_MEMORY_TYPES

    def _set_candidate_status(
        self, candidate: MemoryCandidate, status: str
    ) -> MemoryCandidate:
        """更新候选状态并刷新更新时间。"""

        candidate.status = status
        candidate.updated_at = _utcnow_iso()
        self._repo.update_candidate(candidate)
        return candidate

    def should_promote(self, candidate: MemoryCandidate) -> bool:
        """根据稳定性、一致性、置信度和类型阈值判断能否自动提升。"""

        decision = self.promotion_evaluator.evaluate(candidate)
        candidate.metadata["promotion_decision"] = decision.to_dict()
        candidate.updated_at = _utcnow_iso()
        try:
            self._repo.update_candidate(candidate)
        except Exception:
            logger.debug("候选转正决策写回失败: %s", candidate.id, exc_info=True)
        return decision.should_promote

    def promote_candidate(
        self, candidate_id: str, force: bool = False
    ) -> Optional[MemoryItem]:
        """把候选转为正式记忆；未达阈值时可保持候选或转待确认。"""

        candidate = self._repo.get_candidate(candidate_id)
        if not candidate:
            logger.warning(f"候选记忆不存在: {candidate_id}")
            return None

        if not force and not self.should_promote(candidate):
            if self.is_high_risk_candidate(candidate):
                self._set_candidate_status(candidate, MemoryStatus.NEEDS_CONFIRM)
            logger.debug(f"候选记忆未达提升标准: {candidate_id}")
            return None

        memory_item = candidate.to_memory_item()
        self._apply_probation_metadata(memory_item, candidate)
        self._repo.add_memory(memory_item)
        self._repo.mark_candidate_promoted(candidate_id, memory_item.id)

        self._repo.add_event_log(
            EventLogEntry(
                user_id=candidate.user_id,
                memory_id=memory_item.id,
                event_type=EventType.CANDIDATE_PROMOTED,
                event_detail=f"候选记忆提升为正式记忆: {candidate.memory_type}.{candidate.key}",
                new_value=str(candidate.value),
                metadata={
                    "candidate_id": candidate_id,
                    "occurrence_count": candidate.occurrence_count,
                    "promotion_decision": candidate.metadata.get("promotion_decision"),
                },
            )
        )

        logger.info(
            f"候选记忆提升: {candidate.memory_type}.{candidate.key} (出现{candidate.occurrence_count}次)"
        )
        return memory_item

    def reject_candidate(self, candidate_id: str, reason: str = "") -> bool:
        """拒绝候选记忆并记录事件。"""

        candidate = self._repo.get_candidate(candidate_id)
        if not candidate:
            return False

        self._set_candidate_status(candidate, MemoryStatus.REJECTED)
        self._repo.add_event_log(
            EventLogEntry(
                user_id=candidate.user_id,
                memory_id=None,
                event_type=EventType.CANDIDATE_REJECTED,
                event_detail=f"候选记忆被拒绝: {candidate.memory_type}.{candidate.key}",
                metadata={"candidate_id": candidate_id, "reason": reason},
            )
        )
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryItem]:
        """处理一条抽取结果，必要时创建候选、待确认或正式记忆。"""

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
            metadata=metadata,
        )

        if self.should_promote(candidate):
            return self.promote_candidate(candidate.id)

        conflict_info = candidate.metadata.get("conflict_info") or {}
        if conflict_info.get("relation_type") in ("conflict", "unknown"):
            self._set_candidate_status(candidate, MemoryStatus.NEEDS_CONFIRM)
            return None

        if self.is_high_risk_candidate(candidate):
            self._set_candidate_status(candidate, MemoryStatus.NEEDS_CONFIRM)
            return None

        if candidate.status != MemoryStatus.CANDIDATE:
            self._set_candidate_status(candidate, MemoryStatus.CANDIDATE)
        return None

    def list_candidates(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[MemoryCandidate]:
        """列出用户候选记忆，可按状态筛选。"""

        return self._repo.list_candidates(
            user_id, limit=limit, offset=offset, status=status
        )

    def get_candidate(self, candidate_id: str) -> Optional[MemoryCandidate]:
        """按候选 id 读取单条候选记忆。"""

        return self._repo.get_candidate(candidate_id)

    def promote_all_eligible(self, user_id: str) -> List[MemoryItem]:
        """批量提升当前用户所有达到自动提升条件的候选。"""

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

    def _build_occurrence_record(
        self,
        value: Any,
        confidence: float,
        source_type: str,
        source_conversation_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        seen_at: str,
    ) -> Dict[str, Any]:
        """构造一次候选出现记录，供稳定性/一致性评分使用。"""

        metadata = metadata or {}
        return {
            "value": value,
            "confidence": confidence,
            "source_type": source_type,
            "source_conversation_id": source_conversation_id,
            "seen_at": seen_at,
            "extraction_grade": metadata.get("extraction_grade"),
            "gate_reason": metadata.get("gate_reason"),
        }

    def _append_occurrence_history(
        self, history: Any, occurrence: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """追加候选出现历史，并限制保留最近 20 条。"""

        normalized = [item for item in (history or []) if isinstance(item, dict)]
        normalized.append(occurrence)
        return normalized[-20:]

    def _merge_candidate_value(self, candidate: MemoryCandidate, value: Any) -> Any:
        """根据候选新旧值关系合并、记录冲突或替换候选值。"""

        relation_type = (
            self.promotion_evaluator.conflict_detector.classify_value_relation(
                candidate.value,
                value,
                candidate.memory_type,
                candidate.key,
            )
        )
        if relation_type in ("extension", "refinement"):
            candidate.metadata["candidate_value_relation"] = relation_type
            return self.promotion_evaluator.conflict_detector.merge_values_for_relation(
                candidate.value, value, relation_type
            )
        if relation_type == "conflict":
            candidate.metadata["candidate_value_relation"] = relation_type
            candidate.metadata["candidate_conflict_value"] = value
        return candidate.value if relation_type == "same" else value

    def _apply_probation_metadata(
        self, memory_item: MemoryItem, candidate: MemoryCandidate
    ) -> None:
        """候选转正后写入 30 天 probation 观察期元数据。"""

        promotion_decision = candidate.metadata.get("promotion_decision") or (
            self.promotion_evaluator.evaluate(candidate).to_dict()
        )
        probation_until = (datetime.now() + timedelta(days=30)).isoformat()
        memory_item.metadata.update(
            {
                "probation_until": probation_until,
                "probation_status": "active",
                "promotion_score": promotion_decision.get("promote_score"),
                "promotion_components": {
                    "stability": promotion_decision.get("stability"),
                    "consistency": promotion_decision.get("consistency"),
                    "effective_count": promotion_decision.get("effective_count"),
                    "source_weight": promotion_decision.get("source_weight"),
                    "threshold": promotion_decision.get("threshold"),
                    "reason": promotion_decision.get("reason"),
                },
            }
        )
