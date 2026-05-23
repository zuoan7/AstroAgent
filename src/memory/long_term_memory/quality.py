"""长期记忆质量控制。

集中管理置信度、冲突检测、去重、过期和归档规则，供长期记忆服务在写入和
维护阶段复用。
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.core.logger import logger
from src.memory.long_term_memory.models import (
    ConflictInfo,
    ConflictResolution,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    SourceType,
    _utcnow_iso,
)


class ConfidenceScorer:
    """根据来源、确认、访问和时间衰减计算长期记忆置信度。"""

    def __init__(
        self,
        explicit_base: float = 0.85,
        auto_base: float = 0.5,
        confirmation_boost: float = 0.15,
        access_boost: float = 0.01,
        max_confidence: float = 1.0,
        decay_per_day: float = 0.005,
    ):
        self.explicit_base = explicit_base
        self.auto_base = auto_base
        self.confirmation_boost = confirmation_boost
        self.access_boost = access_boost
        self.max_confidence = max_confidence
        self.decay_per_day = decay_per_day

    def initial_confidence(self, source_type: str, is_explicit: bool = False) -> float:
        """按来源类型给新记忆分配初始置信度。"""

        if is_explicit or source_type == SourceType.EXPLICIT:
            return self.explicit_base
        if source_type == SourceType.CONFIRMED:
            return 0.95
        if source_type == SourceType.MANUAL:
            return 0.9
        return self.auto_base

    def boost_on_confirmation(self, current: float, count: int = 1) -> float:
        return min(current + self.confirmation_boost * count, self.max_confidence)

    def boost_on_access(self, current: float, access_count: int) -> float:
        return min(current + self.access_boost * min(access_count, 10), self.max_confidence)

    def apply_time_decay(self, current: float, days_since_access: int) -> float:
        decayed = current - self.decay_per_day * days_since_access
        return max(decayed, 0.1)

    def compute_confidence(
        self,
        source_type: str,
        is_explicit: bool = False,
        confirmation_count: int = 0,
        access_count: int = 0,
        days_since_access: int = 0,
    ) -> float:
        """综合确认、访问和时间衰减计算置信度。"""

        conf = self.initial_confidence(source_type, is_explicit)
        conf = self.boost_on_confirmation(conf, confirmation_count)
        conf = self.boost_on_access(conf, access_count)
        if days_since_access > 0:
            conf = self.apply_time_decay(conf, days_since_access)
        return round(min(conf, self.max_confidence), 3)


class ConflictDetector:
    """识别同 key 记忆的新旧值冲突并给出默认解决策略。"""

    def detect_conflict(
        self, existing: MemoryItem, new_value: Any, new_confidence: float
    ) -> Optional[ConflictInfo]:
        """比较已有记忆与新值，返回冲突信息或 None。"""

        if existing.value == new_value:
            return None

        if existing.memory_type in (MemoryType.PREFERENCE, MemoryType.BACKGROUND):
            if existing.key == new_value:
                return None
            return ConflictInfo(
                existing_id=existing.id,
                existing_value=existing.value,
                existing_confidence=existing.confidence,
                existing_updated_at=existing.updated_at,
                new_value=new_value,
                new_confidence=new_confidence,
                conflict_type="value_mismatch",
                resolution=self._resolve_preference_conflict(existing, new_confidence),
            )

        if existing.memory_type == MemoryType.CONSTRAINT:
            return ConflictInfo(
                existing_id=existing.id,
                existing_value=existing.value,
                existing_confidence=existing.confidence,
                existing_updated_at=existing.updated_at,
                new_value=new_value,
                new_confidence=new_confidence,
                conflict_type="constraint_conflict",
                resolution=ConflictResolution.NEEDS_CONFIRM,
            )

        return ConflictInfo(
            existing_id=existing.id,
            existing_value=existing.value,
            existing_confidence=existing.confidence,
            existing_updated_at=existing.updated_at,
            new_value=new_value,
            new_confidence=new_confidence,
            conflict_type="value_mismatch",
            resolution=ConflictResolution.UPDATE,
        )

    def _resolve_preference_conflict(
        self, existing: MemoryItem, new_confidence: float
    ) -> str:
        if existing.source_type == SourceType.EXPLICIT and new_confidence < existing.confidence:
            return ConflictResolution.NEEDS_CONFIRM
        if new_confidence > existing.confidence + 0.2:
            return ConflictResolution.OVERWRITE
        if existing.source_type in (SourceType.CONFIRMED, SourceType.MANUAL):
            return ConflictResolution.NEEDS_CONFIRM
        return ConflictResolution.UPDATE


class Deduplicator:
    """基于简单相似度规则检测和合并重复记忆。"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def compute_similarity(self, value_a: Any, value_b: Any) -> float:
        """计算两个记忆值的相似度，支持 list/dict 和普通字符串。"""

        str_a = json.dumps(value_a, ensure_ascii=False, sort_keys=True) if isinstance(value_a, (dict, list)) else str(value_a)
        str_b = json.dumps(value_b, ensure_ascii=False, sort_keys=True) if isinstance(value_b, (dict, list)) else str(value_b)

        if str_a == str_b:
            return 1.0

        if isinstance(value_a, list) and isinstance(value_b, list):
            set_a = set(str(x) for x in value_a)
            set_b = set(str(x) for x in value_b)
            if not set_a and not set_b:
                return 1.0
            if not set_a or not set_b:
                return 0.0
            intersection = set_a & set_b
            union = set_a | set_b
            return len(intersection) / len(union)

        len_a, len_b = len(str_a), len(str_b)
        if len_a == 0 and len_b == 0:
            return 1.0
        if len_a == 0 or len_b == 0:
            return 0.0

        max_len = max(len_a, len_b)
        distance = self._levenshtein_distance(str_a, str_b)
        similarity = 1.0 - (distance / max_len)
        return similarity

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def is_duplicate(self, value_a: Any, value_b: Any) -> bool:
        return self.compute_similarity(value_a, value_b) >= self.similarity_threshold

    def find_duplicates(
        self, new_value: Any, existing_items: List[MemoryItem]
    ) -> List[Tuple[MemoryItem, float]]:
        """在已有记忆中找出与新值超过阈值的重复项。"""

        duplicates = []
        for item in existing_items:
            sim = self.compute_similarity(new_value, item.value)
            if sim >= self.similarity_threshold:
                duplicates.append((item, sim))
        return sorted(duplicates, key=lambda x: x[1], reverse=True)

    def merge_values(self, existing_value: Any, new_value: Any, memory_type: str) -> Any:
        """按值类型合并重复记忆，列表去重合并，字典覆盖合并。"""

        if isinstance(existing_value, list) and isinstance(new_value, list):
            merged = list(existing_value)
            for item in new_value:
                if item not in merged:
                    merged.append(item)
            return merged

        if isinstance(existing_value, dict) and isinstance(new_value, dict):
            merged = dict(existing_value)
            merged.update(new_value)
            return merged

        return new_value


class ExpiryManager:
    """计算长期记忆过期时间，并判断是否应过期或归档。"""

    def __init__(
        self,
        default_expiry_days: int = 180,
        constraint_expiry_days: int = 365,
        fact_expiry_days: int = 730,
        archive_after_days_unused: int = 90,
    ):
        self.default_expiry_days = default_expiry_days
        self.constraint_expiry_days = constraint_expiry_days
        self.fact_expiry_days = fact_expiry_days
        self.archive_after_days_unused = archive_after_days_unused

    def compute_expiry_date(self, memory_type: str, source_type: str) -> Optional[str]:
        """按记忆类型和来源计算 expires_at；已确认记忆默认不过期。"""

        if source_type == SourceType.CONFIRMED:
            return None

        days_map = {
            MemoryType.PREFERENCE: self.default_expiry_days,
            MemoryType.HABIT: self.default_expiry_days,
            MemoryType.CONSTRAINT: self.constraint_expiry_days,
            MemoryType.BACKGROUND: self.default_expiry_days,
            MemoryType.FACT: self.fact_expiry_days,
        }
        days = days_map.get(memory_type, self.default_expiry_days)
        try:
            from datetime import timedelta
            expiry = datetime.now() + timedelta(days=days)
            return expiry.isoformat()
        except Exception:
            return None

    def is_expired(self, item: MemoryItem) -> bool:
        if not item.expires_at:
            return False
        try:
            return item.expires_at < _utcnow_iso()
        except Exception:
            return False

    def should_archive(self, item: MemoryItem) -> bool:
        if item.status != MemoryStatus.ACTIVE:
            return False
        if not item.accessed_at:
            try:
                from datetime import timedelta
                created = datetime.fromisoformat(item.created_at)
                cutoff = datetime.now() - timedelta(days=self.archive_after_days_unused)
                return created < cutoff and item.access_count == 0
            except Exception:
                return False
        try:
            from datetime import timedelta
            accessed = datetime.fromisoformat(item.accessed_at)
            cutoff = datetime.now() - timedelta(days=self.archive_after_days_unused)
            return accessed < cutoff
        except Exception:
            return False

    def invalidate_memory(self, reason: str = "user_denied") -> str:
        return MemoryStatus.EXPIRED


class QualityAssurance:
    """长期记忆质量策略聚合器。"""

    def __init__(
        self,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        conflict_detector: Optional[ConflictDetector] = None,
        deduplicator: Optional[Deduplicator] = None,
        expiry_manager: Optional[ExpiryManager] = None,
        min_confidence_to_store: float = 0.3,
        dedup_similarity_threshold: float = 0.85,
    ):
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.conflict_detector = conflict_detector or ConflictDetector()
        self.deduplicator = deduplicator or Deduplicator(similarity_threshold=dedup_similarity_threshold)
        self.expiry_manager = expiry_manager or ExpiryManager()
        self.min_confidence_to_store = min_confidence_to_store

    def should_store(self, confidence: float, is_explicit: bool = False) -> bool:
        """判断候选置信度是否达到存储阈值。"""

        if is_explicit:
            return True
        return confidence >= self.min_confidence_to_store

    def detect_conflicts(
        self, new_type: str, new_key: str, new_value: Any, new_confidence: float, existing_items: List[MemoryItem]
    ) -> List[ConflictInfo]:
        """在同 type/key 的 active memories 中检测冲突。"""

        conflicts = []
        for item in existing_items:
            if item.memory_type == new_type and item.key == new_key and item.status == MemoryStatus.ACTIVE:
                conflict = self.conflict_detector.detect_conflict(item, new_value, new_confidence)
                if conflict:
                    conflicts.append(conflict)
        return conflicts

    def resolve_conflict(self, conflict: ConflictInfo, strategy: Optional[str] = None) -> ConflictResolution:
        """根据显式策略或默认策略决定冲突处理方式。"""

        resolution = strategy or conflict.resolution

        if resolution == ConflictResolution.NEEDS_CONFIRM:
            if conflict.new_confidence > conflict.existing_confidence + 0.3:
                return ConflictResolution.OVERWRITE
            return ConflictResolution.NEEDS_CONFIRM

        return resolution

    def check_duplicates(
        self, new_value: Any, existing_items: List[MemoryItem]
    ) -> List[Tuple[MemoryItem, float]]:
        return self.deduplicator.find_duplicates(new_value, existing_items)

    def merge_duplicate(
        self, existing: MemoryItem, new_value: Any, new_confidence: float
    ) -> Tuple[Any, float]:
        merged_value = self.deduplicator.merge_values(existing.value, new_value, existing.memory_type)
        merged_confidence = max(existing.confidence, new_confidence)
        return merged_value, merged_confidence

    def compute_expiry(self, memory_type: str, source_type: str) -> Optional[str]:
        return self.expiry_manager.compute_expiry_date(memory_type, source_type)

    def check_expiry(self, item: MemoryItem) -> bool:
        return self.expiry_manager.is_expired(item)

    def should_archive(self, item: MemoryItem) -> bool:
        return self.expiry_manager.should_archive(item)
