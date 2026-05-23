"""长期记忆质量控制。

集中管理置信度、冲突检测、候选转正、去重、过期和归档规则。这里承接
select_strategy 的长期记忆转正策略：按稳定性、一致性、来源权重和类型阈值
评分，并对冲突、扩展、细化和撤回反馈提供统一判断。
"""

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.memory.long_term_memory.models import (
    ConflictInfo,
    ConflictResolution,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    SourceType,
    _utcnow_iso,
)


EXPANDABLE_KEYS = {
    "location",
    "location_info",
    "frequent_topics",
    "domain_experience",
    "observation_type",
}

MUTUALLY_EXCLUSIVE_KEYS = {
    "response_style",
    "knowledge_level",
    "skill_level",
    "device_info",
    "equipment",
    "timezone",
    "unit_preference",
}

AUTO_PROMOTABLE_BACKGROUND_CATEGORIES = {
    "device_info",
    "location",
    "skill_level",
    "domain_experience",
}


@dataclass
class PromotionDecision:
    """候选转正评分结果。"""

    should_promote: bool
    promote_score: float
    threshold: Optional[float]
    stability: float
    consistency: float
    effective_count: float
    source_weight: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """序列化候选转正评分结果，方便写入 metadata 和 trace。"""

        return {
            "should_promote": self.should_promote,
            "promote_score": self.promote_score,
            "threshold": self.threshold,
            "stability": self.stability,
            "consistency": self.consistency,
            "effective_count": self.effective_count,
            "source_weight": self.source_weight,
            "reason": self.reason,
        }


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
        """初始化不同来源的置信度基线、加成和时间衰减参数。"""

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
        """用户确认后按确认次数提升置信度。"""

        return min(current + self.confirmation_boost * count, self.max_confidence)

    def boost_on_access(self, current: float, access_count: int) -> float:
        """记忆被访问后按访问次数小幅提升置信度。"""

        return min(current + self.access_boost * min(access_count, 10), self.max_confidence)

    def apply_time_decay(self, current: float, days_since_access: int) -> float:
        """按未访问天数衰减置信度，并保留最低置信下限。"""

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

    def classify_value_relation(
        self,
        existing_value: Any,
        new_value: Any,
        memory_type: str = "",
        key: str = "",
    ) -> str:
        """确定性地区分相同、细化、范围扩展、真冲突和未知关系。"""

        if self._canonical(existing_value) == self._canonical(new_value):
            return "same"

        existing_items = self._as_sequence(existing_value)
        new_items = self._as_sequence(new_value)
        if existing_items is not None or new_items is not None:
            old_set = set(existing_items or [str(existing_value)])
            new_set = set(new_items or [str(new_value)])
            if old_set == new_set:
                return "same"
            if old_set and old_set.issubset(new_set):
                return "extension"
            if new_set and new_set.issubset(old_set):
                return "same"
            if key in EXPANDABLE_KEYS or memory_type in (MemoryType.HABIT,):
                return "extension"
            return "conflict"

        old_text = str(existing_value).strip()
        new_text = str(new_value).strip()
        old_norm = self._normalize_text(old_text)
        new_norm = self._normalize_text(new_text)
        if old_norm == new_norm:
            return "same"

        if old_norm and old_norm in new_norm:
            if key in EXPANDABLE_KEYS or self._looks_like_range_extension(new_text):
                return "extension"
            return "refinement"
        if new_norm and new_norm in old_norm:
            return "same"

        if key in EXPANDABLE_KEYS and self._shares_location_or_topic(old_text, new_text):
            return "extension"

        if key in MUTUALLY_EXCLUSIVE_KEYS or memory_type in (
            MemoryType.PREFERENCE,
            MemoryType.CONSTRAINT,
        ):
            return "conflict"

        if self._shared_token_count(old_norm, new_norm) > 0:
            return "unknown"
        return "conflict"

    def merge_values_for_relation(
        self, existing_value: Any, new_value: Any, relation_type: str
    ) -> Any:
        """按关系合并值，供 extension/refinement 更新正式记忆。"""

        if relation_type == "extension":
            existing_items = self._as_sequence(existing_value)
            new_items = self._as_sequence(new_value)
            if existing_items is not None or new_items is not None:
                merged = []
                for item in (existing_items or [existing_value]) + (
                    new_items or [new_value]
                ):
                    if item not in merged:
                        merged.append(item)
                return merged
            return new_value
        if relation_type == "refinement":
            return new_value
        return existing_value

    def detect_conflict(
        self, existing: MemoryItem, new_value: Any, new_confidence: float
    ) -> Optional[ConflictInfo]:
        """比较已有记忆与新值，返回冲突信息或 None。"""

        relation_type = self.classify_value_relation(
            existing.value, new_value, existing.memory_type, existing.key
        )
        if relation_type == "same":
            return None

        if relation_type in ("extension", "refinement"):
            return ConflictInfo(
                existing_id=existing.id,
                existing_value=existing.value,
                existing_confidence=existing.confidence,
                existing_updated_at=existing.updated_at,
                new_value=new_value,
                new_confidence=new_confidence,
                conflict_type=relation_type,
                resolution=ConflictResolution.UPDATE,
            )

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
                conflict_type=(
                    "value_mismatch"
                    if relation_type in ("unknown", "conflict")
                    else relation_type
                ),
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
                conflict_type=(
                    "constraint_conflict"
                    if relation_type in ("unknown", "conflict")
                    else relation_type
                ),
                resolution=ConflictResolution.NEEDS_CONFIRM,
            )

        return ConflictInfo(
            existing_id=existing.id,
            existing_value=existing.value,
            existing_confidence=existing.confidence,
            existing_updated_at=existing.updated_at,
            new_value=new_value,
            new_confidence=new_confidence,
            conflict_type=(
                "value_mismatch"
                if relation_type in ("unknown", "conflict")
                else relation_type
            ),
            resolution=ConflictResolution.UPDATE,
        )

    def _resolve_preference_conflict(
        self, existing: MemoryItem, new_confidence: float
    ) -> str:
        """为偏好类冲突选择覆盖、更新或人工确认策略。"""

        if existing.source_type == SourceType.EXPLICIT and new_confidence < existing.confidence:
            return ConflictResolution.NEEDS_CONFIRM
        if new_confidence > existing.confidence + 0.2:
            return ConflictResolution.OVERWRITE
        if existing.source_type in (SourceType.CONFIRMED, SourceType.MANUAL):
            return ConflictResolution.NEEDS_CONFIRM
        return ConflictResolution.UPDATE

    def _canonical(self, value: Any) -> str:
        """把任意值转换为稳定比较字符串。"""

        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return self._normalize_text(str(value))

    def _normalize_text(self, text: str) -> str:
        """归一化文本用于冲突/重复比较。"""

        return re.sub(r"\s+", "", text or "").lower()

    def _as_sequence(self, value: Any) -> Optional[List[Any]]:
        """把 list/tuple 值统一为列表，其他类型返回 None。"""

        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return None

    def _looks_like_range_extension(self, text: str) -> bool:
        """判断文本是否像多值范围扩展表达。"""

        return any(separator in text for separator in ["和", "与", "、", ",", "，", "/", "及"])

    def _shares_location_or_topic(self, old_text: str, new_text: str) -> bool:
        """判断两段文本是否共享常见观测地点或天文主题。"""

        markers = [
            "北京",
            "上海",
            "广州",
            "深圳",
            "杭州",
            "成都",
            "南京",
            "武汉",
            "承德",
            "观测",
            "深空",
            "天体",
        ]
        return any(marker in old_text and marker in new_text for marker in markers)

    def _shared_token_count(self, old_norm: str, new_norm: str) -> int:
        """统计两段归一化文本的共享 token 数。"""

        old_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", old_norm))
        new_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", new_norm))
        return len(old_tokens & new_tokens)


class CandidatePromotionEvaluator:
    """用稳定性、一致性和来源权重评估候选是否可自动转正。"""

    TYPE_THRESHOLDS = {
        MemoryType.PREFERENCE: 0.5,
        MemoryType.HABIT: 0.5,
        MemoryType.CONSTRAINT: 0.65,
        MemoryType.BACKGROUND: 0.7,
    }

    SOURCE_WEIGHTS = {
        "solid": 1.0,
        "tentative": 0.85,
        "inferred": 0.6,
    }

    def __init__(
        self,
        conflict_detector: Optional[ConflictDetector] = None,
        decay_days: float = 30.0,
    ):
        """初始化候选转正评分器和出现次数时间衰减窗口。"""

        self.conflict_detector = conflict_detector or ConflictDetector()
        self.decay_days = decay_days

    def evaluate(self, candidate: Any) -> PromotionDecision:
        """综合稳定性、一致性、来源权重和类型阈值判断候选能否转正。"""

        memory_type = candidate.memory_type
        category = candidate.category
        conflict_info = candidate.metadata.get("conflict_info") or {}
        if conflict_info.get("relation_type") in ("conflict", "unknown"):
            return self._blocked(candidate, "active_memory_conflict")
        if memory_type == MemoryType.FACT:
            return self._blocked(candidate, "fact_never_auto_promote")
        if memory_type == MemoryType.BACKGROUND and category not in AUTO_PROMOTABLE_BACKGROUND_CATEGORIES:
            return self._blocked(candidate, "background_category_needs_confirm")

        threshold = self.TYPE_THRESHOLDS.get(memory_type)
        if threshold is None:
            return self._blocked(candidate, "memory_type_not_auto_promotable")

        history = self.normalized_occurrence_history(candidate)
        effective_count = self.effective_count(history)
        stability = self.stability(candidate, history, effective_count)
        consistency = self.consistency(candidate, history)
        grade = self._grade(candidate, history)
        source_weight = self.SOURCE_WEIGHTS.get(grade, 0.85)
        promote_score = round(
            candidate.confidence * stability * consistency * source_weight,
            3,
        )

        should_promote = promote_score >= threshold
        reason = "score_meets_threshold" if should_promote else "score_below_threshold"
        gate_reason = str(candidate.metadata.get("gate_reason") or "")
        if (
            should_promote
            and candidate.source_type != SourceType.EXPLICIT
            and effective_count < 1.5
            and "stable_profile_signal" not in gate_reason
        ):
            should_promote = False
            reason = "single_auto_observation_needs_repetition"
        if memory_type == MemoryType.BACKGROUND and consistency < 0.9:
            should_promote = False
            reason = "background_consistency_below_threshold"
        if consistency <= 0.0:
            should_promote = False
            reason = "value_conflict"

        return PromotionDecision(
            should_promote=should_promote,
            promote_score=promote_score,
            threshold=threshold,
            stability=round(stability, 3),
            consistency=round(consistency, 3),
            effective_count=round(effective_count, 3),
            source_weight=source_weight,
            reason=reason,
        )

    def normalized_occurrence_history(self, candidate: Any) -> List[Dict[str, Any]]:
        """读取候选出现历史；缺失时用 occurrence_count 合成兼容历史。"""

        history = candidate.metadata.get("occurrence_history") or []
        if isinstance(history, list) and history:
            return [item for item in history if isinstance(item, dict)]

        synthesized = []
        count = max(1, int(getattr(candidate, "occurrence_count", 1) or 1))
        if count == 1:
            synthesized.append(self._history_item(candidate, candidate.last_seen_at))
        else:
            synthesized.append(self._history_item(candidate, candidate.first_seen_at))
            for _ in range(max(0, count - 2)):
                synthesized.append(self._history_item(candidate, candidate.last_seen_at))
            synthesized.append(self._history_item(candidate, candidate.last_seen_at))
        return synthesized

    def effective_count(self, history: List[Dict[str, Any]]) -> float:
        """按 30 天指数衰减计算有效出现次数。"""

        now = datetime.now()
        total = 0.0
        for item in history:
            seen_at = self._parse_dt(item.get("seen_at")) or now
            age_days = max((now - seen_at).total_seconds() / 86400.0, 0.0)
            total += math.exp(-age_days / self.decay_days)
        return total

    def stability(
        self,
        candidate: Any,
        history: List[Dict[str, Any]],
        effective_count: float,
    ) -> float:
        """根据抽取等级、出现次数和跨会话/跨天分布计算稳定性。"""

        grade = self._grade(candidate, history)
        gate_reason = str(candidate.metadata.get("gate_reason") or "")
        if grade == "solid" and "stable_profile_signal" in gate_reason:
            base = 0.75
        elif grade in ("tentative", "inferred") and len(history) <= 1:
            base = 0.4
        else:
            base = 0.55

        spread_factor = 1.0 if self._has_spread(history) else 0.0
        dynamic = min(
            1.0,
            0.45 + 0.35 * min(effective_count / 2.0, 1.0) + 0.20 * spread_factor,
        )
        return max(base, dynamic)

    def consistency(self, candidate: Any, history: List[Dict[str, Any]]) -> float:
        """根据历史值与候选值的关系计算一致性分数。"""

        if len(history) <= 1:
            return 1.0
        scores = []
        for item in history:
            relation = self.conflict_detector.classify_value_relation(
                candidate.value,
                item.get("value"),
                candidate.memory_type,
                candidate.key,
            )
            scores.append(
                {
                    "same": 1.0,
                    "extension": 0.95,
                    "refinement": 0.95,
                    "unknown": 0.7,
                    "conflict": 0.0,
                }.get(relation, 0.7)
            )
        return min(scores) if scores else 1.0

    def _blocked(self, candidate: Any, reason: str) -> PromotionDecision:
        """生成不可转正的评分结果，并保留可解释原因。"""

        history = self.normalized_occurrence_history(candidate)
        return PromotionDecision(
            should_promote=False,
            promote_score=0.0,
            threshold=None,
            stability=0.0,
            consistency=self.consistency(candidate, history),
            effective_count=round(self.effective_count(history), 3),
            source_weight=0.0,
            reason=reason,
        )

    def _history_item(self, candidate: Any, seen_at: str) -> Dict[str, Any]:
        """把候选当前值转换成一条出现历史记录。"""

        return {
            "value": candidate.value,
            "confidence": candidate.confidence,
            "source_type": candidate.source_type,
            "source_conversation_id": candidate.source_conversation_id,
            "seen_at": seen_at or _utcnow_iso(),
            "extraction_grade": candidate.metadata.get("extraction_grade"),
            "gate_reason": candidate.metadata.get("gate_reason"),
        }

    def _grade(self, candidate: Any, history: List[Dict[str, Any]]) -> str:
        """解析候选抽取等级，缺失时根据来源回退。"""

        grade = candidate.metadata.get("extraction_grade")
        if not grade and history:
            grade = history[-1].get("extraction_grade")
        if not grade and candidate.source_type == SourceType.EXPLICIT:
            return "solid"
        return str(grade or "tentative")

    def _has_spread(self, history: List[Dict[str, Any]]) -> bool:
        """判断候选出现是否跨会话或跨日期分布。"""

        conversation_ids = {
            item.get("source_conversation_id")
            for item in history
            if item.get("source_conversation_id")
        }
        if len(conversation_ids) > 1:
            return True
        dates = set()
        for item in history:
            parsed = self._parse_dt(item.get("seen_at"))
            if parsed:
                dates.add(parsed.date().isoformat())
        return len(dates) > 1

    def _parse_dt(self, value: Any) -> Optional[datetime]:
        """安全解析出现时间字符串。"""

        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None


class Deduplicator:
    """基于简单相似度规则检测和合并重复记忆。"""

    def __init__(self, similarity_threshold: float = 0.85):
        """初始化重复判定相似度阈值。"""

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
        """计算两个字符串的 Levenshtein 编辑距离。"""

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
        """判断两个值是否达到重复阈值。"""

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
        """初始化各类长期记忆的过期和未使用归档周期。"""

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
        """判断记忆是否超过 expires_at。"""

        if not item.expires_at:
            return False
        try:
            return item.expires_at < _utcnow_iso()
        except Exception:
            return False

    def should_archive(self, item: MemoryItem) -> bool:
        """判断 active 记忆是否因长期未访问应归档。"""

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
        """返回用户否定等场景下的失效状态。"""

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
        """组合置信度、冲突、去重、过期和候选转正策略对象。"""

        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.conflict_detector = conflict_detector or ConflictDetector()
        self.deduplicator = deduplicator or Deduplicator(similarity_threshold=dedup_similarity_threshold)
        self.expiry_manager = expiry_manager or ExpiryManager()
        self.promotion_evaluator = CandidatePromotionEvaluator(self.conflict_detector)
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
        """查找与新值重复的已有记忆。"""

        return self.deduplicator.find_duplicates(new_value, existing_items)

    def merge_duplicate(
        self, existing: MemoryItem, new_value: Any, new_confidence: float
    ) -> Tuple[Any, float]:
        """合并重复记忆值，并保留较高置信度。"""

        merged_value = self.deduplicator.merge_values(existing.value, new_value, existing.memory_type)
        merged_confidence = max(existing.confidence, new_confidence)
        return merged_value, merged_confidence

    def compute_expiry(self, memory_type: str, source_type: str) -> Optional[str]:
        """根据记忆类型和来源计算过期时间。"""

        return self.expiry_manager.compute_expiry_date(memory_type, source_type)

    def check_expiry(self, item: MemoryItem) -> bool:
        """判断指定记忆是否已过期。"""

        return self.expiry_manager.is_expired(item)

    def should_archive(self, item: MemoryItem) -> bool:
        """判断指定记忆是否应归档。"""

        return self.expiry_manager.should_archive(item)

    def evaluate_candidate_promotion(self, candidate: Any) -> PromotionDecision:
        """对候选记忆运行自动转正评分。"""

        return self.promotion_evaluator.evaluate(candidate)

    def apply_negative_feedback(
        self, item: MemoryItem, reason: str = "user_denied"
    ) -> MemoryItem:
        """用户否定注入记忆时降低置信度，二次否定归档。"""

        metadata = dict(item.metadata or {})
        denial_count = int(metadata.get("denial_count", 0) or 0) + 1
        metadata.update(
            {
                "denial_count": denial_count,
                "last_denial_reason": reason,
                "last_denied_at": _utcnow_iso(),
            }
        )
        item.confidence = max(round(item.confidence - 0.3, 3), 0.1)
        if denial_count >= 2:
            item.status = MemoryStatus.ARCHIVED
            metadata["probation_status"] = "denied"
        item.metadata = metadata
        item.updated_at = _utcnow_iso()
        return item
