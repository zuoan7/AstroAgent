import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    HABIT = "habit"
    CONSTRAINT = "constraint"
    BACKGROUND = "background"
    FACT = "fact"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    CANDIDATE = "candidate"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class SourceType(str, Enum):
    AUTO = "auto"
    EXPLICIT = "explicit"
    MANUAL = "manual"
    CONFIRMED = "confirmed"


class ConfirmationStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ConflictResolution(str, Enum):
    UPDATE = "update"
    OVERWRITE = "overwrite"
    KEEP_OLD = "keep_old"
    NEEDS_CONFIRM = "needs_confirm"


class EventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    ACCESSED = "accessed"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    RESTORED = "restored"
    CANDIDATE_CREATED = "candidate_created"
    CANDIDATE_PROMOTED = "candidate_promoted"
    CANDIDATE_REJECTED = "candidate_rejected"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    CONFIRMATION_REQUESTED = "confirmation_requested"
    CONFIRMATION_RESOLVED = "confirmation_resolved"
    MERGED = "merged"
    DEDUPLICATED = "deduplicated"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"


PREFERENCE_CATEGORIES = {
    "response_style": {"type": "string", "values": ["正式", "口语化", "专业", "简洁", "适中"]},
    "explanation_depth": {"type": "string", "values": ["入门", "进阶", "专家", "适中"]},
    "output_format": {"type": "string", "values": ["列表", "段落", "代码块", "表格", "混合"]},
    "knowledge_level": {"type": "string", "values": ["专业", "通俗", "适中"]},
    "language": {"type": "string", "values": ["中文", "英文", "其他"]},
    "observation_experience": {"type": "string", "values": ["初学者", "中级", "高级"]},
}

HABIT_CATEGORIES = {
    "frequent_topics": {"type": "list", "description": "常问主题分类"},
    "preferred_time": {"type": "string", "values": ["白天", "夜晚", "凌晨", "全天"], "description": "活跃时间段"},
    "observation_type": {"type": "string", "values": ["目视", "摄影", "深空", "行星", "其他"], "description": "观测类型"},
    "usage_scenario": {"type": "string", "values": ["学习", "观测规划", "摄影指导", "闲聊", "其他"], "description": "使用场景"},
    "topic_frequency": {"type": "dict", "description": "主题频率统计"},
    "active_hours": {"type": "dict", "description": "活跃时段统计"},
}

CONSTRAINT_CATEGORIES = {
    "content_taboo": {"type": "string", "description": "内容禁忌"},
    "output_length_limit": {"type": "string", "description": "输出长度限制"},
    "no_jargon": {"type": "bool", "description": "避免专业术语"},
    "custom": {"type": "string", "description": "自定义约束"},
}

BACKGROUND_CATEGORIES = {
    "skill_level": {"type": "string", "values": ["入门", "进阶", "专家"], "description": "能力水平"},
    "device_info": {"type": "string", "description": "使用设备信息"},
    "domain_experience": {"type": "list", "description": "领域经验背景"},
    "education": {"type": "string", "description": "教育背景"},
    "location": {"type": "string", "description": "常用观测位置"},
}

FACT_CATEGORIES = {
    "basic_info": {"type": "string", "description": "用户基本信息"},
    "fixed_preference": {"type": "string", "description": "固定偏好"},
    "equipment": {"type": "string", "description": "观测设备"},
    "location_info": {"type": "string", "description": "位置信息"},
}


CATEGORY_REGISTRY: Dict[str, Dict[str, Any]] = {}
for _name, _spec in PREFERENCE_CATEGORIES.items():
    CATEGORY_REGISTRY[f"preference.{_name}"] = _spec
for _name, _spec in HABIT_CATEGORIES.items():
    CATEGORY_REGISTRY[f"habit.{_name}"] = _spec
for _name, _spec in CONSTRAINT_CATEGORIES.items():
    CATEGORY_REGISTRY[f"constraint.{_name}"] = _spec
for _name, _spec in BACKGROUND_CATEGORIES.items():
    CATEGORY_REGISTRY[f"background.{_name}"] = _spec
for _name, _spec in FACT_CATEGORIES.items():
    CATEGORY_REGISTRY[f"fact.{_name}"] = _spec


def _utcnow_iso() -> str:
    return datetime.now().isoformat()


def _generate_id() -> str:
    return uuid.uuid4().hex


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Optional[str], default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


@dataclass
class MemoryItem:
    id: str
    user_id: str
    memory_type: str
    category: str
    key: str
    value: Any
    confidence: float = 0.5
    source_type: str = SourceType.AUTO
    source_conversation_id: Optional[str] = None
    source_content_snippet: Optional[str] = None
    status: str = MemoryStatus.ACTIVE
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    accessed_at: Optional[str] = None
    expires_at: Optional[str] = None
    access_count: int = 0
    confirmation_count: int = 0
    confirmed_by_user: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = _generate_id()
        if not self.created_at:
            self.created_at = _utcnow_iso()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "source_conversation_id": self.source_conversation_id,
            "source_content_snippet": self.source_content_snippet,
            "status": self.status,
            "priority": self.priority,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "confirmation_count": self.confirmation_count,
            "confirmed_by_user": self.confirmed_by_user,
        }

    def to_db_row(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "category": self.category,
            "key": self.key,
            "value": _json_dumps(self.value),
            "confidence": self.confidence,
            "source_type": self.source_type,
            "source_conversation_id": self.source_conversation_id,
            "source_content_snippet": self.source_content_snippet,
            "status": self.status,
            "priority": self.priority,
            "metadata": _json_dumps(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "confirmation_count": self.confirmation_count,
            "confirmed_by_user": int(self.confirmed_by_user),
        }

    @classmethod
    def from_db_row(cls, row: Any) -> "MemoryItem":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            memory_type=row["memory_type"],
            category=row["category"],
            key=row["key"],
            value=_json_loads(row["value"]),
            confidence=row["confidence"],
            source_type=row["source_type"],
            source_conversation_id=row["source_conversation_id"],
            source_content_snippet=row["source_content_snippet"],
            status=row["status"],
            priority=row["priority"],
            metadata=_json_loads(row["metadata"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            accessed_at=row["accessed_at"],
            expires_at=row["expires_at"],
            access_count=row["access_count"],
            confirmation_count=row["confirmation_count"],
            confirmed_by_user=bool(row["confirmed_by_user"]),
        )

    @classmethod
    def create(
        cls,
        user_id: str,
        memory_type: str,
        category: str,
        key: str,
        value: Any,
        confidence: float = 0.5,
        source_type: str = SourceType.AUTO,
        source_conversation_id: Optional[str] = None,
        source_content_snippet: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[str] = None,
    ) -> "MemoryItem":
        return cls(
            id=_generate_id(),
            user_id=user_id,
            memory_type=memory_type,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            source_type=source_type,
            source_conversation_id=source_conversation_id,
            source_content_snippet=source_content_snippet,
            status=MemoryStatus.ACTIVE,
            priority=priority,
            metadata=metadata or {},
            expires_at=expires_at,
        )


@dataclass
class MemoryVersion:
    id: int = 0
    memory_id: str = ""
    version: int = 1
    value: Any = None
    confidence: float = 0.5
    change_reason: str = ""
    changed_at: str = ""

    def __post_init__(self):
        if not self.changed_at:
            self.changed_at = _utcnow_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "version": self.version,
            "value": self.value,
            "confidence": self.confidence,
            "change_reason": self.change_reason,
            "changed_at": self.changed_at,
        }


@dataclass
class MemoryCandidate:
    id: str = ""
    user_id: str = ""
    memory_type: str = ""
    category: str = ""
    key: str = ""
    value: Any = None
    confidence: float = 0.3
    source_type: str = SourceType.AUTO
    source_conversation_id: Optional[str] = None
    source_content_snippet: Optional[str] = None
    occurrence_count: int = 1
    first_seen_at: str = ""
    last_seen_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = _generate_id()
        if not self.first_seen_at:
            self.first_seen_at = _utcnow_iso()
        if not self.last_seen_at:
            self.last_seen_at = self.first_seen_at
        if not self.created_at:
            self.created_at = _utcnow_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "source_conversation_id": self.source_conversation_id,
            "source_content_snippet": self.source_content_snippet,
            "occurrence_count": self.occurrence_count,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    def to_db_row(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "category": self.category,
            "key": self.key,
            "value": _json_dumps(self.value),
            "confidence": self.confidence,
            "source_type": self.source_type,
            "source_conversation_id": self.source_conversation_id,
            "source_content_snippet": self.source_content_snippet,
            "occurrence_count": self.occurrence_count,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "metadata": _json_dumps(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_db_row(cls, row: Any) -> "MemoryCandidate":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            memory_type=row["memory_type"],
            category=row["category"],
            key=row["key"],
            value=_json_loads(row["value"]),
            confidence=row["confidence"],
            source_type=row["source_type"],
            source_conversation_id=row["source_conversation_id"],
            source_content_snippet=row["source_content_snippet"],
            occurrence_count=row["occurrence_count"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            metadata=_json_loads(row["metadata"], {}),
            created_at=row["created_at"],
        )

    def to_memory_item(self) -> MemoryItem:
        return MemoryItem.create(
            user_id=self.user_id,
            memory_type=self.memory_type,
            category=self.category,
            key=self.key,
            value=self.value,
            confidence=min(self.confidence + 0.2, 1.0),
            source_type=self.source_type,
            source_conversation_id=self.source_conversation_id,
            source_content_snippet=self.source_content_snippet,
            metadata={**self.metadata, "promoted_from_candidate": self.id, "occurrence_count": self.occurrence_count},
        )


@dataclass
class MemoryConfirmation:
    id: str = ""
    user_id: str = ""
    memory_id: str = ""
    confirmation_type: str = ""
    content: str = ""
    status: str = ConfirmationStatus.PENDING
    created_at: str = ""
    resolved_at: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = _generate_id()
        if not self.created_at:
            self.created_at = _utcnow_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_id": self.memory_id,
            "confirmation_type": self.confirmation_type,
            "content": self.content,
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

    def to_db_row(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_id": self.memory_id,
            "confirmation_type": self.confirmation_type,
            "content": self.content,
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_db_row(cls, row: Any) -> "MemoryConfirmation":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            memory_id=row["memory_id"],
            confirmation_type=row["confirmation_type"],
            content=row["content"],
            status=row["status"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )


@dataclass
class EventLogEntry:
    id: int = 0
    user_id: str = ""
    memory_id: Optional[str] = None
    event_type: str = ""
    event_detail: str = ""
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = _utcnow_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_id": self.memory_id,
            "event_type": self.event_type,
            "event_detail": self.event_detail,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    def to_db_row(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "memory_id": self.memory_id,
            "event_type": self.event_type,
            "event_detail": self.event_detail,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "metadata": _json_dumps(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    habits: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    background: Dict[str, Any] = field(default_factory=dict)
    facts: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = _utcnow_iso()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "habits": self.habits,
            "constraints": self.constraints,
            "background": self.background,
            "facts": self.facts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class MemoryEvent:
    user_id: str
    event_type: str
    key: str
    value: Any
    source_text: str = ""
    confidence: float = 0.5
    status: str = "candidate"
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = ""
    last_confirmed_at: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = _generate_id()
        if not self.created_at:
            self.created_at = _utcnow_iso()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "key": self.key,
            "value": self.value,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "status": self.status,
            "last_confirmed_at": self.last_confirmed_at,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_db_row(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload["value"] = _json_dumps(self.value)
        payload["metadata"] = _json_dumps(self.metadata)
        return payload

    @classmethod
    def from_db_row(cls, row: Any) -> "MemoryEvent":
        return cls(
            event_id=row["event_id"],
            user_id=row["user_id"],
            event_type=row["event_type"],
            key=row["key"],
            value=_json_loads(row["value"]),
            source_text=row["source_text"],
            confidence=row["confidence"],
            status=row["status"],
            last_confirmed_at=row["last_confirmed_at"],
            metadata=_json_loads(row["metadata"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass
class CandidateMemory:
    user_id: str
    event_type: str
    key: str
    value: Any
    confidence: float = 0.3
    created_at: str = ""
    promoted: bool = False

    def __post_init__(self):
        if not self.created_at:
            self.created_at = _utcnow_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "event_type": self.event_type,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "promoted": self.promoted,
        }


@dataclass
class ExtractionResult:
    should_extract: bool = False
    memory_type: str = ""
    category: str = ""
    key: str = ""
    value: Any = None
    confidence: float = 0.5
    source_type: str = SourceType.AUTO
    is_explicit: bool = False
    is_temporary: bool = False
    raw_content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_extract": self.should_extract,
            "memory_type": self.memory_type,
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "is_explicit": self.is_explicit,
            "is_temporary": self.is_temporary,
            "raw_content": self.raw_content,
        }


@dataclass
class ConflictInfo:
    existing_id: str
    existing_value: Any
    existing_confidence: float
    existing_updated_at: str
    new_value: Any
    new_confidence: float
    conflict_type: str
    resolution: str = ConflictResolution.NEEDS_CONFIRM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "existing_id": self.existing_id,
            "existing_value": self.existing_value,
            "existing_confidence": self.existing_confidence,
            "existing_updated_at": self.existing_updated_at,
            "new_value": self.new_value,
            "new_confidence": self.new_confidence,
            "conflict_type": self.conflict_type,
            "resolution": self.resolution,
        }


@dataclass
class MemoryQuery:
    user_id: str
    memory_type: Optional[str] = None
    category: Optional[str] = None
    key: Optional[str] = None
    status: Optional[str] = None
    source_type: Optional[str] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    updated_after: Optional[str] = None
    keyword: Optional[str] = None
    limit: int = 50
    offset: int = 0
    order_by: str = "updated_at"
    order_desc: bool = True

    def to_where_clause(self) -> tuple:
        conditions = ["user_id = ?"]
        params: list = [self.user_id]

        if self.memory_type:
            conditions.append("memory_type = ?")
            params.append(self.memory_type)
        if self.category:
            conditions.append("category = ?")
            params.append(self.category)
        if self.key:
            conditions.append("key = ?")
            params.append(self.key)
        if self.status:
            conditions.append("status = ?")
            params.append(self.status)
        if self.source_type:
            conditions.append("source_type = ?")
            params.append(self.source_type)
        if self.min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(self.min_confidence)
        if self.max_confidence is not None:
            conditions.append("confidence <= ?")
            params.append(self.max_confidence)
        if self.created_after:
            conditions.append("created_at >= ?")
            params.append(self.created_after)
        if self.created_before:
            conditions.append("created_at <= ?")
            params.append(self.created_before)
        if self.updated_after:
            conditions.append("updated_at >= ?")
            params.append(self.updated_after)
        if self.keyword:
            conditions.append("(key LIKE ? OR value LIKE ? OR category LIKE ?)")
            kw = f"%{self.keyword}%"
            params.extend([kw, kw, kw])

        direction = "DESC" if self.order_desc else "ASC"
        order_clause = f"{self.order_by} {direction}"

        where_sql = " AND ".join(conditions)
        return where_sql, params, order_clause
