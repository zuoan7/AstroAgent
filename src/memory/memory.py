import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.config import settings
from src.core.logger import logger


EXTRACTION_SYSTEM_PROMPT = """你是一个天文领域用户画像信息提取专家。你的任务是从用户与天文助手的对话中，提取结构化的用户偏好、习惯和约束信息。

请严格按照以下JSON格式输出，不要输出任何其他内容：
{
    "should_extract": true/false,
    "preferences": {
        "response_style": "简短/详细/适中",
        "knowledge_level": "专业/通俗/适中",
        "language": "中文/英文/其他",
        "observation_experience": "初学者/中级/高级"
    },
    "habits": {
        "frequent_topics": ["话题1", "话题2"],
        "preferred_time": "白天/夜晚/凌晨",
        "observation_type": "目视/摄影/深空/行星/其他"
    },
    "constraints": ["约束1", "约束2"]
}

判断规则：
- should_extract: 仅当对话中包含可提取的用户偏好、习惯或约束信息时为true
- 如果用户明确表达了偏好（如"简短回答"、"我懂天文"等），提取对应信息
- 如果用户提到了特定天体或观测类型，记录到frequent_topics
- 如果用户表达了观测经验水平，记录到observation_experience
- 如果用户提到了观测设备或拍摄方式，记录到observation_type
- 如果用户表达了限制条件，记录到constraints
- 不要从一般性问答中推断偏好，仅提取明确表达的信息
- 如果没有可提取的信息，设置should_extract为false，其他字段留空"""


EXTRACTION_USER_TEMPLATE = """请从以下对话中提取用户画像信息：

用户消息：{user_message}
助手回复：{assistant_message}

请输出JSON格式的提取结果："""


SUMMARY_SYSTEM_PROMPT = """你是一个对话摘要专家。请将以下天文助手与用户的对话历史压缩为简洁的摘要。

要求：
1. 保留关键天文信息（天体名称、观测条件、日期时间、位置等）
2. 保留用户的目标和意图
3. 保留已确认的约束条件
4. 保留中间结论和重要发现
5. 去除重复和冗余信息
6. 摘要应简洁但信息完整

请直接输出摘要内容，不要添加额外说明："""


SUMMARY_USER_TEMPLATE = """请摘要以下对话历史：

{conversation_text}

摘要："""


ROLE_LABELS = {
    "user": "用户",
    "assistant": "助手",
    "tool": "工具",
    "system": "系统",
}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _utcnow_iso() -> str:
    return datetime.now().isoformat()


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _dedupe_keep_order(items: Sequence[Any]) -> List[Any]:
    seen = set()
    result: List[Any] = []
    for item in items:
        marker = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else item
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


@dataclass
class Message:
    role: str
    content: str
    timestamp: float
    importance: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"],
            importance=data.get("importance", 0),
            metadata=data.get("metadata", {}) or {},
        )


@dataclass
class ToolCallRecord:
    tool_name: str
    tool_input: str
    result_summary: str
    timestamp: float
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "result_summary": self.result_summary,
            "timestamp": self.timestamp,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallRecord":
        return cls(
            tool_name=data["tool_name"],
            tool_input=data["tool_input"],
            result_summary=data["result_summary"],
            timestamp=data["timestamp"],
            success=data.get("success", True),
        )


@dataclass
class SalientFact:
    fact_type: str
    content: str
    timestamp: float
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_type": self.fact_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SalientFact":
        return cls(
            fact_type=data["fact_type"],
            content=data["content"],
            timestamp=data["timestamp"],
            source=data.get("source", ""),
        )


@dataclass
class MemoryEvent:
    event_type: str
    content: str
    source: str
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, Any]
    habits: Dict[str, Any]
    constraints: List[str]
    created_at: str
    updated_at: str


@dataclass
class ShortTermMemoryConfig:
    max_size: int
    memory_window: int
    context_max_tokens: int
    summary_max_tokens: int
    summary_trigger_messages: int
    summary_trigger_tokens: int
    persistence_enabled: bool
    persistence_path: str
    high_importance_roles: set
    tool_result_max_length: int
    recent_message_limit: int = 6
    max_tool_summary_entries: int = 5
    max_salient_facts: int = 32

    @classmethod
    def from_settings(cls) -> "ShortTermMemoryConfig":
        return cls(
            max_size=settings.MEMORY_SIZE,
            memory_window=settings.MEMORY_WINDOW,
            context_max_tokens=settings.STM_CONTEXT_MAX_TOKENS,
            summary_max_tokens=settings.STM_SUMMARY_MAX_TOKENS,
            summary_trigger_messages=settings.STM_SUMMARY_TRIGGER_MESSAGES,
            summary_trigger_tokens=settings.STM_SUMMARY_TRIGGER_TOKENS,
            persistence_enabled=settings.STM_PERSISTENCE_ENABLED,
            persistence_path=settings.STM_PERSISTENCE_PATH,
            high_importance_roles=set(settings.STM_IMPORTANCE_HIGH_ROLES),
            tool_result_max_length=settings.STM_TOOL_RESULT_MAX_LENGTH,
        )


@dataclass
class LongTermMemoryConfig:
    db_path: str
    max_prompt_events: int = 5
    max_topic_candidates: int = 12

    @classmethod
    def from_settings(cls, db_path: Optional[str] = None) -> "LongTermMemoryConfig":
        return cls(db_path=db_path or settings.LONG_TERM_MEMORY_PATH)


class SQLiteRepository:
    """统一封装 SQLite 连接配置，保证多线程和落地部署时更稳。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        _ensure_parent_dir(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


class ShortTermMemoryRepository(SQLiteRepository):
    def initialize(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS stm_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_timestamp REAL NOT NULL DEFAULT 0.0,
                    trimmed_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stm_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    seq INTEGER NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES stm_sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_stm_messages_session_seq
                    ON stm_messages(session_id, seq);
                CREATE TABLE IF NOT EXISTS stm_tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_input TEXT NOT NULL DEFAULT '',
                    result_summary TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL,
                    success INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY (session_id) REFERENCES stm_sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_stm_tool_calls_session_time
                    ON stm_tool_calls(session_id, timestamp);
                CREATE TABLE IF NOT EXISTS stm_salient_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES stm_sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_stm_salient_facts_session_time
                    ON stm_salient_facts(session_id, timestamp);
                """
            )

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id, user_id, summary, summary_timestamp, trimmed_count "
                "FROM stm_sessions WHERE session_id = ?",
                (session_id,),
            )
            session_row = cursor.fetchone()
            if not session_row:
                return None

            cursor.execute(
                "SELECT role, content, timestamp, importance, metadata "
                "FROM stm_messages WHERE session_id = ? ORDER BY seq",
                (session_id,),
            )
            messages = [
                Message(
                    role=row["role"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    importance=row["importance"],
                    metadata=_json_loads(row["metadata"], {}),
                )
                for row in cursor.fetchall()
            ]

            cursor.execute(
                "SELECT tool_name, tool_input, result_summary, timestamp, success "
                "FROM stm_tool_calls WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            )
            tool_calls = [
                ToolCallRecord(
                    tool_name=row["tool_name"],
                    tool_input=row["tool_input"],
                    result_summary=row["result_summary"],
                    timestamp=row["timestamp"],
                    success=bool(row["success"]),
                )
                for row in cursor.fetchall()
            ]

            cursor.execute(
                "SELECT fact_type, content, timestamp, source "
                "FROM stm_salient_facts WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            )
            salient_facts = [
                SalientFact(
                    fact_type=row["fact_type"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    source=row["source"],
                )
                for row in cursor.fetchall()
            ]

            return {
                "user_id": session_row["user_id"],
                "summary": session_row["summary"] or "",
                "summary_timestamp": session_row["summary_timestamp"] or 0.0,
                "trimmed_count": session_row["trimmed_count"] or 0,
                "messages": messages,
                "tool_calls": tool_calls,
                "salient_facts": salient_facts,
            }

    def save_session(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        summary_timestamp: float,
        trimmed_count: int,
        messages: Sequence[Message],
        tool_calls: Sequence[ToolCallRecord],
        salient_facts: Sequence[SalientFact],
    ):
        now = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO stm_sessions (
                    session_id, user_id, summary, summary_timestamp, trimmed_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    summary=excluded.summary,
                    summary_timestamp=excluded.summary_timestamp,
                    trimmed_count=excluded.trimmed_count,
                    updated_at=excluded.updated_at
                """,
                (session_id, user_id, summary, summary_timestamp, trimmed_count, now, now),
            )

            cursor.execute("DELETE FROM stm_messages WHERE session_id = ?", (session_id,))
            cursor.executemany(
                """
                INSERT INTO stm_messages (
                    session_id, role, content, timestamp, importance, metadata, seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        msg.role,
                        msg.content,
                        msg.timestamp,
                        msg.importance,
                        _json_dumps(msg.metadata),
                        idx,
                    )
                    for idx, msg in enumerate(messages)
                ],
            )

            cursor.execute("DELETE FROM stm_tool_calls WHERE session_id = ?", (session_id,))
            cursor.executemany(
                """
                INSERT INTO stm_tool_calls (
                    session_id, tool_name, tool_input, result_summary, timestamp, success
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        call.tool_name,
                        call.tool_input,
                        call.result_summary,
                        call.timestamp,
                        int(call.success),
                    )
                    for call in tool_calls
                ],
            )

            cursor.execute("DELETE FROM stm_salient_facts WHERE session_id = ?", (session_id,))
            cursor.executemany(
                """
                INSERT INTO stm_salient_facts (
                    session_id, fact_type, content, timestamp, source
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (session_id, fact.fact_type, fact.content, fact.timestamp, fact.source)
                    for fact in salient_facts
                ],
            )

    def delete_session(self, session_id: str):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM stm_messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM stm_tool_calls WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM stm_salient_facts WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM stm_sessions WHERE session_id = ?", (session_id,))

    def load_session_user(self, session_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id FROM stm_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["user_id"] if row else None


class LongTermMemoryRepository(SQLiteRepository):
    def initialize(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL,
                    habits TEXT NOT NULL,
                    constraints TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    timestamp REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_events_user_time
                    ON memory_events(user_id, timestamp DESC);
                """
            )

    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_id, preferences, habits, constraints, created_at, updated_at
                FROM user_profiles WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

        if not row:
            return None
        return UserProfile(
            user_id=row["user_id"],
            preferences=_json_loads(row["preferences"], {}),
            habits=_json_loads(row["habits"], {}),
            constraints=_json_loads(row["constraints"], []),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save_profile(self, profile: UserProfile):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id, preferences, habits, constraints, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferences=excluded.preferences,
                    habits=excluded.habits,
                    constraints=excluded.constraints,
                    updated_at=excluded.updated_at
                """,
                (
                    profile.user_id,
                    _json_dumps(profile.preferences),
                    _json_dumps(profile.habits),
                    _json_dumps(profile.constraints),
                    profile.created_at,
                    profile.updated_at,
                ),
            )

    def add_event(self, user_id: str, event: MemoryEvent):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_events (
                    user_id, event_type, content, source, confidence, metadata, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    event.event_type,
                    event.content,
                    event.source,
                    event.confidence,
                    _json_dumps(event.metadata),
                    event.timestamp,
                    _utcnow_iso(),
                ),
            )

    def get_recent_events(self, user_id: str, limit: int = 10) -> List[MemoryEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_type, content, source, confidence, metadata, timestamp
                FROM memory_events
                WHERE user_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        return [
            MemoryEvent(
                event_type=row["event_type"],
                content=row["content"],
                source=row["source"],
                confidence=row["confidence"],
                metadata=_json_loads(row["metadata"], {}),
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def delete_profile(self, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memory_events WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            return cursor.rowcount > 0


class ShortTermMemory:
    """会话内工作记忆，负责上下文裁剪、摘要和关键事实抽取。"""

    def __init__(self, session_id: Optional[str] = None, user_id: Optional[str] = None):
        self.config = ShortTermMemoryConfig.from_settings()
        self.session_id = session_id or f"session_{int(time.time())}"
        self.user_id = user_id or settings.DEFAULT_USER_ID
        self.messages: List[Message] = []
        self.tool_calls: List[ToolCallRecord] = []
        self.salient_facts: List[SalientFact] = []
        self.summary = ""
        self.summary_timestamp = 0.0
        self.trimmed_count = 0
        self.last_trimmed_content: List[Dict[str, Any]] = []
        self._restored_from_db = False
        self._repository: Optional[ShortTermMemoryRepository] = None

        if self.config.persistence_enabled:
            self._repository = ShortTermMemoryRepository(self.config.persistence_path)
            self._repository.initialize()
            self._try_restore_session()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)

    def _persist_session(self):
        if not self._repository:
            return
        self._repository.save_session(
            session_id=self.session_id,
            user_id=self.user_id,
            summary=self.summary,
            summary_timestamp=self.summary_timestamp,
            trimmed_count=self.trimmed_count,
            messages=self.messages,
            tool_calls=self.tool_calls,
            salient_facts=self.salient_facts,
        )

    def _try_restore_session(self):
        if self._restored_from_db or not self._repository:
            return
        payload = self._repository.load_session(self.session_id)
        if not payload:
            return
        self.user_id = payload["user_id"]
        self.summary = payload["summary"]
        self.summary_timestamp = payload["summary_timestamp"]
        self.trimmed_count = payload["trimmed_count"]
        self.messages = payload["messages"]
        self.tool_calls = payload["tool_calls"]
        self.salient_facts = payload["salient_facts"]
        self._restored_from_db = True

    def _delete_session_from_db(self):
        if self._repository:
            self._repository.delete_session(self.session_id)

    def _calculate_importance(self, role: str, content: str) -> int:
        importance = 1 if role in self.config.high_importance_roles else 0
        high_value_keywords = [
            "目标", "任务", "要求", "约束", "必须", "一定要",
            "不要", "避免", "注意", "确认", "决定", "计划",
            "今晚", "时间", "地点", "观测", "拍摄",
        ]
        if any(keyword in content for keyword in high_value_keywords):
            importance += 1
        if role == "user" and ("？" in content or "?" in content):
            importance += 1
        if role == "tool" and ("error" in content.lower() or "错误" in content):
            importance += 1
        return min(importance, 3)

    def add_message(
        self,
        role: str,
        content: str,
        timestamp: Optional[float] = None,
        importance: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        timestamp = timestamp or time.time()
        msg_importance = importance if importance is not None else self._calculate_importance(role, content)
        self.messages.append(
            Message(
                role=role,
                content=content,
                timestamp=timestamp,
                importance=msg_importance,
                metadata=metadata or {},
            )
        )
        self._trim_messages_if_needed()
        self._check_and_trigger_summary()
        self._persist_session()

    def add_tool_call(
        self,
        tool_name: str,
        tool_input: str,
        result: str,
        timestamp: Optional[float] = None,
        success: bool = True,
    ):
        timestamp = timestamp or time.time()
        result_summary = self._summarize_tool_result(result)
        self.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                tool_input=(tool_input or "")[:200],
                result_summary=result_summary,
                timestamp=timestamp,
                success=success,
            )
        )
        self.add_message(
            role="tool",
            content=f"[{tool_name}] {result_summary}",
            timestamp=timestamp,
            importance=2 if success else 3,
            metadata={"tool_name": tool_name, "success": success},
        )

    def add_salient_fact(
        self,
        fact_type: str,
        content: str,
        source: str = "",
        timestamp: Optional[float] = None,
    ):
        timestamp = timestamp or time.time()
        signature = (fact_type, content)
        if any((item.fact_type, item.content) == signature for item in self.salient_facts):
            return
        self.salient_facts.append(
            SalientFact(fact_type=fact_type, content=content, source=source, timestamp=timestamp)
        )
        # 限制事实数量，避免短期记忆自己膨胀。
        self.salient_facts = sorted(
            self.salient_facts,
            key=lambda item: (item.timestamp, item.fact_type, item.content),
        )[-self.config.max_salient_facts :]
        self._persist_session()

    def _summarize_tool_result(self, result: str) -> str:
        if not result:
            return ""
        max_len = self.config.tool_result_max_length
        try:
            payload = json.loads(result)
            if isinstance(payload, dict):
                if "error" in payload:
                    return f"错误: {payload['error']}"
                if "answer" in payload:
                    return str(payload["answer"])[:max_len]
                key_items = []
                for key in list(payload.keys())[:5]:
                    value = str(payload[key])
                    if len(value) > 100:
                        value = value[:100] + "..."
                    key_items.append(f"{key}: {value}")
                return "; ".join(key_items)[:max_len]
        except (TypeError, json.JSONDecodeError):
            pass
        return result if len(result) <= max_len else result[:max_len] + "..."

    def _trim_messages_if_needed(self):
        overflow = len(self.messages) - self.config.max_size
        if overflow <= 0:
            return
        trimmed = self.messages[:overflow]
        self.last_trimmed_content = [item.to_dict() for item in trimmed]
        self.trimmed_count += len(trimmed)
        self.messages = self.messages[overflow:]
        for message in trimmed:
            if message.importance >= 2:
                self.add_salient_fact(
                    fact_type="important_message",
                    content=f"[{message.role}] {message.content[:200]}",
                    source="context_trimming",
                    timestamp=message.timestamp,
                )

    def _estimate_total_tokens(self) -> int:
        total = sum(self._estimate_tokens(message.content) for message in self.messages)
        total += self._estimate_tokens(self.summary)
        return total

    def _check_and_trigger_summary(self):
        if len(self.messages) <= 3:
            return
        should_summarize = (
            len(self.messages) >= self.config.summary_trigger_messages
            or self._estimate_total_tokens() >= self.config.summary_trigger_tokens
        )
        if should_summarize:
            self._generate_summary()

    def _generate_summary(self):
        history = self.messages[:-3]
        if not history:
            return
        conversation_text = self._format_messages_for_summary(history)
        new_summary = self._do_summarize(conversation_text)
        if not new_summary:
            return
        self.summary = f"{self.summary}\n\n[更新摘要] {new_summary}".strip() if self.summary else new_summary
        if self._estimate_tokens(self.summary) > self.config.summary_max_tokens * 2:
            self.summary = self._do_summarize(self.summary) or self.summary
        self.summary_timestamp = time.time()
        for message in history:
            if message.importance >= 2:
                self.add_salient_fact(
                    fact_type="from_summary",
                    content=f"[{message.role}] {message.content[:200]}",
                    source="summary_generation",
                    timestamp=message.timestamp,
                )
        self.messages = self.messages[-3:]

    def _do_summarize(self, text: str) -> str:
        if not text or len(text) < 50:
            return ""
        if settings.DASHSCOPE_API_KEY:
            try:
                return self._summarize_with_llm(text)
            except Exception as exc:
                logger.warning(f"LLM摘要生成失败，回退到截断摘要: {exc}")
        return self._fallback_truncate_summary(text)

    def _summarize_with_llm(self, text: str) -> str:
        from langchain_community.chat_models import ChatTongyi
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatTongyi(
            model=settings.MODEL_NAME,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            temperature=0.0,
            request_timeout=15,
        )
        response = llm.invoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=SUMMARY_USER_TEMPLATE.format(conversation_text=text[:3000])),
            ]
        )
        return response.content.strip()

    def _fallback_truncate_summary(self, text: str) -> str:
        max_chars = self.config.summary_max_tokens * 2
        if len(text) <= max_chars:
            return text
        lines = text.split("\n")
        result_lines: List[str] = []
        current_len = 0
        for line in lines:
            if current_len + len(line) + 1 > max_chars:
                break
            result_lines.append(line)
            current_len += len(line) + 1
        return "\n".join(result_lines) + "\n...(已截断)"

    def _format_messages_for_summary(self, messages: Sequence[Message]) -> str:
        return "\n".join(f"{ROLE_LABELS.get(msg.role, msg.role)}: {msg.content}" for msg in messages)

    def get_recent_messages(self, window: Optional[int] = None) -> List[Dict[str, str]]:
        window = window or self.config.memory_window
        return [{"role": msg.role, "content": msg.content} for msg in self.messages[-window:]]

    def _select_recent_messages_for_context(self, token_budget: int) -> List[Message]:
        if not self.messages:
            return []
        always_keep = self.messages[-self.config.recent_message_limit :]
        chosen = list(always_keep)
        used_tokens = sum(self._estimate_tokens(msg.content) for msg in chosen)

        candidates = sorted(
            self.messages[:-self.config.recent_message_limit],
            key=lambda msg: (msg.importance, msg.timestamp),
            reverse=True,
        )
        for candidate in candidates:
            if candidate in chosen:
                continue
            msg_tokens = self._estimate_tokens(candidate.content)
            if used_tokens + msg_tokens > token_budget:
                continue
            chosen.append(candidate)
            used_tokens += msg_tokens

        return sorted(chosen, key=lambda msg: msg.timestamp)

    def _build_recent_context(self, token_budget: int) -> Tuple[str, List[Message]]:
        selected = self._select_recent_messages_for_context(token_budget)
        if not selected:
            return "无最近对话", []
        lines = [f"{ROLE_LABELS.get(msg.role, msg.role)}: {msg.content}" for msg in selected]
        return "\n".join(lines), selected

    def _format_salient_facts(self) -> str:
        if not self.salient_facts:
            return ""
        grouped: Dict[str, List[str]] = {}
        for fact in self.salient_facts:
            grouped.setdefault(fact.fact_type, []).append(fact.content)
        labels = {
            "user_goal": "用户目标",
            "constraint": "约束条件",
            "intermediate_conclusion": "中间结论",
            "important_message": "重要消息",
            "from_summary": "历史要点",
        }
        blocks = []
        for fact_type, contents in grouped.items():
            blocks.append(f"【{labels.get(fact_type, fact_type)}】")
            for content in contents:
                blocks.append(f"  - {content}")
        return "\n".join(blocks)

    def _format_tool_calls_summary(self) -> str:
        if not self.tool_calls:
            return ""
        recent_calls = self.tool_calls[-self.config.max_tool_summary_entries :]
        return "\n".join(
            f"[{'✓' if call.success else '✗'}] {call.tool_name}: {call.result_summary[:100]}"
            for call in recent_calls
        )

    def _assemble_context_text(self, context_parts: Dict[str, str]) -> str:
        sections: List[str] = []
        if context_parts.get("key_facts"):
            sections.append(f"=== 关键事实 ===\n{context_parts['key_facts']}")
        if context_parts.get("history_summary"):
            sections.append(f"=== 历史摘要 ===\n{context_parts['history_summary']}")
        if context_parts.get("recent_dialog"):
            sections.append(f"=== 最近对话 ===\n{context_parts['recent_dialog']}")
        if context_parts.get("tool_summary"):
            sections.append(f"=== 工具调用摘要 ===\n{context_parts['tool_summary']}")
        return "\n\n".join(sections) if sections else "无对话上下文"

    def build_context(self, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        max_tokens = max_tokens or self.config.context_max_tokens
        token_budget = max_tokens
        context_parts: Dict[str, str] = {}

        facts_text = self._format_salient_facts()
        facts_tokens = self._estimate_tokens(facts_text)
        if facts_text and facts_tokens <= int(max_tokens * 0.25):
            context_parts["key_facts"] = facts_text
            token_budget -= facts_tokens

        summary_tokens = self._estimate_tokens(self.summary)
        if self.summary and summary_tokens <= int(max_tokens * 0.35):
            context_parts["history_summary"] = self.summary
            token_budget -= summary_tokens

        recent_text, recent_messages = self._build_recent_context(max(token_budget, int(max_tokens * 0.3)))
        context_parts["recent_dialog"] = recent_text

        tool_summary = self._format_tool_calls_summary()
        if tool_summary and self._estimate_tokens(tool_summary) <= int(max_tokens * 0.15):
            context_parts["tool_summary"] = tool_summary

        context_text = self._assemble_context_text(context_parts)
        return {
            "context_text": context_text,
            "key_facts": facts_text,
            "history_summary": self.summary,
            "recent_dialog": recent_text,
            "tool_summary": tool_summary,
            "total_tokens": self._estimate_tokens(context_text),
            "message_count": len(self.messages),
            "selected_recent_messages": [msg.to_dict() for msg in recent_messages],
            "summary_tokens": summary_tokens,
            "facts_count": len(self.salient_facts),
        }

    def clear(self):
        self.messages.clear()
        self.tool_calls.clear()
        self.salient_facts.clear()
        self.summary = ""
        self.summary_timestamp = 0.0
        self.trimmed_count = 0
        self.last_trimmed_content.clear()
        self._delete_session_from_db()

    def get_size(self) -> int:
        return len(self.messages)

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message_count": len(self.messages),
            "tool_call_count": len(self.tool_calls),
            "salient_fact_count": len(self.salient_facts),
            "summary_length": len(self.summary),
            "summary_tokens": self._estimate_tokens(self.summary),
            "total_tokens": self._estimate_total_tokens(),
            "trimmed_count": self.trimmed_count,
            "persistence_enabled": self.config.persistence_enabled,
            "config": {
                "max_size": self.config.max_size,
                "context_max_tokens": self.config.context_max_tokens,
                "summary_max_tokens": self.config.summary_max_tokens,
                "summary_trigger_messages": self.config.summary_trigger_messages,
                "summary_trigger_tokens": self.config.summary_trigger_tokens,
            },
        }

    def get_context_debug_info(self) -> Dict[str, Any]:
        context = self.build_context()
        return {
            "context_text_preview": context["context_text"][:500],
            "context_total_tokens": context["total_tokens"],
            "key_facts_preview": context.get("key_facts", "")[:300],
            "history_summary_preview": context.get("history_summary", "")[:300],
            "recent_dialog_preview": context.get("recent_dialog", "")[:300],
            "last_trimmed_content": self.last_trimmed_content[:5],
            "trimmed_count": self.trimmed_count,
        }

    def get_all_messages(self) -> List[Dict[str, Any]]:
        return [msg.to_dict() for msg in self.messages]

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.tool_calls]

    def get_salient_facts(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.salient_facts]

    def get_summary(self) -> str:
        return self.summary

    @classmethod
    def restore_session(cls, session_id: str) -> Optional["ShortTermMemory"]:
        config = ShortTermMemoryConfig.from_settings()
        if not config.persistence_enabled:
            return None
        repository = ShortTermMemoryRepository(config.persistence_path)
        repository.initialize()
        user_id = repository.load_session_user(session_id)
        if not user_id:
            return None
        return cls(session_id=session_id, user_id=user_id)


class LongTermMemory:
    """跨会话用户记忆，维护画像快照和事件流水。"""

    EXTRACTION_KEYWORDS = [
        "简短", "详细", "专业", "通俗", "易懂", "不要", "喜欢", "偏好",
        "习惯", "经常", "总是", "希望", "要求", "建议", "初学者", "入门",
        "高级", "进阶", "望远镜", "相机", "拍摄", "观测", "深空", "行星",
        "月相", "流星雨", "日食", "月食", "星系", "星云", "星团",
    ]
    TOPIC_KEYWORDS = [
        "火星", "木星", "土星", "金星", "月球", "太阳", "黑洞",
        "星系", "星云", "星团", "流星雨", "彗星", "银河", "深空",
    ]

    def __init__(self, db_path: Optional[str] = None):
        self.config = LongTermMemoryConfig.from_settings(db_path)
        self._repository = LongTermMemoryRepository(self.config.db_path)
        self._repository.initialize()

    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        return self._repository.load_profile(user_id)

    def save_profile(self, profile: UserProfile):
        profile.updated_at = profile.updated_at or _utcnow_iso()
        profile.created_at = profile.created_at or profile.updated_at
        self._repository.save_profile(profile)

    def record_memory_event(
        self,
        user_id: str,
        event_type: str,
        content: str,
        source: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ):
        event = MemoryEvent(
            event_type=event_type,
            content=content,
            source=source,
            confidence=confidence,
            metadata=metadata or {},
            timestamp=timestamp or time.time(),
        )
        self._repository.add_event(user_id, event)

    def get_recent_memory_events(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self._repository.get_recent_events(user_id, limit=limit)]

    def _should_attempt_extraction(self, user_message: str) -> bool:
        if not user_message or len(user_message.strip()) < 2:
            return False
        if any(keyword in user_message for keyword in self.EXTRACTION_KEYWORDS):
            return True
        celestial_pattern = (
            r"(火星|木星|土星|金星|月球|太阳|黑洞|星系|星云|星团|流星|彗星|"
            r"望远镜|赤道仪|拍摄|摄影|观测)"
        )
        if re.search(celestial_pattern, user_message):
            return True
        preference_indicators = [
            r"我[喜欢需要希望想]",
            r"[不要别]\S*",
            r"给我",
            r"能不能",
            r"可以吗",
            r"怎么[样看做]",
            r"什么[时候地方]",
        ]
        return any(re.search(pattern, user_message) for pattern in preference_indicators)

    def _extract_with_llm(self, user_message: str, assistant_message: str) -> Dict[str, Any]:
        from langchain_community.chat_models import ChatTongyi
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatTongyi(
            model=settings.MODEL_NAME,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            temperature=0.0,
            request_timeout=15,
        )
        response = llm.invoke(
            [
                SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=EXTRACTION_USER_TEMPLATE.format(
                        user_message=user_message,
                        assistant_message=assistant_message[:500],
                    )
                ),
            ]
        )
        content = response.content.strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(content)
        if not parsed.get("should_extract", False):
            return {"preferences": {}, "habits": {}, "constraints": []}
        return {
            "preferences": parsed.get("preferences", {}) or {},
            "habits": parsed.get("habits", {}) or {},
            "constraints": parsed.get("constraints", []) or [],
        }

    def _fallback_keyword_extraction(self, user_message: str, assistant_message: str) -> Dict[str, Any]:
        extracted = {"preferences": {}, "habits": {}, "constraints": []}

        if "简短" in user_message or "简单" in user_message:
            extracted["preferences"]["response_style"] = "简短"
        elif "详细" in user_message or "深入" in user_message:
            extracted["preferences"]["response_style"] = "详细"

        if "专业" in user_message:
            extracted["preferences"]["knowledge_level"] = "专业"
        elif "通俗" in user_message or "易懂" in user_message:
            extracted["preferences"]["knowledge_level"] = "通俗"

        if any(token in user_message for token in ["初学者", "入门", "刚开始"]):
            extracted["preferences"]["observation_experience"] = "初学者"
        elif any(token in user_message for token in ["高级", "进阶", "有经验"]):
            extracted["preferences"]["observation_experience"] = "高级"

        if "不要" in user_message and "术语" in user_message:
            extracted["constraints"].append("避免使用专业术语")
        if "不要超过" in user_message or "字以内" in user_message:
            extracted["constraints"].append("控制回答长度")
        if "夜里" in user_message or "晚上" in user_message:
            extracted["habits"]["preferred_time"] = "夜晚"

        if any(token in user_message for token in ["摄影", "拍摄", "相机", "赤道仪"]):
            extracted["habits"]["observation_type"] = "摄影"
        elif "深空" in user_message:
            extracted["habits"]["observation_type"] = "深空"
        elif "行星" in user_message:
            extracted["habits"]["observation_type"] = "行星"

        found_topics = [topic for topic in self.TOPIC_KEYWORDS if topic in user_message or topic in assistant_message]
        if found_topics:
            extracted["habits"]["frequent_topics"] = _dedupe_keep_order(found_topics)

        return extracted

    def extract_from_conversation(self, user_message: str, assistant_message: str) -> Dict[str, Any]:
        if not self._should_attempt_extraction(user_message):
            return {"preferences": {}, "habits": {}, "constraints": []}
        if settings.DASHSCOPE_API_KEY:
            try:
                return self._extract_with_llm(user_message, assistant_message)
            except Exception as exc:
                logger.warning(f"LLM画像提取失败，回退规则提取: {exc}")
        return self._fallback_keyword_extraction(user_message, assistant_message)

    def _merge_profile(self, existing: Optional[UserProfile], user_id: str, new_info: Dict[str, Any]) -> UserProfile:
        now = _utcnow_iso()
        if existing is None:
            return UserProfile(
                user_id=user_id,
                preferences=dict(new_info.get("preferences", {}) or {}),
                habits=dict(new_info.get("habits", {}) or {}),
                constraints=list(_dedupe_keep_order(new_info.get("constraints", []) or [])),
                created_at=now,
                updated_at=now,
            )

        merged_preferences = dict(existing.preferences)
        merged_preferences.update(new_info.get("preferences", {}) or {})

        merged_habits = dict(existing.habits)
        for key, value in (new_info.get("habits", {}) or {}).items():
            current_value = merged_habits.get(key)
            if isinstance(value, list):
                merged_habits[key] = _dedupe_keep_order((current_value or []) + value if isinstance(current_value, list) else value)
            else:
                merged_habits[key] = value

        merged_constraints = _dedupe_keep_order(existing.constraints + (new_info.get("constraints", []) or []))
        return UserProfile(
            user_id=user_id,
            preferences=merged_preferences,
            habits=merged_habits,
            constraints=merged_constraints,
            created_at=existing.created_at,
            updated_at=now,
        )

    def _record_profile_events(self, user_id: str, new_info: Dict[str, Any]):
        event_time = time.time()
        for key, value in (new_info.get("preferences", {}) or {}).items():
            self.record_memory_event(
                user_id=user_id,
                event_type="preference",
                content=f"{key}={value}",
                source="profile_merge",
                metadata={"field": key, "value": value},
                timestamp=event_time,
            )
        for key, value in (new_info.get("habits", {}) or {}).items():
            if isinstance(value, list):
                for item in value:
                    self.record_memory_event(
                        user_id=user_id,
                        event_type="habit",
                        content=f"{key}={item}",
                        source="profile_merge",
                        metadata={"field": key, "value": item},
                        timestamp=event_time,
                    )
            else:
                self.record_memory_event(
                    user_id=user_id,
                    event_type="habit",
                    content=f"{key}={value}",
                    source="profile_merge",
                    metadata={"field": key, "value": value},
                    timestamp=event_time,
                )
        for item in new_info.get("constraints", []) or []:
            self.record_memory_event(
                user_id=user_id,
                event_type="constraint",
                content=item,
                source="profile_merge",
                metadata={"value": item},
                timestamp=event_time,
            )

    def merge_and_update(self, user_id: str, new_info: Dict[str, Any]) -> UserProfile:
        existing = self.load_profile(user_id)
        profile = self._merge_profile(existing, user_id, new_info)
        self.save_profile(profile)
        if any(new_info.get(key) for key in ("preferences", "habits", "constraints")):
            self._record_profile_events(user_id, new_info)
        return profile

    def export_profile_snapshot(self, user_id: str) -> Dict[str, Any]:
        profile = self.load_profile(user_id)
        if not profile:
            return {}
        return {
            "user_id": profile.user_id,
            "preferences": profile.preferences,
            "habits": profile.habits,
            "constraints": profile.constraints,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "recent_events": self.get_recent_memory_events(user_id, limit=self.config.max_prompt_events),
        }

    def format_profile_for_prompt(self, user_id: str) -> str:
        profile = self.load_profile(user_id)
        if not profile:
            return "暂无用户偏好信息"

        parts: List[str] = []
        if profile.preferences:
            parts.append(
                "【用户偏好】\n" + "\n".join(f"- {key}: {value}" for key, value in profile.preferences.items())
            )
        if profile.habits:
            habit_lines = []
            for key, value in profile.habits.items():
                if isinstance(value, list):
                    habit_lines.append(f"- {key}: {', '.join(value[:self.config.max_topic_candidates])}")
                else:
                    habit_lines.append(f"- {key}: {value}")
            parts.append("【用户习惯】\n" + "\n".join(habit_lines))
        if profile.constraints:
            parts.append("【约束条件】\n" + "\n".join(f"- {item}" for item in profile.constraints))

        recent_events = self._repository.get_recent_events(user_id, limit=self.config.max_prompt_events)
        if recent_events:
            parts.append(
                "【近期记忆事件】\n"
                + "\n".join(
                    f"- {event.event_type}: {event.content}"
                    for event in recent_events
                )
            )

        return "\n\n".join(parts) if parts else "暂无用户偏好信息"

    def delete_profile(self, user_id: str) -> bool:
        return self._repository.delete_profile(user_id)


__all__ = [
    "LongTermMemory",
    "MemoryEvent",
    "Message",
    "SalientFact",
    "ShortTermMemory",
    "ToolCallRecord",
    "UserProfile",
]
