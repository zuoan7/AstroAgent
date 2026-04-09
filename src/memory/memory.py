from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from src.core.config import settings
import sqlite3
import json
import os
from datetime import datetime
from src.core.logger import logger


@dataclass
class Message:
    role: str
    content: str
    timestamp: float


class ShortTermMemory:

    def __init__(self):
        self.messages: List[Message] = []
        self.max_size = settings.MEMORY_SIZE

    def add_message(self, role: str, content: str, timestamp: float):
        message = Message(role=role, content=content, timestamp=timestamp)
        self.messages.append(message)

        if len(self.messages) > self.max_size:
            self.messages = self.messages[-self.max_size:]

    def get_recent_messages(self, window: int = None) -> List[Dict[str, str]]:
        if window is None:
            window = settings.MEMORY_WINDOW

        recent_messages = self.messages[-window:]
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ]

    def clear(self):
        self.messages.clear()

    def get_size(self) -> int:
        return len(self.messages)


@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, Any]
    habits: Dict[str, Any]
    constraints: List[str]
    created_at: str
    updated_at: str


class LongTermMemory:

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "memory",
            "long_term_memory",
            "user_profiles.sqlite"
        )
        self._ensure_db_exists()
        self._ensure_table_exists()

    def _ensure_db_exists(self):
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"✅ 创建长期记忆存储目录: {db_dir}")

    def _ensure_table_exists(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL,
                    habits TEXT NOT NULL,
                    constraints TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        logger.info(f"✅ 长期记忆数据库已初始化: {self.db_path}")

    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_id, preferences, habits, constraints, created_at, updated_at "
                "FROM user_profiles WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()

        if row:
            return UserProfile(
                user_id=row[0],
                preferences=json.loads(row[1]),
                habits=json.loads(row[2]),
                constraints=json.loads(row[3]),
                created_at=row[4],
                updated_at=row[5]
            )
        return None

    def save_profile(self, profile: UserProfile):
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_profiles
                (user_id, preferences, habits, constraints, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                json.dumps(profile.preferences, ensure_ascii=False),
                json.dumps(profile.habits, ensure_ascii=False),
                json.dumps(profile.constraints, ensure_ascii=False),
                profile.created_at or now,
                now
            ))
            conn.commit()
        logger.info(f"✅ 用户画像已保存: {profile.user_id}")

    def extract_from_conversation(self, user_message: str, assistant_message: str) -> Dict[str, Any]:
        extracted = {
            "preferences": {},
            "habits": {},
            "constraints": []
        }

        if "简短" in user_message or "简单" in user_message:
            extracted["preferences"]["response_style"] = "简短"
        elif "详细" in user_message or "深入" in user_message:
            extracted["preferences"]["response_style"] = "详细"

        if "专业" in user_message:
            extracted["preferences"]["knowledge_level"] = "专业"
        elif "通俗" in user_message or "易懂" in user_message:
            extracted["preferences"]["knowledge_level"] = "通俗"

        if "不要" in user_message and "术语" in user_message:
            extracted["constraints"].append("避免使用专业术语")
        if "不要超过" in user_message or "字以内" in user_message:
            extracted["constraints"].append("控制回答长度")

        import re
        celestial_keywords = ["火星", "木星", "土星", "金星", "月球", "太阳", "黑洞", "星系", "星云", "星团"]
        found_celestial = [kw for kw in celestial_keywords if kw in user_message]
        if found_celestial:
            if "frequent_topics" not in extracted["habits"]:
                extracted["habits"]["frequent_topics"] = []
            for topic in found_celestial:
                if topic not in extracted["habits"]["frequent_topics"]:
                    extracted["habits"]["frequent_topics"].append(topic)

        return extracted

    def merge_and_update(self, user_id: str, new_info: Dict[str, Any]) -> UserProfile:
        existing = self.load_profile(user_id)

        if existing:
            for key, value in new_info.get("preferences", {}).items():
                existing.preferences[key] = value

            for key, value in new_info.get("habits", {}).items():
                if key in existing.habits:
                    if isinstance(value, list):
                        if isinstance(existing.habits[key], list):
                            existing.habits[key] = list(set(existing.habits[key] + value))
                        else:
                            existing.habits[key] = value
                else:
                    existing.habits[key] = value

            new_constraints = new_info.get("constraints", [])
            for constraint in new_constraints:
                if constraint not in existing.constraints:
                    existing.constraints.append(constraint)

            existing.updated_at = datetime.now().isoformat()

            profile = existing
        else:
            now = datetime.now().isoformat()
            profile = UserProfile(
                user_id=user_id,
                preferences=new_info.get("preferences", {}),
                habits=new_info.get("habits", {}),
                constraints=new_info.get("constraints", []),
                created_at=now,
                updated_at=now
            )

        self.save_profile(profile)
        return profile

    def format_profile_for_prompt(self, user_id: str) -> str:
        profile = self.load_profile(user_id)
        if not profile:
            return "暂无用户偏好信息"

        parts = []

        if profile.preferences:
            prefs = []
            for key, value in profile.preferences.items():
                prefs.append(f"- {key}: {value}")
            parts.append("**用户偏好**:\n" + "\n".join(prefs))

        if profile.habits:
            habits = []
            for key, value in profile.habits.items():
                if isinstance(value, list):
                    habits.append(f"- {key}: {', '.join(value)}")
                else:
                    habits.append(f"- {key}: {value}")
            parts.append("**用户习惯**:\n" + "\n".join(habits))

        if profile.constraints:
            constraints = "\n".join(f"- {c}" for c in profile.constraints)
            parts.append("**约束条件**:\n" + constraints)

        return "\n\n".join(parts) if parts else "暂无用户偏好信息"

    def delete_profile(self, user_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            deleted = cursor.rowcount > 0
            conn.commit()

        if deleted:
            logger.info(f"✅ 用户画像已删除: {user_id}")

        return deleted
