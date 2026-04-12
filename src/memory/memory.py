from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from src.core.config import settings
import sqlite3
import json
import os
from datetime import datetime
from src.core.logger import logger
from src.core.errors import AgentError, ErrorCode


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
- 如果用户提到了特定天体或观测类型，记录到frequent_topics（如"木星""星云""流星雨"等）
- 如果用户表达了观测经验水平（如"我刚入门""我用8寸DOB"等），记录到observation_experience
- 如果用户提到了观测设备（如"望远镜""赤道仪""单反"等），记录到observation_type
- 如果用户表达了限制条件（如"不要用术语""字数限制"等），记录到constraints
- 不要从一般性问答中推断偏好，仅提取明确表达的信息
- 如果没有可提取的信息，设置should_extract为false，其他字段留空"""


EXTRACTION_USER_TEMPLATE = """请从以下对话中提取用户画像信息：

用户消息：{user_message}
助手回复：{assistant_message}

请输出JSON格式的提取结果："""


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

    EXTRACTION_KEYWORDS = [
        "简短", "详细", "专业", "通俗", "易懂", "不要", "喜欢", "偏好",
        "习惯", "经常", "总是", "希望", "要求", "建议", "初学者", "入门",
        "高级", "进阶", "望远镜", "相机", "拍摄", "观测", "深空", "行星",
        "月相", "流星雨", "日食", "月食", "星系", "星云", "星团",
    ]

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

    def _should_attempt_extraction(self, user_message: str) -> bool:
        """
        增量提取判断：仅当用户消息中包含可能包含偏好/习惯/约束信息的信号时才触发提取
        """
        if not user_message or len(user_message.strip()) < 2:
            return False

        message_lower = user_message.lower()

        for keyword in self.EXTRACTION_KEYWORDS:
            if keyword in message_lower:
                return True

        import re
        celestial_pattern = r'(火星|木星|土星|金星|月球|太阳|黑洞|星系|星云|星团|流星|彗星|望远镜|赤道仪|拍摄|摄影|观测)'
        if re.search(celestial_pattern, user_message):
            return True

        preference_indicators = [
            r'我[喜欢需要希望]',
            r'[不要别][想用看说]',
            r'给我',
            r'能不能',
            r'可以吗',
            r'怎么[样看做]',
            r'什么[时候地方]',
        ]
        for pattern in preference_indicators:
            if re.search(pattern, user_message):
                return True

        return False

    def _extract_with_llm(self, user_message: str, assistant_message: str) -> Dict[str, Any]:
        """
        使用LLM进行结构化信息提取
        """
        try:
            from langchain_community.chat_models import ChatTongyi
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatTongyi(
                model=settings.MODEL_NAME,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
                temperature=0.0
            )

            messages = [
                SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=EXTRACTION_USER_TEMPLATE.format(
                    user_message=user_message,
                    assistant_message=assistant_message[:500]
                ))
            ]

            response = llm.invoke(messages)
            result_text = response.content.strip()

            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            parsed = json.loads(result_text)

            if not parsed.get("should_extract", False):
                logger.debug("LLM判断无需提取用户画像信息")
                return {"preferences": {}, "habits": {}, "constraints": []}

            return {
                "preferences": parsed.get("preferences", {}),
                "habits": parsed.get("habits", {}),
                "constraints": parsed.get("constraints", [])
            }

        except json.JSONDecodeError as e:
            logger.warning(f"LLM提取结果JSON解析失败: {e}")
            return self._fallback_keyword_extraction(user_message, assistant_message)
        except Exception as e:
            logger.warning(f"LLM提取失败，回退到关键词提取: {e}")
            return self._fallback_keyword_extraction(user_message, assistant_message)

    def _fallback_keyword_extraction(self, user_message: str, assistant_message: str) -> Dict[str, Any]:
        """
        关键词提取作为降级方案
        """
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

    def extract_from_conversation(self, user_message: str, assistant_message: str) -> Dict[str, Any]:
        """
        从对话中提取用户画像信息

        使用增量提取判断逻辑：先判断是否值得提取，再使用LLM进行结构化提取
        如果LLM不可用，回退到关键词提取
        """
        if not self._should_attempt_extraction(user_message):
            logger.debug("增量判断：跳过本次提取（用户消息未包含偏好信号）")
            return {"preferences": {}, "habits": {}, "constraints": []}

        if settings.DASHSCOPE_API_KEY:
            return self._extract_with_llm(user_message, assistant_message)
        else:
            logger.debug("DASHSCOPE_API_KEY未配置，使用关键词提取降级方案")
            return self._fallback_keyword_extraction(user_message, assistant_message)

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
