"""长期记忆抽取器。

只从明确的用户画像表达中抽取长期记忆信号，避免普通天文问题被误写入
用户画像。优先使用 LLM 抽取，失败或关闭时回退到保守规则。
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.agent.prompts import get_prompt_renderer
from src.core.config import settings
from src.core.logger import logger
from src.core.llm_factory import build_chat_model
from src.memory.long_term_memory.models import (
    ExtractionResult,
    MemoryType,
    SourceType,
)


EXPLICIT_PATTERNS = [
    r"我(喜欢|偏好|希望|要求|习惯)(.{1,30})",
    r"(不要|别|避免)(.{1,30})",
    r"给我(.{1,30})",
    r"我(是|有)(.{1,30})(经验|基础|背景)",
    r"请(用|以|按)(.{1,30})",
    r"我(的)(.{1,30})(是|叫|在)",
    r"记住(.{1,50})",
    r"以后(都|一直|总是)(.{1,30})",
    r"永远(不要|别)(.{1,30})",
]

TEMPORARY_INDICATORS = [
    "这次", "本次", "这回", "暂时", "仅此一次",
    "just this time", "for now", "temporarily",
]

TOPIC_KEYWORDS = [
    "火星", "木星", "土星", "金星", "月球", "太阳", "黑洞",
    "星系", "星云", "星团", "流星雨", "彗星", "银河", "深空",
    "望远镜", "赤道仪", "拍摄", "摄影", "观测",
]

EXTRACTION_KEYWORDS = [
    "简短", "详细", "专业", "通俗", "易懂", "不要", "喜欢", "偏好",
    "习惯", "经常", "总是", "希望", "要求", "建议", "初学者", "入门",
    "高级", "进阶", "望远镜", "相机", "拍摄", "观测", "深空", "行星",
    "月相", "流星雨", "日食", "月食", "星系", "星云", "星团",
]


class MemoryExtractor:
    """识别并抽取用户偏好、约束、背景、事实等长期记忆候选。"""

    def __init__(self):
        self._explicit_patterns = [re.compile(p) for p in EXPLICIT_PATTERNS]

    def should_attempt_extraction(self, user_message: str) -> bool:
        """Only trigger extraction when the user expresses a clear profile signal.

        Disqualifying patterns:
        - Generic astronomy questions (what, when, how, can I, is it)
        - Bare celestial body names (Mars, Jupiter, nebula, etc.)
        - Observation/topic keywords without explicit preference framing
        """
        if not user_message or len(user_message.strip()) < 2:
            return False

        msg = user_message

        # Explicit memory / long-term preference signals
        memory_signals = [
            r"记住",
            r"请记住",
            r"以后",
            r"以后都",
            r"以后默认",
            r"下次",
            r"下回",
            r"永远",
            r"永远不要",
            r"一直",
            r"总是",
            r"每次都",
        ]
        if any(re.search(pattern, msg) for pattern in memory_signals):
            return True

        # Explicit user profile: preference / habit / constraint
        profile_signals = [
            r"我[喜欢偏好习惯希望要求]",
            r"我不[喜欢想希望要]",
            r"我[更较]喜欢",
            r"我不太",
            r"不要",
            r"别[说给提]",
            r"避免",
            r"禁止",
        ]
        if any(re.search(pattern, msg) for pattern in profile_signals):
            # Check it's not a one-off request disguised as a preference
            if not self.is_temporary_request(msg):
                return True
            # Even temporary requests with explicit memory signals qualify
            if any(re.search(p, msg) for p in memory_signals):
                return True

        # Formatting-style signals (请用/请以/请按) only count with memory signal
        formatting_signal = re.search(r"请(用|以|按|根据|按照)", msg)
        if formatting_signal and any(re.search(p, msg) for p in memory_signals):
            return True

        # Explicit equipment / device mention
        equipment_signals = [
            r"我有一[台个架]",
            r"我的[观测天]?[设备望远镜相机赤道仪]",
            r"我用的[是]",
            r"我用.{1,6}(望远镜|相机|赤道仪|镜头|目镜|CCD|CMOS)",
            r"我.{1,6}(望远镜|相机|赤道仪)",
            r"我主要[拍观测看]",
        ]
        if any(re.search(pattern, msg) for pattern in equipment_signals):
            return True

        # Explicit location declaration
        location_signals = [
            r"我在.{1,10}(观测|看|拍照)",
            r"我的观测地[点址]",
            r"观测地[点址][是在]",
        ]
        if any(re.search(pattern, msg) for pattern in location_signals):
            return True

        # Skill level declaration
        skill_signals = [
            r"我是(初学|新手|入门|刚开|有经|进阶|高级|专家|资深)",
            r"我.{1,4}(初学|新手|入门|刚开|有经|进阶|高级|专家|资深)",
        ]
        if any(re.search(pattern, msg) for pattern in skill_signals):
            return True

        # City name + explicit context (only when combined with profile/equipment signal)
        city_with_context = re.search(
            r"(北京|上海|广州|深圳|杭州|苏州|成都|南京|武汉)",
            msg,
        )
        if city_with_context and any(
            cue in msg
            for cue in [
                "观测",
                "拍照",
                "拍摄",
                "望远镜",
                "设备",
                "经纬度",
                "地点",
                "光害",
            ]
        ):
            return True

        return False

    def is_explicit_expression(self, text: str) -> bool:
        """判断文本是否包含明确的长期偏好或约束表达。"""

        return any(pattern.search(text) for pattern in self._explicit_patterns)

    def is_temporary_request(self, text: str) -> bool:
        """识别“这次/暂时”等一次性请求，降低写入长期记忆的置信度。"""

        return any(indicator in text for indicator in TEMPORARY_INDICATORS)

    def extract_with_llm(
        self, user_message: str, assistant_message: str, conversation_id: Optional[str] = None
    ) -> List[ExtractionResult]:
        """调用轻量 LLM 按结构化协议抽取长期记忆候选。"""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = build_chat_model(
                model=settings.LTM_EXTRACT_MODEL_NAME or settings.SMALL_MODEL_NAME,
                temperature=0.0,
                request_timeout=settings.LTM_EXTRACT_TIMEOUT_SECONDS,
                max_retries=settings.LTM_EXTRACT_MAX_RETRIES,
            )
            response = llm.invoke(
                [
                    SystemMessage(
                        content=get_prompt_renderer().render(
                            "memory.long_term_extractor.system"
                        )
                    ),
                    HumanMessage(
                        content=get_prompt_renderer().render(
                            "memory.long_term_extractor.user",
                            {
                                "user_message": user_message,
                                "assistant_message": assistant_message[:500],
                            },
                        )
                    ),
                ]
            )
            content = response.content.strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(content)
            extractions = parsed.get("extractions", [])
            results = []
            for ext in extractions:
                is_explicit = ext.get("is_explicit", False)
                is_temporary = ext.get("is_temporary", False)
                confidence = ext.get("confidence", 0.5)
                if is_explicit:
                    confidence = max(confidence, 0.8)
                if is_temporary:
                    confidence *= 0.5
                results.append(
                    ExtractionResult(
                        should_extract=True,
                        memory_type=ext.get("memory_type", "preference"),
                        category=ext.get("category", ""),
                        key=ext.get("key", ""),
                        value=ext.get("value"),
                        confidence=min(confidence, 1.0),
                        source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                        is_explicit=is_explicit,
                        is_temporary=is_temporary,
                        raw_content=user_message[:200],
                    )
                )
            return results
        except Exception as exc:
            logger.warning(f"LLM记忆提取失败，回退规则提取: {exc}")
            return self._fallback_keyword_extraction(user_message, assistant_message)

    def _fallback_keyword_extraction(
        self, user_message: str, assistant_message: str
    ) -> List[ExtractionResult]:
        """在 LLM 不可用时，用保守关键词规则抽取明确画像信号。"""

        results: List[ExtractionResult] = []
        is_explicit = self.is_explicit_expression(user_message)
        is_temporary = self.is_temporary_request(user_message)
        base_confidence = 0.8 if is_explicit else 0.4
        if is_temporary:
            base_confidence *= 0.5

        if "简短" in user_message or "简单" in user_message:
            results.append(ExtractionResult(
                should_extract=True, memory_type=MemoryType.PREFERENCE,
                category="response_style", key="response_style", value="简短",
                confidence=base_confidence, source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                is_explicit=is_explicit, is_temporary=is_temporary, raw_content=user_message[:200],
            ))
        elif "详细" in user_message or "深入" in user_message:
            results.append(ExtractionResult(
                should_extract=True, memory_type=MemoryType.PREFERENCE,
                category="response_style", key="response_style", value="详细",
                confidence=base_confidence, source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                is_explicit=is_explicit, is_temporary=is_temporary, raw_content=user_message[:200],
            ))

        if "专业" in user_message:
            results.append(ExtractionResult(
                should_extract=True, memory_type=MemoryType.PREFERENCE,
                category="knowledge_level", key="knowledge_level", value="专业",
                confidence=base_confidence, source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                is_explicit=is_explicit, is_temporary=is_temporary, raw_content=user_message[:200],
            ))
        elif "通俗" in user_message or "易懂" in user_message:
            results.append(ExtractionResult(
                should_extract=True, memory_type=MemoryType.PREFERENCE,
                category="knowledge_level", key="knowledge_level", value="通俗",
                confidence=base_confidence, source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                is_explicit=is_explicit, is_temporary=is_temporary, raw_content=user_message[:200],
            ))

        if any(token in user_message for token in ["初学者", "入门", "刚开始"]):
            results.append(ExtractionResult(
                should_extract=True, memory_type=MemoryType.BACKGROUND,
                category="skill_level", key="skill_level", value="入门",
                confidence=base_confidence, source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                is_explicit=is_explicit, is_temporary=is_temporary, raw_content=user_message[:200],
            ))
        elif any(token in user_message for token in ["高级", "进阶", "有经验"]):
            results.append(ExtractionResult(
                should_extract=True, memory_type=MemoryType.BACKGROUND,
                category="skill_level", key="skill_level", value="专家",
                confidence=base_confidence, source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                is_explicit=is_explicit, is_temporary=is_temporary, raw_content=user_message[:200],
            ))

        if "不要" in user_message and "术语" in user_message:
            results.append(ExtractionResult(
                should_extract=True, memory_type=MemoryType.CONSTRAINT,
                category="no_jargon", key="no_jargon", value=True,
                confidence=base_confidence, source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                is_explicit=is_explicit, is_temporary=is_temporary, raw_content=user_message[:200],
            ))
        if "不要超过" in user_message or "字以内" in user_message:
            results.append(ExtractionResult(
                should_extract=True, memory_type=MemoryType.CONSTRAINT,
                category="output_length_limit", key="output_length_limit", value="控制回答长度",
                confidence=base_confidence, source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                is_explicit=is_explicit, is_temporary=is_temporary, raw_content=user_message[:200],
            ))

        # Device / equipment: only extract when explicitly declared with ownership framing
        equipment_match = re.search(
            r"我(有|用|的).{0,10}(望远镜|相机|赤道仪|镜头|目镜|CCD|CMOS)",
            user_message,
        )
        if equipment_match:
            results.append(ExtractionResult(
                should_extract=True, memory_type=MemoryType.BACKGROUND,
                category="device_info", key="device_info", value=equipment_match.group(0).strip(),
                confidence=base_confidence, source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                is_explicit=is_explicit, is_temporary=is_temporary, raw_content=user_message[:200],
            ))

        # Location: only extract when explicitly declared
        location_match = re.search(
            r"(北京|上海|广州|深圳|杭州|苏州|成都|南京|武汉)",
            user_message,
        )
        if location_match and any(
            (cue in user_message)
            for cue in ["观测", "拍照", "拍摄", "地点", "经纬度", "我家在", "我在"]
        ):
            results.append(ExtractionResult(
                should_extract=True, memory_type=MemoryType.FACT,
                category="location_info", key="location_info",
                value=location_match.group(1),
                confidence=base_confidence, source_type=SourceType.EXPLICIT if is_explicit else SourceType.AUTO,
                is_explicit=is_explicit, is_temporary=is_temporary, raw_content=user_message[:200],
            ))

        # Only return results for explicit profile expressions; ignore general topics.
        # Equipment / location results from the conservative rules above are valid.
        if not results:
            return []
        return results

    def extract_from_conversation(
        self, user_message: str, assistant_message: str, conversation_id: Optional[str] = None
    ) -> List[ExtractionResult]:
        """从一轮对话中抽取长期记忆候选，统一处理开关、门控和 fallback。"""

        if not settings.LTM_EXTRACT_ENABLED:
            return []
        if not self.should_attempt_extraction(user_message):
            return []

        if settings.LTM_LLM_EXTRACT_ENABLED and settings.DASHSCOPE_API_KEY:
            try:
                return self.extract_with_llm(user_message, assistant_message, conversation_id)
            except Exception as exc:
                logger.warning(f"LLM记忆提取失败，回退规则提取: {exc}")

        # Conservative fallback: only extract from explicit expressions
        return self._fallback_keyword_extraction(user_message, assistant_message)

    def extract_legacy_format(
        self, user_message: str, assistant_message: str
    ) -> Dict[str, Any]:
        """把抽取结果转换为旧版 profile 字段结构。"""

        results = self.extract_from_conversation(user_message, assistant_message)
        if not results:
            return {"preferences": {}, "habits": {}, "constraints": [], "background": {}, "facts": []}

        preferences: Dict[str, Any] = {}
        habits: Dict[str, Any] = {}
        constraints: List[str] = []
        background: Dict[str, Any] = {}
        facts: List[Dict[str, Any]] = []
        legacy_constraint_labels = {
            "no_jargon": "避免使用专业术语",
            "output_length_limit": "控制回答长度",
        }

        for r in results:
            if r.is_temporary:
                continue
            if r.memory_type == MemoryType.PREFERENCE:
                preferences[r.key] = r.value
            elif r.memory_type == MemoryType.HABIT:
                if r.key == "frequent_topics" and isinstance(r.value, list):
                    existing = habits.get("frequent_topics", [])
                    merged = list(dict.fromkeys(existing + r.value))
                    habits["frequent_topics"] = merged
                else:
                    habits[r.key] = r.value
            elif r.memory_type == MemoryType.CONSTRAINT:
                if isinstance(r.value, str):
                    constraints.append(r.value)
                elif isinstance(r.value, bool) and r.value:
                    constraints.append(legacy_constraint_labels.get(r.key, r.key))
            elif r.memory_type == MemoryType.BACKGROUND:
                background[r.key] = r.value
            elif r.memory_type == MemoryType.FACT:
                facts.append({"key": r.key, "value": r.value, "category": r.category})

        return {
            "preferences": preferences,
            "habits": habits,
            "constraints": constraints,
            "background": background,
            "facts": facts,
        }
