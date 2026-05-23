"""长期记忆抽取器。

实现 select_strategy 中长期记忆抽取 gating：先用快速规则筛出可能画像信号，
再用最近对话窗口聚合稳定/临时/撤回等信号，最后优先交给 LLM 做判别+抽取，
失败或关闭时回退到保守规则，避免普通天文问题被误写入用户画像。
"""

import json
import re
from collections import Counter
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
    "今天", "今晚", "这一次", "临时",
    "just this time", "for now", "temporarily", "today", "tonight",
]

STABLE_INDICATORS = [
    "记住", "以后", "以后默认", "下次", "下回", "每次", "每次都",
    "总是", "一直", "通常", "默认", "长期", "永远",
    "remember", "next time", "always", "usually", "default",
]

REVOCATION_PATTERNS = [
    r"忘掉",
    r"别记",
    r"不要记",
    r"不用记",
    r"不再",
    r"现在不再",
    r"不是了",
    r"作废",
    r"撤回",
    r"取消之前",
    r"之前.*(不算|作废)",
    r"那次只是(开玩笑|临时)",
    r"只是开玩笑",
    r"改成",
    r"forget",
    r"do not remember",
    r"don't remember",
    r"no longer",
    r"revoke",
]

EQUIPMENT_TERMS = [
    "望远镜", "镜", "相机", "赤道仪", "镜头", "目镜", "CCD", "CMOS",
    "道布森", "小黑", "星特朗", "信达", "佳能", "尼康", "索尼",
    "telescope", "camera", "mount", "eyepiece",
]

LOCATION_TERMS = [
    "观测地", "观测地点", "地点", "位置", "经纬度", "纬度", "经度",
    "北京", "上海", "广州", "深圳", "杭州", "苏州", "成都", "南京", "武汉",
    "location", "site", "latitude", "longitude", "timezone", "time zone",
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
        """预编译显式画像表达正则，减少每次抽取的重复开销。"""

        self._explicit_patterns = [re.compile(p) for p in EXPLICIT_PATTERNS]

    def should_attempt_extraction(
        self,
        user_message: str,
        conversation_window: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """判断是否需要进入长期记忆抽取。"""

        if not user_message or len(user_message.strip()) < 2:
            return False

        if self._fast_should_attempt_extraction(user_message):
            return True

        signals = self.analyze_conversation_window(
            user_message=user_message,
            conversation_window=conversation_window,
        )
        return bool(signals.get("should_attempt"))

    def _fast_should_attempt_extraction(self, user_message: str) -> bool:
        """单轮快速过滤层，保守排除普通天文问答和一次性风格请求。"""

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
            r"以后还",
            r"默认这个",
            r"同一台",
            r"同一个",
        ]
        if any(re.search(pattern, msg) for pattern in memory_signals):
            return True

        if self.is_revocation_request(msg):
            return True

        # Explicit user profile: preference / habit / constraint
        profile_signals = [
            r"我[喜欢偏好习惯希望要求]",
            r"我不[喜欢想希望要]",
            r"我[更较]喜欢",
            r"我不太",
            r"不要",
            r"别[说给提]",
            r"别用",
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
            r"下次.{0,12}(同一台|这台|这个).{0,8}(镜|望远镜|相机|赤道仪)",
        ]
        if any(re.search(pattern, msg) for pattern in equipment_signals):
            return True

        # Explicit location declaration
        location_signals = [
            r"我在.{1,10}(观测|看|拍照)",
            r"我的观测地[点址]",
            r"观测地[点址][是在]",
            r"以后默认.{0,12}(这个|这里|地点|位置|观测地)",
            r"下次.{0,12}(这个|这里|地点|位置|观测地)",
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

    def is_revocation_request(self, text: str) -> bool:
        """识别用户撤回、作废或更改之前长期记忆的表达。"""

        lowered = (text or "").lower()
        return any(re.search(pattern, lowered) for pattern in REVOCATION_PATTERNS)

    def _normalize_conversation_window(
        self,
        user_message: str,
        assistant_message: str = "",
        conversation_id: Optional[str] = None,
        conversation_window: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """规范化最近 K=4 轮窗口，兼容调用方未包含当前轮的情况。"""

        normalized: List[Dict[str, Any]] = []
        for turn in conversation_window or []:
            if not isinstance(turn, dict):
                continue
            normalized.append(
                {
                    "user_message": str(turn.get("user_message") or ""),
                    "assistant_message": str(turn.get("assistant_message") or ""),
                    "conversation_id": turn.get("conversation_id"),
                }
            )

        if not normalized or normalized[-1].get("user_message") != user_message:
            normalized.append(
                {
                    "user_message": user_message or "",
                    "assistant_message": assistant_message or "",
                    "conversation_id": conversation_id,
                }
            )
        return normalized[-4:]

    def analyze_conversation_window(
        self,
        user_message: str,
        conversation_window: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """窗口聚合层：计算信息密度和 gating 理由。"""

        window = self._normalize_conversation_window(
            user_message=user_message,
            conversation_window=conversation_window,
        )
        user_texts = [turn.get("user_message", "") for turn in window]
        joined = "\n".join(user_texts)
        lowered = joined.lower()

        equipment_hits = self._extract_repeated_terms(joined, EQUIPMENT_TERMS)
        location_hits = self._extract_repeated_terms(joined, LOCATION_TERMS)
        correction_targets = self._extract_correction_targets(joined)
        repeated_signal = bool(equipment_hits or location_hits)

        signals: Dict[str, Any] = {
            "self_reference": bool(
                re.search(r"(我|我的|咱们|俺|\bI\b|\bmy\b|\bme\b)", joined, re.I)
            ),
            "action_modal": bool(
                re.search(
                    r"(喜欢|偏好|希望|要求|习惯|想|要|不要|避免|禁止|有|用|在|"
                    r"默认|记住|忘掉|不再|作废|改成|prefer|want|avoid|own|use|default)",
                    lowered,
                    re.I,
                )
            ),
            "stable_marker": any(indicator in lowered for indicator in STABLE_INDICATORS),
            "temporary_marker": self.is_temporary_request(lowered),
            "equipment": any(term.lower() in lowered for term in EQUIPMENT_TERMS),
            "location": any(term.lower() in lowered for term in LOCATION_TERMS),
            "correction": bool(correction_targets) or self.is_revocation_request(joined),
            "repeated_signal": repeated_signal,
            "repeated_equipment": equipment_hits,
            "repeated_location": location_hits,
            "correction_targets": correction_targets,
            "revocation": self.is_revocation_request(joined),
        }

        score = 0.0
        if signals["self_reference"]:
            score += 1.0
        if signals["action_modal"]:
            score += 1.0
        if signals["stable_marker"]:
            score += 1.5
        if signals["equipment"]:
            score += 0.8
        if signals["location"]:
            score += 0.8
        if signals["correction"]:
            score += 1.0
        if signals["repeated_signal"]:
            score += 1.5
        if signals["temporary_marker"] and not signals["stable_marker"]:
            score -= 1.0

        should_attempt = False
        reasons: List[str] = []
        if signals["revocation"]:
            should_attempt = True
            reasons.append("revocation_signal")
        if signals["stable_marker"] and signals["action_modal"] and score >= 2.5:
            should_attempt = True
            reasons.append("stable_profile_signal")
        if signals["repeated_signal"] and score >= 3.0:
            should_attempt = True
            reasons.append("window_repeated_signal")
        if signals["correction"] and (
            signals["stable_marker"] or signals["repeated_signal"]
        ):
            should_attempt = True
            reasons.append("correction_signal")

        signals["density_score"] = round(max(score, 0.0), 2)
        signals["gate_reason"] = ",".join(dict.fromkeys(reasons)) or "no_profile_signal"
        signals["should_attempt"] = should_attempt
        signals["window_size"] = len(window)
        return signals

    def _extract_repeated_terms(self, text: str, terms: List[str]) -> List[str]:
        """返回窗口内出现两次以上的设备、地点等稳定项。"""

        lowered = text.lower()
        counts = Counter(
            term
            for term in terms
            if lowered.count(term.lower()) >= 2
        )

        model_like = re.findall(
            r"(?:星特朗|信达|佳能|尼康|索尼|Celestron|Canon|Nikon|Sony)"
            r"[A-Za-z0-9一-龥\-]{1,12}",
            text,
            flags=re.I,
        )
        counts.update(item for item in model_like if model_like.count(item) >= 2)

        coordinate_like = re.findall(
            r"(?:东经|西经|北纬|南纬)?\s*\d{1,3}(?:\.\d+)?\s*(?:°|度)",
            text,
        )
        counts.update(item.strip() for item in coordinate_like if coordinate_like.count(item) >= 2)
        return list(counts.keys())

    def _extract_correction_targets(self, text: str) -> List[str]:
        """抽取多次否定的工具或数据源名称，用于 tentative 约束候选。"""

        targets = re.findall(
            r"(?:不要用|别用|不信任|不准|不准确|不可靠|错了|不要相信)\s*"
            r"([A-Za-z0-9_\-一-龥]{2,30})",
            text,
        )
        counts = Counter(target.strip(" ，。,.") for target in targets if target.strip())
        return [target for target, count in counts.items() if count >= 2]

    def extract_with_llm(
        self,
        user_message: str,
        assistant_message: str,
        conversation_id: Optional[str] = None,
        conversation_window: Optional[List[Dict[str, Any]]] = None,
        gating_signals: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionResult]:
        """调用轻量 LLM 按结构化协议抽取长期记忆候选。"""

        window = self._normalize_conversation_window(
            user_message=user_message,
            assistant_message=assistant_message,
            conversation_id=conversation_id,
            conversation_window=conversation_window,
        )
        signals = gating_signals or self.analyze_conversation_window(
            user_message=user_message,
            conversation_window=window,
        )

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
                                "conversation_window_json": json.dumps(
                                    window, ensure_ascii=False
                                ),
                                "gating_signals_json": json.dumps(
                                    signals, ensure_ascii=False
                                ),
                            },
                        )
                    ),
                ]
            )
            content = response.content.strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(content)
            if "should_extract" not in parsed or "reason" not in parsed:
                raise ValueError("LLM extraction response missing should_extract/reason")
            if not parsed.get("should_extract"):
                return []
            extractions = parsed.get("extractions", [])
            if not isinstance(extractions, list):
                raise ValueError("LLM extraction response extractions is not a list")
            results = []
            for ext in extractions:
                if not isinstance(ext, dict):
                    continue
                is_explicit = ext.get("is_explicit", False)
                is_temporary = ext.get("is_temporary", False)
                confidence = ext.get("confidence", 0.5)
                if is_explicit:
                    confidence = max(confidence, 0.8)
                if is_temporary:
                    confidence *= 0.5
                grade = ext.get(
                    "extraction_grade",
                    "solid" if is_explicit else "tentative",
                )
                action = ext.get("action", "upsert")
                metadata = {
                    "llm_reason": parsed.get("reason", ""),
                    "gate_reason": signals.get("gate_reason", ""),
                    "density_score": signals.get("density_score"),
                    **(ext.get("metadata") or {}),
                }
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
                        extraction_grade=grade,
                        gate_reason=signals.get("gate_reason", ""),
                        action=action,
                        metadata=metadata,
                    )
                )
            return results
        except Exception as exc:
            logger.warning(f"LLM记忆提取失败，回退规则提取: {exc}")
            return self._fallback_keyword_extraction(
                user_message,
                assistant_message,
                conversation_window=window,
                gating_signals=signals,
            )

    def _fallback_keyword_extraction(
        self,
        user_message: str,
        assistant_message: str,
        conversation_window: Optional[List[Dict[str, Any]]] = None,
        gating_signals: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionResult]:
        """在 LLM 不可用时，用保守关键词规则抽取明确画像信号。"""

        results: List[ExtractionResult] = []
        window = self._normalize_conversation_window(
            user_message=user_message,
            assistant_message=assistant_message,
            conversation_window=conversation_window,
        )
        signals = gating_signals or self.analyze_conversation_window(
            user_message=user_message,
            conversation_window=window,
        )
        is_explicit = self.is_explicit_expression(user_message)
        is_temporary = self.is_temporary_request(user_message)
        base_confidence = 0.8 if is_explicit else 0.4
        if is_temporary:
            base_confidence *= 0.5

        revoke_result = None
        if self.is_revocation_request(user_message):
            revoke_result = self._build_revoke_extraction(user_message, signals)
            results.append(revoke_result)
            if "改成" not in user_message:
                return self._finalize_fallback_results(results, signals)

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

        results.extend(self._implicit_window_extractions(window, signals, user_message))

        # Only return results for profile expressions; ignore general topics.
        if not results:
            return []
        return self._finalize_fallback_results(results, signals)

    def _build_revoke_extraction(
        self, user_message: str, signals: Dict[str, Any]
    ) -> ExtractionResult:
        """构造撤回动作，尽量携带可精确定位的 type/category/key。"""

        target = self._infer_revoke_target(user_message)
        metadata = {
            "target_text": user_message[:200],
            "revoke_reason": user_message[:200],
            "gate_reason": signals.get("gate_reason", "revocation_signal"),
            "density_score": signals.get("density_score"),
        }
        metadata.update(target.get("metadata", {}))
        return ExtractionResult(
            should_extract=True,
            memory_type=target.get("memory_type", ""),
            category=target.get("category", ""),
            key=target.get("key", ""),
            value=f"用户撤回请求: {user_message[:120]}",
            confidence=0.85,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
            is_temporary=False,
            raw_content=user_message[:200],
            extraction_grade="solid",
            gate_reason=signals.get("gate_reason", "revocation_signal"),
            action="revoke",
            metadata=metadata,
        )

    def _infer_revoke_target(self, text: str) -> Dict[str, Any]:
        """从撤回请求中推断目标记忆键；不明确时留空，交给服务降级处理。"""

        if re.search(r"(设备|望远镜|相机|赤道仪|镜头|目镜|CCD|CMOS|同一台|这台|镜)", text):
            return {
                "memory_type": MemoryType.BACKGROUND,
                "category": "device_info",
                "key": "device_info",
            }
        if re.search(r"(地点|位置|观测地|这里|北京|上海|广州|深圳|杭州|成都|南京|武汉)", text):
            return {
                "memory_type": MemoryType.FACT,
                "category": "location_info",
                "key": "location_info",
                "metadata": {
                    "alternate_targets": [
                        {
                            "memory_type": MemoryType.BACKGROUND,
                            "category": "location",
                            "key": "location",
                        }
                    ]
                },
            }
        if re.search(r"(简短|简单|详细|深入|回答|风格|表格|格式)", text):
            return {
                "memory_type": MemoryType.PREFERENCE,
                "category": "response_style",
                "key": "response_style",
            }
        if re.search(r"(术语|专业|公式|太长|长度)", text):
            return {
                "memory_type": MemoryType.CONSTRAINT,
                "category": "custom",
                "key": "no_jargon" if "术语" in text else "output_length_limit",
            }
        if re.search(r"(初学|新手|入门|进阶|高级|经验|技能)", text):
            return {
                "memory_type": MemoryType.BACKGROUND,
                "category": "skill_level",
                "key": "skill_level",
            }
        return {"metadata": {"ambiguous_target": True}}

    def _implicit_window_extractions(
        self,
        window: List[Dict[str, Any]],
        signals: Dict[str, Any],
        user_message: str,
    ) -> List[ExtractionResult]:
        """从窗口重复、纠正和语言/单位/时区信号生成低置信候选。"""

        results: List[ExtractionResult] = []
        raw_content = user_message[:200]
        gate_reason = signals.get("gate_reason", "")

        for equipment in signals.get("repeated_equipment") or []:
            results.append(
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.BACKGROUND,
                    category="device_info",
                    key="device_info",
                    value=equipment,
                    confidence=0.45,
                    source_type=SourceType.AUTO,
                    is_explicit=False,
                    is_temporary=False,
                    raw_content=raw_content,
                    extraction_grade="tentative",
                    gate_reason=gate_reason,
                    metadata={"signal": "repeated_equipment"},
                )
            )

        for location in signals.get("repeated_location") or []:
            results.append(
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.BACKGROUND,
                    category="location",
                    key="location",
                    value=location,
                    confidence=0.45,
                    source_type=SourceType.AUTO,
                    is_explicit=False,
                    is_temporary=False,
                    raw_content=raw_content,
                    extraction_grade="tentative",
                    gate_reason=gate_reason,
                    metadata={"signal": "repeated_location"},
                )
            )

        for target in signals.get("correction_targets") or []:
            results.append(
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.CONSTRAINT,
                    category="custom",
                    key="data_source_constraint",
                    value=f"用户多次否定 {target} 的结果",
                    confidence=0.45,
                    source_type=SourceType.AUTO,
                    is_explicit=False,
                    is_temporary=False,
                    raw_content=raw_content,
                    extraction_grade="tentative",
                    gate_reason=gate_reason,
                    metadata={"signal": "repeated_correction", "target": target},
                )
            )

        joined = "\n".join(turn.get("user_message", "") for turn in window)
        language = self._detect_language_preference(joined)
        if language:
            results.append(
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.PREFERENCE,
                    category="language",
                    key="language",
                    value=language,
                    confidence=0.35,
                    source_type=SourceType.AUTO,
                    is_explicit=False,
                    is_temporary=False,
                    raw_content=raw_content,
                    extraction_grade="inferred",
                    gate_reason=gate_reason,
                    metadata={"signal": "language_detected"},
                )
            )

        unit_preference = self._detect_unit_preference(joined)
        if unit_preference:
            results.append(
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.PREFERENCE,
                    category="unit_preference",
                    key="unit_preference",
                    value=unit_preference,
                    confidence=0.35,
                    source_type=SourceType.AUTO,
                    is_explicit=False,
                    is_temporary=False,
                    raw_content=raw_content,
                    extraction_grade="inferred",
                    gate_reason=gate_reason,
                    metadata={"signal": "unit_detected"},
                )
            )

        timezone = self._detect_timezone(joined)
        if timezone:
            results.append(
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.FACT,
                    category="basic_info",
                    key="timezone",
                    value=timezone,
                    confidence=0.35,
                    source_type=SourceType.AUTO,
                    is_explicit=False,
                    is_temporary=False,
                    raw_content=raw_content,
                    extraction_grade="inferred",
                    gate_reason=gate_reason,
                    metadata={"signal": "timezone_detected"},
                )
            )
        return results

    def _detect_language_preference(self, text: str) -> Optional[str]:
        """从文本中识别用户偏好的回答语言。"""

        if re.search(r"(中文|用中文|Chinese)", text, re.I):
            return "中文"
        if re.search(r"(英文|英语|English)", text, re.I):
            return "英文"
        return None

    def _detect_unit_preference(self, text: str) -> Optional[str]:
        """从文本中识别用户偏好的度量单位体系。"""

        if re.search(r"(公里|千米|摄氏|°C|\bkm\b|metric)", text, re.I):
            return "metric"
        if re.search(r"(英里|华氏|mile|fahrenheit|°F)", text, re.I):
            return "imperial"
        return None

    def _detect_timezone(self, text: str) -> Optional[str]:
        """从文本中识别用户显式提到的时区。"""

        match = re.search(r"\b(?:UTC|GMT)\s*([+-]\s*\d{1,2})\b", text, re.I)
        if match:
            return f"UTC{match.group(1).replace(' ', '')}"
        if "北京时间" in text:
            return "Asia/Shanghai"
        if re.search(r"(美东|Eastern Time|ET)", text, re.I):
            return "America/New_York"
        if re.search(r"(太平洋时间|Pacific Time|PT)", text, re.I):
            return "America/Los_Angeles"
        return None

    def _finalize_fallback_results(
        self, results: List[ExtractionResult], signals: Dict[str, Any]
    ) -> List[ExtractionResult]:
        """补齐 fallback 结果的 gating 元信息和候选分级。"""

        gate_reason = signals.get("gate_reason", "")
        density_score = signals.get("density_score")
        finalized: List[ExtractionResult] = []
        seen = set()
        for result in results:
            identity = (
                result.action,
                str(result.memory_type),
                result.category,
                result.key,
                json.dumps(result.value, ensure_ascii=False, sort_keys=True)
                if not isinstance(result.value, str)
                else result.value,
            )
            if identity in seen:
                continue
            seen.add(identity)
            if not result.gate_reason:
                result.gate_reason = gate_reason
            if result.extraction_grade == "solid" and not result.is_explicit:
                result.extraction_grade = "tentative"
            result.metadata = {
                "gate_reason": result.gate_reason,
                "density_score": density_score,
                **(result.metadata or {}),
            }
            finalized.append(result)
        return finalized

    def extract_from_conversation(
        self,
        user_message: str,
        assistant_message: str,
        conversation_id: Optional[str] = None,
        conversation_window: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ExtractionResult]:
        """从一轮对话中抽取长期记忆候选，统一处理开关、门控和 fallback。"""

        if not settings.LTM_EXTRACT_ENABLED:
            return []

        window = self._normalize_conversation_window(
            user_message=user_message,
            assistant_message=assistant_message,
            conversation_id=conversation_id,
            conversation_window=conversation_window,
        )
        gating_signals = self.analyze_conversation_window(
            user_message=user_message,
            conversation_window=window,
        )
        if not self._fast_should_attempt_extraction(user_message) and not gating_signals.get("should_attempt"):
            return []

        if settings.LTM_LLM_EXTRACT_ENABLED and settings.DASHSCOPE_API_KEY:
            try:
                return self.extract_with_llm(
                    user_message,
                    assistant_message,
                    conversation_id,
                    conversation_window=window,
                    gating_signals=gating_signals,
                )
            except Exception as exc:
                logger.warning(f"LLM记忆提取失败，回退规则提取: {exc}")

        # Conservative fallback: only extract from explicit expressions
        return self._fallback_keyword_extraction(
            user_message,
            assistant_message,
            conversation_window=window,
            gating_signals=gating_signals,
        )

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
