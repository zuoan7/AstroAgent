from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.skills.registry import get_skill_specs
from src.agent.models.task_profile import TaskProfile
from src.agent.tool_necessity_gate import ToolNecessityDecision, ToolNecessityGate


SMALLTALK_PATTERNS = (
    r"^\s*(你好|您好|嗨|hello|hi|thanks|thank you|谢谢|辛苦了|在吗)[!！。.\s]*$",
)

SMALLTALK_PREFIXES = (
    "你好",
    "您好",
    "嗨",
    "hello",
    "hi",
)

SIMPLE_QA_HINTS = (
    "是什么",
    "什么意思",
    "解释",
    "介绍",
    "原理",
    "区别",
    "为什么",
    "如何理解",
)

COMPLEX_HINTS = (
    "比较",
    "对比",
    "分析",
    "方案",
    "步骤",
    "分阶段",
    "推导",
    "同时",
    "并且",
    "多种",
)

OPEN_ENDED_HINTS = (
    "写一篇",
    "小说",
    "故事",
    "虚构",
    "哲学",
    "随便聊聊",
    "脑洞",
    "开放式",
    "不限",
)

WEATHER_HINTS = (
    "天气",
    "云量",
    "云多",
    "多云",
    "晴不晴",
    "晴吗",
    "下雨",
    "降雨",
    "降水",
    "风大",
    "大风",
    "风力",
    "风速",
    "湿度",
    "温度",
    "气温",
    "雾霾",
    "能见度",
    "透明度",
    "视宁度",
    "结露",
    "露水",
)

OBSERVATION_RECOMMENDATION_HINTS = (
    "观测计划",
    "观测建议",
    "观测推荐",
    "观测目标",
    "今晚看什么",
    "今晚观测什么",
    "今晚能看什么",
    "今晚适合看什么",
    "适合看什么",
    "最值得看什么",
    "值得看什么",
    "看什么",
    "观测什么",
    "推荐观测",
    "推荐看",
)

CELESTIAL_EVENT_HINTS = (
    "天象",
    "流星雨",
    "月食",
    "日食",
    "合月",
    "冲日",
    "掩星",
    "凌日",
    "食甚",
    "极大",
)

DEEP_SKY_HINTS = (
    "深空",
    "星云",
    "星系",
    "星团",
    "仙女座星系",
    "猎户座大星云",
    "昴星团",
)

NEO_HINTS = (
    "近地天体",
    "近地小行星",
    "小行星",
    "neo",
    "飞掠",
    "潜在威胁",
    "撞击风险",
)

ASTROPHOTOGRAPHY_HINTS = (
    "摄影",
    "拍摄",
    "曝光",
    "叠加",
    "相机",
    "镜头",
    "焦距",
    "快门",
    "iso",
    "ISO",
    "光圈",
    "导星",
    "固定三脚架",
    "星野",
    "星轨",
)

POSITION_HINTS = (
    "位置",
    "坐标",
    "升起",
    "落下",
    "方向",
    "方位",
    "方位角",
    "高度角",
    "地平高度",
    "altaz",
    "在哪",
    "哪里",
    "可见",
    "能看到",
)

CELESTIAL_TARGET_HINTS = (
    "木星",
    "土星",
    "火星",
    "金星",
    "水星",
    "天王星",
    "海王星",
    "月球",
    "月亮",
    "太阳",
    "彗星",
    "银河",
)

ASTRONOMY_CONTEXT_HINTS = (
    "天文",
    "天象",
    "观测",
    "望远镜",
    "双筒",
    "星空",
    "星野",
    "星图",
    "亮星",
    "行星",
    "恒星",
    "星云",
    "星系",
    "星团",
    "月相",
    "目视",
)

DYNAMIC_CONTEXT_HINTS = (
    "今晚",
    "明晚",
    "今天",
    "明天",
    "本周",
    "周末",
    "现在",
    "当前",
    "实时",
    "可见",
    "能看到",
    "适合",
    "推荐",
)

TASK_TYPE_TO_OUTPUT_SCHEMA = {
    "smalltalk": "chat_answer_v1",
    "simple_qa": "qa_answer_v1",
    "clarification": "clarification_answer_v1",
    "direct_answer_no_tool": "qa_answer_v1",
    "single_tool_lookup": "tool_answer_v1",
    "observation_recommendation": "observation_answer_v1",
    "celestial_event_analysis": "event_analysis_answer_v1",
    "deep_sky_guidance": "deep_sky_answer_v1",
    "astrophotography_advice": "astrophotography_answer_v1",
    "open_domain_reasoning": "react_answer_v1",
}


@dataclass
class RouteDecision:
    """Deprecated legacy router output.

    新代码应优先消费 TaskProfile；RouteDecision 仅保留给兼容调用链：
    1. 外部兼容 API（仍直接依赖 route() / RouteDecision 的调用方）
    2. 旧 stream event 输出（仍需要 legacy route/task_type 元信息）
    3. 历史基线测试

    删除条件：
    - StreamingService 主路径不再需要 legacy route 元信息适配
    - 所有外部调用方迁移到 TaskProfile/ExecutionDecision
    - 基线/兼容测试完成退场
    """
    route: str
    task_type: str
    confidence: float
    reason: str
    matched_skills: List[str] = field(default_factory=list)
    capability_hints: List[str] = field(default_factory=list)
    expected_output_schema: str = "generic_answer_v1"
    router_source: str = "rule"
    rule_confidence: Optional[float] = None
    llm_confidence: Optional[float] = None
    tool_necessity_action: str = ""
    tool_necessity_reason: str = ""
    tool_necessity_confidence: Optional[float] = None
    answer_hint: str = ""
    clarification_prompt: str = ""
    tool_necessity_missing_params: List[str] = field(default_factory=list)
    tool_necessity_allowed_skill_hints: List[str] = field(default_factory=list)
    tool_necessity_forbidden_skill_hints: List[str] = field(default_factory=list)

    def to_meta(self) -> Dict[str, object]:
        return {
            "route": self.route,
            "task_type": self.task_type,
            "route_confidence": self.confidence,
            "route_reason": self.reason,
            "matched_skills": list(self.matched_skills),
            "capability_hints": list(self.capability_hints or self.matched_skills),
            "expected_output_schema": self.expected_output_schema,
            "router_source": self.router_source,
            "rule_confidence": self.rule_confidence,
            "llm_confidence": self.llm_confidence,
            "tool_necessity_action": self.tool_necessity_action,
            "tool_necessity_reason": self.tool_necessity_reason,
            "tool_necessity_confidence": self.tool_necessity_confidence,
            "tool_necessity_missing_params": list(self.tool_necessity_missing_params),
            "tool_necessity_allowed_skill_hints": list(
                self.tool_necessity_allowed_skill_hints
            ),
            "tool_necessity_forbidden_skill_hints": list(
                self.tool_necessity_forbidden_skill_hints
            ),
        }

    @classmethod
    def from_task_profile(cls, profile: TaskProfile) -> "RouteDecision":
        """兼容层：将 Router 主输出 TaskProfile 转换为旧 RouteDecision。"""
        return cls(
            route=profile.legacy_route,
            task_type=profile.task_type,
            confidence=profile.confidence,
            reason=profile.reason,
            matched_skills=list(profile.matched_skills),
            capability_hints=list(getattr(profile, "capability_hints", []) or []),
            expected_output_schema=profile.expected_output_schema,
            router_source=getattr(profile, "router_source", "rule"),
            rule_confidence=getattr(profile, "rule_confidence", None),
            llm_confidence=getattr(profile, "llm_confidence", None),
            tool_necessity_action=getattr(profile, "tool_necessity_action", ""),
            tool_necessity_reason=getattr(profile, "tool_necessity_reason", ""),
            tool_necessity_confidence=getattr(
                profile, "tool_necessity_confidence", None
            ),
            answer_hint=getattr(profile, "answer_hint", ""),
            clarification_prompt=getattr(profile, "clarification_prompt", ""),
            tool_necessity_missing_params=list(
                getattr(profile, "tool_necessity_missing_params", []) or []
            ),
            tool_necessity_allowed_skill_hints=list(
                getattr(profile, "tool_necessity_allowed_skill_hints", []) or []
            ),
            tool_necessity_forbidden_skill_hints=list(
                getattr(profile, "tool_necessity_forbidden_skill_hints", []) or []
            ),
        )

    @property
    def is_direct_task(self) -> bool:
        return self.route == "direct_task"

    @property
    def is_planned_task(self) -> bool:
        return self.route == "planned_task"

    @property
    def is_fallback_react(self) -> bool:
        return self.route == "fallback_react"


class RequestRouter:
    def __init__(
        self,
        llm_intent_classifier: Optional[Any] = None,
        *,
        enable_llm_fallback: Optional[bool] = None,
        llm_confidence_threshold: Optional[float] = None,
        tool_necessity_gate: Optional[ToolNecessityGate] = None,
    ) -> None:
        self._skill_specs = get_skill_specs()
        self._llm_intent_classifier = llm_intent_classifier
        self._tool_necessity_gate = tool_necessity_gate or ToolNecessityGate()
        self._enable_llm_fallback = (
            bool(getattr(settings, "ENABLE_LLM_INTENT_FALLBACK", False))
            if enable_llm_fallback is None
            else bool(enable_llm_fallback)
        )
        self._llm_confidence_threshold = (
            float(getattr(settings, "LLM_INTENT_CONFIDENCE_THRESHOLD", 0.8))
            if llm_confidence_threshold is None
            else float(llm_confidence_threshold)
        )

    def configure_llm_fallback(
        self,
        classifier: Optional[Any],
        *,
        enabled: Optional[bool] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        self._llm_intent_classifier = classifier
        if enabled is not None:
            self._enable_llm_fallback = bool(enabled)
        if confidence_threshold is not None:
            self._llm_confidence_threshold = float(confidence_threshold)

    def route(self, query: str) -> RouteDecision:
        """Deprecated compatibility entry.

        新代码应优先调用 profile() 获取 TaskProfile；route() 仅保留给：
        1. 外部兼容 API
        2. 旧 stream event 输出适配
        3. 历史测试/基线快照

        删除条件：
        - 所有主路径调用已改为 profile() / TaskProfile
        - 不再有外部调用方依赖 RouteDecision
        """
        return RouteDecision.from_task_profile(self.profile(query))

    def profile(self, query: str) -> TaskProfile:
        """Router 内部主分类入口，返回 TaskProfile。

        历史 ENABLE_TASK_PROFILE 配置位已退场，不再切换该主路径。
        """
        gate_decision = self._tool_necessity_gate.evaluate(query)
        if gate_decision.action in {"clarify", "answer_without_tool"}:
            return self._profile_from_gate_decision(gate_decision)

        rule_profile = self._rule_profile(query)
        profile = self._apply_llm_fallback(query, rule_profile)
        profile = self._apply_tool_necessity_hints(query, profile, gate_decision)

        route_gate_decision = self._tool_necessity_gate.validate_tool_route(
            query,
            list(profile.capability_hints),
        )
        if route_gate_decision.action in {"clarify", "answer_without_tool"}:
            return self._profile_from_gate_decision(route_gate_decision)
        if route_gate_decision.reason == "route_allowed":
            return profile

        return self._apply_tool_necessity_hints(query, profile, route_gate_decision)

    def _profile_from_gate_decision(
        self,
        gate_decision: ToolNecessityDecision,
    ) -> TaskProfile:
        task_type = (
            "clarification"
            if gate_decision.action == "clarify"
            else "direct_answer_no_tool"
        )
        return self._profile(
            task_type=task_type,
            legacy_route="direct_task",
            confidence=gate_decision.confidence,
            reason=f"tool_necessity_gate:{gate_decision.reason}",
            matched_skills=[],
            router_source=f"tool_necessity_gate:{gate_decision.source}",
            gate_decision=gate_decision,
        )

    def _apply_tool_necessity_hints(
        self,
        query: str,
        profile: TaskProfile,
        gate_decision: ToolNecessityDecision,
    ) -> TaskProfile:
        if gate_decision.action != "use_tool":
            return profile

        allowed = list(gate_decision.allowed_skill_hints or [])
        forbidden = set(gate_decision.forbidden_skill_hints or [])
        if not allowed and not forbidden:
            if gate_decision.reason == "no_high_confidence_gate_rule":
                return profile
            return self._with_gate_decision(profile, gate_decision)

        capability_hints = [
            capability
            for capability in (allowed or list(profile.capability_hints))
            if capability not in forbidden
        ]
        if not capability_hints:
            return self._with_gate_decision(profile, gate_decision)

        if (
            list(profile.capability_hints) == capability_hints
            and not forbidden
            and (
                len(capability_hints) != 1
                or capability_hints[0] == "observation-planner"
                or profile.legacy_route == "direct_task"
            )
        ):
            return self._with_gate_decision(profile, gate_decision)

        if len(capability_hints) == 1 and capability_hints[0] != "observation-planner":
            return self._profile(
                task_type="single_tool_lookup",
                legacy_route="direct_task",
                confidence=max(profile.confidence, gate_decision.confidence),
                reason=f"tool_necessity_gate_hint:{gate_decision.reason}",
                capability_hints=capability_hints,
                router_source=profile.router_source,
                rule_confidence=profile.rule_confidence,
                llm_confidence=profile.llm_confidence,
                gate_decision=gate_decision,
            )

        return self._profile(
            task_type=self._infer_task_type(query, capability_hints),
            legacy_route="planned_task",
            confidence=max(profile.confidence, gate_decision.confidence),
            reason=f"tool_necessity_gate_hint:{gate_decision.reason}",
            capability_hints=capability_hints,
            router_source=profile.router_source,
            rule_confidence=profile.rule_confidence,
            llm_confidence=profile.llm_confidence,
            gate_decision=gate_decision,
        )

    def _with_gate_decision(
        self,
        profile: TaskProfile,
        gate_decision: ToolNecessityDecision,
    ) -> TaskProfile:
        profile.tool_necessity_action = gate_decision.action
        profile.tool_necessity_reason = gate_decision.reason
        profile.tool_necessity_confidence = gate_decision.confidence
        profile.answer_hint = gate_decision.answer_hint
        profile.clarification_prompt = gate_decision.clarification_prompt
        profile.tool_necessity_missing_params = list(gate_decision.missing_params)
        profile.tool_necessity_allowed_skill_hints = list(
            gate_decision.allowed_skill_hints
        )
        profile.tool_necessity_forbidden_skill_hints = list(
            gate_decision.forbidden_skill_hints
        )
        return profile

    def _rule_profile(self, query: str) -> TaskProfile:
        """Deterministic rule router used as the first-pass classification."""
        text = (query or "").strip()
        lowered = text.lower()

        if self._is_smalltalk(text):
            return self._profile(
                task_type="smalltalk",
                legacy_route="direct_task",
                confidence=0.98,
                reason="matched_smalltalk_pattern",
            )

        if self._is_open_ended(text):
            return self._profile(
                task_type="open_domain_reasoning",
                legacy_route="fallback_react",
                confidence=0.58,
                reason="matched_open_ended_hint",
            )

        capability_hints = self._match_capability_hints(text, lowered)
        if capability_hints:
            if capability_hints == ["observation-planner"]:
                return self._profile(
                    task_type="observation_recommendation",
                    legacy_route="planned_task",
                    confidence=0.78,
                    reason="matched_observation_recommendation_intent",
                    capability_hints=capability_hints,
                )

            if len(capability_hints) == 1 and not self._is_complex(text):
                return self._profile(
                    task_type="single_tool_lookup",
                    legacy_route="direct_task",
                    confidence=0.9,
                    reason="matched_single_skill",
                    capability_hints=capability_hints,
                )

            if (
                len(capability_hints) == 1
                and self._should_keep_single_skill_direct(capability_hints[0], text)
            ):
                return self._profile(
                    task_type="single_tool_lookup",
                    legacy_route="direct_task",
                    confidence=0.88,
                    reason="matched_single_skill_direct_intent",
                    capability_hints=capability_hints,
                )

            return self._profile(
                task_type=self._infer_task_type(text, capability_hints),
                legacy_route="planned_task",
                confidence=0.82 if len(capability_hints) > 1 else 0.74,
                reason=(
                    "matched_multiple_skills"
                    if len(capability_hints) > 1
                    else "complex_single_skill_promoted_to_planned_task"
                ),
                capability_hints=capability_hints,
            )

        if self._is_simple_qa(text):
            return self._profile(
                task_type="simple_qa",
                legacy_route="direct_task",
                confidence=0.8,
                reason="matched_simple_qa_hint",
            )

        if self._is_complex(text):
            return self._profile(
                task_type=self._infer_task_type(text, capability_hints),
                legacy_route="planned_task",
                confidence=0.7,
                reason="matched_complex_hint",
                capability_hints=self._infer_supporting_skills(text),
            )

        return self._profile(
            task_type="open_domain_reasoning",
            legacy_route="fallback_react",
            confidence=0.45,
            reason="fallback_react_for_unclassified_query",
        )

    def _apply_llm_fallback(self, query: str, rule_profile: TaskProfile) -> TaskProfile:
        text = (query or "").strip()
        if not self._should_consult_llm_fallback(text, rule_profile):
            return rule_profile

        try:
            result = self._llm_intent_classifier.classify(
                text,
                rule_profile=rule_profile,
            )
        except Exception:
            return rule_profile

        llm_profile = self._profile_from_llm_result(text, result, rule_profile)
        return llm_profile or rule_profile

    def _should_consult_llm_fallback(
        self,
        text: str,
        rule_profile: TaskProfile,
    ) -> bool:
        if (
            not self._enable_llm_fallback
            or self._llm_intent_classifier is None
            or not text
        ):
            return False

        if rule_profile.reason in {
            "matched_smalltalk_pattern",
            "matched_open_ended_hint",
        }:
            return False

        if (
            rule_profile.capability_hints
            and rule_profile.confidence >= self._llm_confidence_threshold
        ):
            return False

        if rule_profile.task_type == "simple_qa":
            return self._has_dynamic_context(text) and self._looks_astronomy_related(
                text
            )

        if rule_profile.legacy_route == "fallback_react":
            return self._looks_astronomy_related(text)

        return (
            rule_profile.confidence < self._llm_confidence_threshold
            and self._looks_astronomy_related(text)
        )

    def _profile_from_llm_result(
        self,
        text: str,
        result: Any,
        rule_profile: TaskProfile,
    ) -> Optional[TaskProfile]:
        if result is None:
            return None

        skills = list(getattr(result, "skills", []) or [])
        requires_tool = bool(getattr(result, "requires_tool", bool(skills)))
        route = str(getattr(result, "route", "") or "")
        task_type = str(getattr(result, "task_type", "") or "")
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        reason = str(getattr(result, "reason", "") or "llm_intent_classifier")

        if requires_tool:
            if not skills or route == "fallback_react":
                return None
            if route == "direct_task" and len(skills) == 1:
                normalized_route = "direct_task"
                normalized_task_type = "single_tool_lookup"
            else:
                normalized_route = "planned_task"
                normalized_task_type = (
                    task_type
                    if task_type
                    in {
                        "observation_recommendation",
                        "celestial_event_analysis",
                        "deep_sky_guidance",
                        "astrophotography_advice",
                    }
                    else self._infer_task_type(text, skills)
                )
        else:
            if skills or route == "planned_task":
                return None
            normalized_route = (
                route if route in {"direct_task", "fallback_react"} else "direct_task"
            )
            if normalized_route == "fallback_react":
                normalized_task_type = "open_domain_reasoning"
            else:
                normalized_task_type = (
                    task_type
                    if task_type in {"smalltalk", "simple_qa"}
                    else "simple_qa"
                )

        normalized_confidence = min(max(confidence, 0.0), 0.95)
        return self._profile(
            task_type=normalized_task_type,
            legacy_route=normalized_route,
            confidence=normalized_confidence,
            reason=f"llm_intent_fallback:{reason}",
            capability_hints=skills,
            router_source="llm_fallback",
            rule_confidence=rule_profile.confidence,
            llm_confidence=normalized_confidence,
        )

    def _profile(
        self,
        *,
        task_type: str,
        legacy_route: str,
        confidence: float,
        reason: str,
        matched_skills: List[str] | None = None,
        capability_hints: List[str] | None = None,
        router_source: str = "rule",
        rule_confidence: Optional[float] = None,
        llm_confidence: Optional[float] = None,
        gate_decision: Optional[ToolNecessityDecision] = None,
    ) -> TaskProfile:
        gate = gate_decision
        return TaskProfile.from_legacy_route(
            route=legacy_route,
            task_type=task_type,
            confidence=confidence,
            reason=reason,
            matched_skills=list(matched_skills or capability_hints or []),
            capability_hints=list(capability_hints or matched_skills or []),
            expected_output_schema=TASK_TYPE_TO_OUTPUT_SCHEMA.get(
                task_type, "generic_answer_v1"
            ),
            router_source=router_source,
            rule_confidence=(
                confidence
                if rule_confidence is None and router_source == "rule"
                else rule_confidence
            ),
            llm_confidence=llm_confidence,
            tool_necessity_action=gate.action if gate else "",
            tool_necessity_reason=gate.reason if gate else "",
            tool_necessity_confidence=gate.confidence if gate else None,
            answer_hint=gate.answer_hint if gate else "",
            clarification_prompt=gate.clarification_prompt if gate else "",
            tool_necessity_missing_params=list(gate.missing_params if gate else []),
            tool_necessity_allowed_skill_hints=list(
                gate.allowed_skill_hints if gate else []
            ),
            tool_necessity_forbidden_skill_hints=list(
                gate.forbidden_skill_hints if gate else []
            ),
        )

    def _is_smalltalk(self, text: str) -> bool:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in SMALLTALK_PATTERNS):
            return True

        lowered = text.lower().strip()
        if lowered.startswith(SMALLTALK_PREFIXES) and len(text) <= 32:
            return True

        return False

    def _is_simple_qa(self, text: str) -> bool:
        if len(text) <= 24 and text.endswith(("?", "？")):
            return True
        return any(hint in text for hint in SIMPLE_QA_HINTS)

    def _is_complex(self, text: str) -> bool:
        if len(text) > 40 and any(token in text for token in ("，", ",", "；", ";")):
            return True
        return any(hint in text for hint in COMPLEX_HINTS)

    def _is_open_ended(self, text: str) -> bool:
        if any(hint in text for hint in OPEN_ENDED_HINTS):
            return True
        return len(text) > 120 and not any(
            token in text for token in ("天气", "观测", "天象", "摄影", "星云", "星系")
        )

    def _infer_task_type(self, text: str, capability_hints: List[str]) -> str:
        skill_set = set(capability_hints)
        if "observation-planner" in skill_set and any(
            token in text for token in ("计划", "安排", "推荐", "观测顺序", "看什么", "先看")
        ):
            return "observation_recommendation"
        if "astrophotography-calculator" in skill_set or "摄影" in text:
            return "astrophotography_advice"
        if "deep-sky-observing-guide" in skill_set or any(
            word in text for word in DEEP_SKY_HINTS
        ):
            return "deep_sky_guidance"
        if "celestial-events-forecast" in skill_set or any(
            word in text for word in CELESTIAL_EVENT_HINTS
        ):
            return "celestial_event_analysis"
        return "observation_recommendation"

    def _infer_supporting_skills(self, text: str) -> List[str]:
        matched = self._match_capability_hints(text, text.lower())
        if matched:
            return matched

        inferred: List[str] = []
        if self._is_weather_intent(text):
            inferred.append("weather-lookup")
        if self._is_observation_recommendation_intent(text):
            inferred.append("observation-planner")
        if any(word in text for word in CELESTIAL_EVENT_HINTS):
            inferred.append("celestial-events-forecast")
        if self._is_deep_sky_intent(text):
            inferred.append("deep-sky-observing-guide")
        if self._is_astrophotography_intent(text):
            inferred.append("astrophotography-calculator")
        if self._is_position_intent(text):
            inferred.append("celestial-position-calculator")
        return inferred

    def _match_capability_hints(self, text: str, lowered: str) -> List[str]:
        hints = self._match_skills(text, lowered)
        if self._is_apod_lookup_intent(text):
            hints.append("get_nasa_apod")
        if self._is_web_search_intent(text):
            hints.append("web_search")
        return list(dict.fromkeys(hints))

    def _match_skills(self, text: str, lowered: str) -> List[str]:
        matched: List[str] = []
        for spec in self._skill_specs:
            tokens = [
                spec.skill_name,
                spec.langchain_tool_name.lower(),
                spec.summary.lower(),
            ]
            if any(token and token in lowered for token in tokens):
                matched.append(spec.skill_name)
                continue

            skill_name = spec.skill_name
            if skill_name == "weather-lookup" and self._is_weather_intent(text):
                matched.append(skill_name)
            elif (
                skill_name == "observation-planner"
                and self._is_observation_recommendation_intent(text)
            ):
                matched.append(skill_name)
            elif skill_name == "celestial-events-forecast" and any(
                word in text for word in CELESTIAL_EVENT_HINTS
            ):
                matched.append(skill_name)
            elif (
                skill_name == "deep-sky-observing-guide"
                and self._is_deep_sky_intent(text)
            ):
                matched.append(skill_name)
            elif skill_name == "neo-tracker" and any(
                word in lowered for word in NEO_HINTS
            ):
                matched.append(skill_name)
            elif (
                skill_name == "astrophotography-calculator"
                and self._is_astrophotography_intent(text)
            ):
                matched.append(skill_name)
            elif (
                skill_name == "celestial-position-calculator"
                and self._is_position_intent(text)
            ):
                matched.append(skill_name)

        return list(dict.fromkeys(matched))

    def _is_weather_intent(self, text: str) -> bool:
        if not any(word in text for word in WEATHER_HINTS):
            return False
        if any(token in text for token in ("预报说", "天气晴但", "湿度很高", "视宁度不好")):
            return False
        has_city = any(city in text for city in ("北京", "上海", "广州", "深圳", "杭州", "成都", "南京", "武汉", "西安", "重庆", "天津"))
        has_time = any(token in text for token in ("今晚", "明晚", "今天", "明天", "当前", "现在"))
        asks_lookup = any(token in text for token in ("云多吗", "云少", "会不会下雨", "适合架望远镜", "适合出门观星", "适合观星", "观测条件"))
        if has_city and (
            any(token in text for token in ("天气怎么样", "查天气", "天气如何", "今天天气"))
            or ("查" in text and "天气" in text)
        ):
            return True
        return has_city and has_time and asks_lookup

    def _is_observation_recommendation_intent(self, text: str) -> bool:
        if "拍" in text:
            return False
        if re.search(r"\bM\s?\d{1,3}\b", text, re.IGNORECASE) and "差别" in text:
            return False
        if any(token in text for token in ("提前准备", "提前多久到", "出门观星前需要")):
            return False
        if self._is_observation_ordering_intent(text):
            return True
        if any(word in text for word in OBSERVATION_RECOMMENDATION_HINTS):
            return True
        if self._has_observing_equipment_constraint(text) and any(
            token in text for token in ("看什么", "安排", "目标", "推荐", "观测顺序")
        ):
            return True
        if any(
            token in text
            for token in (
                "今晚推荐什么",
                "明晚应该安排什么",
                "今晚安排什么",
                "计划要不要换",
                "周末晚上能不能安排",
                "带孩子看星星",
                "朋友观星",
                "不容易冷场",
                "城市阳台",
                "郊外公园",
                "哪天更适合观星",
                "只有半小时",
                "观测顺序",
                "从容易找到的目标开始",
            )
        ):
            return True
        return any(
            re.search(pattern, text)
            for pattern in (
                r"(今晚|明晚|今天|明天|本周|周末).*(观测|看).*(目标|推荐|什么|哪些)",
                r"(推荐|安排|规划).*(观测|看).*(目标|清单|列表)?",
            )
        )

    def _is_deep_sky_intent(self, text: str) -> bool:
        if any(token in text for token in ("区别", "是不是一个", "同一个", "为什么更适合", "会不会影响", "应该先选", "基本没戏", "帮助大吗")):
            return False
        has_catalog = bool(
            re.search(r"\b(M\d{1,3}|NGC\s?\d{1,5}|IC\s?\d{1,5})\b", text, re.IGNORECASE)
        )
        if has_catalog:
            return True
        if any(word in text for word in ("仙女座星系", "猎户座大星云", "昴星团", "银河系")):
            return True
        if any(word in text for word in DEEP_SKY_HINTS):
            return any(
                token in text
                for token in (
                    "观测",
                    "用肉眼",
                    "用双筒",
                    "用多大倍率",
                    "找",
                    "看出",
                    "适合看",
                    "能看到",
                    "值得试",
                    "离我们",
                    "什么天体",
                    "类型",
                    "提到",
                    "差别大",
                )
            )
        return False

    def _is_astrophotography_intent(self, text: str) -> bool:
        if any(token in text for token in ("怎么避免", "直接拍单张", "规则现在", "怎么对焦", "有必要买吗", "一定要拍", "可能是哪一步", "分别是干什么")):
            return False
        if any(word in text for word in ASTROPHOTOGRAPHY_HINTS):
            if any(token in text for token in ("曝光", "增益", "单张", "累计", "流程", "安排", "计划", "架机", "收片", "参数", "快门")):
                return True
            if re.search(r"\d{1,4}\s*(?:mm|毫米)", text):
                return True
            return False
        if "参数思路" in text:
            return True
        return bool(
            re.search(
                r"拍(银河|星野|星轨|星空|月亮|月球|太阳|木星|土星|火星|星云|星系|星团|彗星)",
                text,
            )
        )

    def _is_position_intent(self, text: str) -> bool:
        if self._is_current_sky_intent(text) or self._is_coordinate_transform_intent(text):
            return True
        if any(token in text for token in ("怎么理解", "这是赤经赤纬吗", "只有 12 度高度")):
            return False
        return self._has_celestial_target(text) and any(word in text for word in POSITION_HINTS)

    def _has_celestial_target(self, text: str) -> bool:
        return any(
            word in text for word in CELESTIAL_TARGET_HINTS
        ) or self._is_deep_sky_intent(text)

    def _has_dynamic_context(self, text: str) -> bool:
        return any(word in text for word in DYNAMIC_CONTEXT_HINTS)

    def _looks_astronomy_related(self, text: str) -> bool:
        if self._has_celestial_target(text):
            return True
        if any(word in text for word in ASTRONOMY_CONTEXT_HINTS):
            return True
        if self._is_weather_intent(text) or self._is_astrophotography_intent(text):
            return True
        if any(word in text for word in CELESTIAL_EVENT_HINTS + NEO_HINTS):
            return True
        return bool(
            re.search(
                r"\b(M\d{1,3}|NGC\s?\d{1,5}|IC\s?\d{1,5})\b",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_apod_lookup_intent(text: str) -> bool:
        has_apod_reference = "APOD" in text or "每日天文图" in text
        if not has_apod_reference:
            return False
        has_lookup_context = any(
            token in text for token in ("今天", "今日", "昨天", "日期", "查", "查询", "图片")
        ) or bool(re.search(r"\d{4}-\d{2}-\d{2}", text))
        if any(token in text for token in ("什么意思", "怎么理解")):
            return False
        if "是什么" in text and not has_lookup_context:
            return False
        return has_lookup_context

    @staticmethod
    def _is_web_search_intent(text: str) -> bool:
        if any(token in text for token in ("天象", "流星雨", "月食", "日食", "合月")):
            return False
        return any(token in text for token in ("最近", "最新", "新闻", "新结果", "新发现")) and any(
            token in text for token in ("天文", "韦布", "JWST", "发现", "结果")
        )

    @staticmethod
    def _is_current_sky_intent(text: str) -> bool:
        return any(token in text for token in ("天上有什么", "能看到哪些", "能看哪些", "哪些亮星", "亮星或行星")) and any(
            token in text for token in ("现在", "今晚", "当前")
        )

    @staticmethod
    def _is_coordinate_transform_intent(text: str) -> bool:
        return ("赤经" in text and "赤纬" in text and any(token in text for token in ("哪里", "在哪里", "位置", "大概"))) or bool(
            re.search(r"\bRA\b.*\bDec\b", text, re.IGNORECASE)
            and any(token in text for token in ("哪里", "在哪里", "位置"))
        )

    @staticmethod
    def _is_observation_ordering_intent(text: str) -> bool:
        has_order_word = any(token in text for token in ("先看", "先观测", "优先看", "先看哪个"))
        has_choice = "还是" in text or "哪个" in text
        target_count = sum(
            1
            for target in ("月亮", "月球", "木星", "土星", "火星", "金星", "深空", "星云", "星团", "星系")
            if target in text
        )
        return has_order_word and has_choice and target_count >= 2

    @staticmethod
    def _has_observing_equipment_constraint(text: str) -> bool:
        return bool(re.search(r"\b\d{1,2}\s*x\s*\d{2}\b", text, re.IGNORECASE)) or any(
            token in text for token in ("双筒", "望远镜", "DOB", "小折射", "目镜")
        )

    def _should_keep_single_skill_direct(self, skill_name: str, text: str) -> bool:
        if skill_name != "astrophotography-calculator":
            return False

        if self._is_weather_intent(text):
            return False
        if any(
            word in text
            for word in ("方案", "计划", "分析", "步骤", "分阶段", "同时", "并且", "对比")
        ):
            return False
        if "比较" in text and not any(
            phrase in text for phrase in ("比较稳", "比较好", "比较合适", "比较安全")
        ):
            return False
        return True
