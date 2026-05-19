from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.skills.registry import get_skill_specs
from src.agent.models.task_profile import TaskProfile


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

TASK_TYPE_TO_OUTPUT_SCHEMA = {
    "smalltalk": "chat_answer_v1",
    "simple_qa": "qa_answer_v1",
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
    3. TaskOrchestrator 兼容层
    4. 历史基线测试

    删除条件：
    - StreamingService 主路径不再需要 legacy route 元信息适配
    - TaskOrchestrator 及所有外部调用方迁移到 TaskProfile/ExecutionDecision
    - 基线/兼容测试完成退场
    """
    route: str
    task_type: str
    confidence: float
    reason: str
    matched_skills: List[str] = field(default_factory=list)
    expected_output_schema: str = "generic_answer_v1"

    def to_meta(self) -> Dict[str, object]:
        return {
            "route": self.route,
            "task_type": self.task_type,
            "route_confidence": self.confidence,
            "route_reason": self.reason,
            "matched_skills": list(self.matched_skills),
            "expected_output_schema": self.expected_output_schema,
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
            expected_output_schema=profile.expected_output_schema,
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
    def __init__(self) -> None:
        self._skill_specs = get_skill_specs()

    def route(self, query: str) -> RouteDecision:
        """Deprecated compatibility entry.

        新代码应优先调用 profile() 获取 TaskProfile；route() 仅保留给：
        1. 外部兼容 API
        2. 旧 stream event 输出适配
        3. TaskOrchestrator 兼容层
        4. 历史测试/基线快照

        删除条件：
        - 所有主路径调用已改为 profile() / TaskProfile
        - 不再有外部调用方依赖 RouteDecision
        - TaskOrchestrator 兼容层完成清理
        """
        return RouteDecision.from_task_profile(self.profile(query))

    def profile(self, query: str) -> TaskProfile:
        """Router 内部主分类入口，返回 TaskProfile。

        ENABLE_TASK_PROFILE 配置位仅为历史兼容保留，不再切换该主路径。
        """
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

        matched_skills = self._match_skills(text, lowered)
        if matched_skills:
            if matched_skills == ["observation-planner"]:
                return self._profile(
                    task_type="observation_recommendation",
                    legacy_route="planned_task",
                    confidence=0.78,
                    reason="matched_observation_recommendation_intent",
                    matched_skills=matched_skills,
                )

            if len(matched_skills) == 1 and not self._is_complex(text):
                return self._profile(
                    task_type="single_tool_lookup",
                    legacy_route="direct_task",
                    confidence=0.9,
                    reason="matched_single_skill",
                    matched_skills=matched_skills,
                )

            if (
                len(matched_skills) == 1
                and self._should_keep_single_skill_direct(matched_skills[0], text)
            ):
                return self._profile(
                    task_type="single_tool_lookup",
                    legacy_route="direct_task",
                    confidence=0.88,
                    reason="matched_single_skill_direct_intent",
                    matched_skills=matched_skills,
                )

            return self._profile(
                task_type=self._infer_task_type(text, matched_skills),
                legacy_route="planned_task",
                confidence=0.82 if len(matched_skills) > 1 else 0.74,
                reason=(
                    "matched_multiple_skills"
                    if len(matched_skills) > 1
                    else "complex_single_skill_promoted_to_planned_task"
                ),
                matched_skills=matched_skills,
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
                task_type=self._infer_task_type(text, matched_skills),
                legacy_route="planned_task",
                confidence=0.7,
                reason="matched_complex_hint",
                matched_skills=self._infer_supporting_skills(text),
            )

        return self._profile(
            task_type="open_domain_reasoning",
            legacy_route="fallback_react",
            confidence=0.45,
            reason="fallback_react_for_unclassified_query",
        )

    def _profile(
        self,
        *,
        task_type: str,
        legacy_route: str,
        confidence: float,
        reason: str,
        matched_skills: List[str] | None = None,
    ) -> TaskProfile:
        return TaskProfile.from_legacy_route(
            route=legacy_route,
            task_type=task_type,
            confidence=confidence,
            reason=reason,
            matched_skills=list(matched_skills or []),
            expected_output_schema=TASK_TYPE_TO_OUTPUT_SCHEMA.get(
                task_type, "generic_answer_v1"
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

    def _infer_task_type(self, text: str, matched_skills: List[str]) -> str:
        skill_set = set(matched_skills)
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
        matched = self._match_skills(text, text.lower())
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
        return any(word in text for word in WEATHER_HINTS)

    def _is_observation_recommendation_intent(self, text: str) -> bool:
        if any(word in text for word in OBSERVATION_RECOMMENDATION_HINTS):
            return True
        return any(
            re.search(pattern, text)
            for pattern in (
                r"(今晚|明晚|今天|明天|本周|周末).*(观测|看).*(目标|推荐|什么|哪些)",
                r"(推荐|安排|规划).*(观测|看).*(目标|清单|列表)?",
            )
        )

    def _is_deep_sky_intent(self, text: str) -> bool:
        return any(word in text for word in DEEP_SKY_HINTS) or bool(
            re.search(
                r"\b(M\d{1,3}|NGC\s?\d{1,5}|IC\s?\d{1,5})\b",
                text,
                re.IGNORECASE,
            )
        )

    def _is_astrophotography_intent(self, text: str) -> bool:
        if any(word in text for word in ASTROPHOTOGRAPHY_HINTS):
            return True
        return bool(
            re.search(
                r"拍(银河|星野|星轨|星空|月亮|月球|太阳|木星|土星|火星|星云|星系|星团|彗星)",
                text,
            )
        )

    def _is_position_intent(self, text: str) -> bool:
        return self._has_celestial_target(text) and any(
            word in text for word in POSITION_HINTS
        )

    def _has_celestial_target(self, text: str) -> bool:
        return any(
            word in text for word in CELESTIAL_TARGET_HINTS
        ) or self._is_deep_sky_intent(text)

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
