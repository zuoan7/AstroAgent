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
            if len(matched_skills) == 1 and not self._is_complex(text):
                return self._profile(
                    task_type="single_tool_lookup",
                    legacy_route="direct_task",
                    confidence=0.9,
                    reason="matched_single_skill",
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
            word in text for word in ("深空", "星云", "星系", "星团")
        ):
            return "deep_sky_guidance"
        if "celestial-events-forecast" in skill_set or any(
            word in text for word in ("天象", "流星雨", "月食", "日食")
        ):
            return "celestial_event_analysis"
        return "observation_recommendation"

    def _infer_supporting_skills(self, text: str) -> List[str]:
        matched = self._match_skills(text, text.lower())
        if matched:
            return matched

        inferred: List[str] = []
        if any(word in text for word in ("天气", "云量", "湿度")):
            inferred.append("weather-lookup")
        if any(word in text for word in ("观测", "今晚看什么", "观测计划")):
            inferred.append("observation-planner")
        if any(word in text for word in ("天象", "流星雨", "月食", "日食")):
            inferred.append("celestial-events-forecast")
        if any(word in text for word in ("深空", "星云", "星系", "星团")):
            inferred.append("deep-sky-observing-guide")
        if any(word in text for word in ("摄影", "曝光", "相机")):
            inferred.append("astrophotography-calculator")
        if any(word in text for word in ("位置", "坐标", "升起", "落下")):
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
            if skill_name == "weather-lookup" and any(
                word in text for word in ("天气", "云量", "湿度")
            ):
                matched.append(skill_name)
            elif skill_name == "observation-planner" and any(
                word in text for word in ("观测计划", "观测建议", "今晚看什么", "适合看什么")
            ):
                matched.append(skill_name)
            elif skill_name == "celestial-events-forecast" and any(
                word in text for word in ("天象", "流星雨", "月食", "日食")
            ):
                matched.append(skill_name)
            elif skill_name == "deep-sky-observing-guide" and any(
                word in text for word in ("深空", "星云", "星系", "星团")
            ):
                matched.append(skill_name)
            elif skill_name == "neo-tracker" and any(
                word in text for word in ("近地天体", "小行星", "neo")
            ):
                matched.append(skill_name)
            elif skill_name == "astrophotography-calculator" and any(
                word in text for word in ("摄影", "曝光", "叠加", "相机")
            ):
                matched.append(skill_name)
            elif skill_name == "celestial-position-calculator" and any(
                word in text for word in ("位置", "坐标", "升起", "落下")
            ):
                matched.append(skill_name)

        return list(dict.fromkeys(matched))
