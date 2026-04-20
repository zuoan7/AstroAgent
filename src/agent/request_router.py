from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.skills.registry import get_skill_specs


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


@dataclass
class RouteDecision:
    route: str
    confidence: float
    reason: str
    matched_skills: List[str] = field(default_factory=list)

    def to_meta(self) -> Dict[str, object]:
        return {
            "route": self.route,
            "route_confidence": self.confidence,
            "route_reason": self.reason,
            "matched_skills": list(self.matched_skills),
        }


class RequestRouter:
    def __init__(self) -> None:
        self._skill_specs = get_skill_specs()

    def route(self, query: str) -> RouteDecision:
        text = (query or "").strip()
        lowered = text.lower()

        if self._is_smalltalk(text):
            return RouteDecision("smalltalk", 0.98, "matched_smalltalk_pattern")

        matched_skills = self._match_skills(text, lowered)
        if matched_skills:
            if len(matched_skills) == 1:
                return RouteDecision(
                    "tool_task",
                    0.9,
                    "matched_single_skill",
                    matched_skills=matched_skills,
                )
            return RouteDecision(
                "complex_agent",
                0.7,
                "matched_multiple_skills",
                matched_skills=matched_skills,
            )

        if self._is_complex(text):
            return RouteDecision("complex_agent", 0.75, "matched_complex_hint")

        if self._is_simple_qa(text):
            return RouteDecision("simple_qa", 0.8, "matched_simple_qa_hint")

        return RouteDecision("complex_agent", 0.5, "fallback_complex_agent")

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
            if skill_name == "weather-lookup" and any(word in text for word in ("天气", "云量", "湿度")):
                matched.append(skill_name)
            elif skill_name == "observation-planner" and any(word in text for word in ("观测计划", "观测建议", "今晚看什么")):
                matched.append(skill_name)
            elif skill_name == "celestial-events-forecast" and any(word in text for word in ("天象", "流星雨", "月食", "日食")):
                matched.append(skill_name)
            elif skill_name == "deep-sky-observing-guide" and any(word in text for word in ("深空", "星云", "星系", "星团")):
                matched.append(skill_name)
            elif skill_name == "neo-tracker" and any(word in text for word in ("近地天体", "小行星", "neo")):
                matched.append(skill_name)
            elif skill_name == "astrophotography-calculator" and any(word in text for word in ("摄影", "曝光", "叠加", "相机")):
                matched.append(skill_name)
            elif skill_name == "celestial-position-calculator" and any(word in text for word in ("位置", "坐标", "升起", "落下")):
                matched.append(skill_name)
        return matched
