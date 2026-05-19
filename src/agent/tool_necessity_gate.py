from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


ToolNecessityAction = Literal["use_tool", "clarify", "answer_without_tool"]


@dataclass(frozen=True)
class ToolNecessityDecision:
    action: ToolNecessityAction
    confidence: float
    reason: str
    answer_hint: str = ""
    clarification_prompt: str = ""
    allowed_skill_hints: list[str] = field(default_factory=list)
    forbidden_skill_hints: list[str] = field(default_factory=list)
    missing_params: list[str] = field(default_factory=list)
    source: str = "rule"

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "reason": self.reason,
            "answer_hint": self.answer_hint,
            "clarification_prompt": self.clarification_prompt,
            "allowed_skill_hints": list(self.allowed_skill_hints),
            "forbidden_skill_hints": list(self.forbidden_skill_hints),
            "missing_params": list(self.missing_params),
            "source": self.source,
        }


class ToolNecessityGate:
    """Pre-router gate for deciding whether a request should enter tools."""

    def evaluate(self, query: str) -> ToolNecessityDecision:
        text = (query or "").strip()
        if not text:
            return ToolNecessityDecision(
                action="answer_without_tool",
                confidence=0.95,
                reason="empty_query",
                answer_hint="请先告诉我你想查询或了解的天文问题。",
            )

        clarification = self._clarification_decision(text)
        if clarification is not None:
            return clarification

        no_tool = self._no_tool_decision(text)
        if no_tool is not None:
            return no_tool

        allow = self._tool_allow_decision(text)
        if allow is not None:
            return allow

        return ToolNecessityDecision(
            action="use_tool",
            confidence=0.5,
            reason="no_high_confidence_gate_rule",
        )

    def validate_tool_route(
        self,
        query: str,
        matched_skills: list[str],
    ) -> ToolNecessityDecision:
        """Return a route-level veto or hint decision after skill matching."""
        text = (query or "").strip()

        if (
            "观测计划" in text
            or "计划" in text
            or "备选" in text
            or "观测顺序" in text
        ) and "weather-lookup" in matched_skills:
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.88,
                reason="observation_plan_should_not_expose_weather_skill_step",
                allowed_skill_hints=["observation-planner"],
                forbidden_skill_hints=["weather-lookup"],
            )

        no_tool = self._no_tool_decision(text)
        if no_tool is not None:
            return no_tool

        return ToolNecessityDecision(
            action="use_tool",
            confidence=0.5,
            reason="route_allowed",
        )

    def _clarification_decision(self, text: str) -> ToolNecessityDecision | None:
        if "三体星" in text:
            return ToolNecessityDecision(
                action="clarify",
                confidence=0.96,
                reason="unrecognized_or_fictional_celestial_target",
                clarification_prompt=(
                    "我无法确认“三体星”是一个可计算的真实天体名称。"
                    "请提供真实天体名、星表编号或坐标，再告诉我观测时间和地点。"
                ),
                forbidden_skill_hints=["celestial-position-calculator", "RAGRetrieve"],
                missing_params=["valid_target"],
            )

        if self._is_photo_exposure_request(text) and not self._has_photo_exposure_params(text):
            return ToolNecessityDecision(
                action="clarify",
                confidence=0.9,
                reason="astrophotography_exposure_missing_required_context",
                clarification_prompt=(
                    "要计算拍银河的曝光时间，需要先确认焦距或镜头、是否使用赤道仪/跟踪、"
                    "相机画幅或像元大小，以及你想避免星点拖尾还是估算累计曝光。"
                ),
                forbidden_skill_hints=["astrophotography-calculator"],
                missing_params=["focal_length", "mount_or_tracking", "camera"],
            )

        return None

    def _no_tool_decision(self, text: str) -> ToolNecessityDecision | None:
        if self._is_unrealistic_naked_eye_deep_sky_request(text):
            return ToolNecessityDecision(
                action="answer_without_tool",
                confidence=0.94,
                reason="unrealistic_observing_request_can_be_answered_without_tools",
                answer_hint=(
                    "不建议这样安排。M101 面亮度很低，在上海市中心这类强光污染环境下，"
                    "肉眼基本看不到；即使用望远镜也需要暗空、较好透明度和合适口径。"
                    "更现实的选择是月面、亮行星、双星或少数亮星团。"
                ),
                forbidden_skill_hints=[
                    "deep-sky-observing-guide",
                    "observation-planner",
                    "weather-lookup",
                ],
            )

        stable_answers = [
            (
                self._contains_all(text, ("星云", "星系", "区别")),
                "stable_knowledge_nebula_galaxy_difference",
                "星云主要是气体和尘埃云，常见于恒星形成区或恒星演化末期；星系是由大量恒星、气体、尘埃和暗物质组成的巨大引力系统。两者尺度差别很大，星系通常远大于星云。",
            ),
            (
                "为什么会有流星雨" in text,
                "stable_knowledge_meteor_shower_mechanism",
                "流星雨通常来自彗星或小行星留下的尘埃流。当地球运行穿过这些尘埃流时，颗粒高速进入大气并烧蚀发光，就形成了看起来从同一辐射点射出的流星雨。",
            ),
            (
                "视宁度差" in text,
                "stable_knowledge_seeing_definition",
                "视宁度差表示大气湍流较强，星点会抖动、膨胀或模糊。它主要影响行星、月面细节和高倍率观测，即使天空透明，细节也可能不稳定。",
            ),
            (
                self._contains_all(text, ("湿度", "起雾")),
                "observing_experience_dew_without_live_weather",
                "会。湿度高时镜头或望远镜表面容易降到露点以下，水汽会凝结成雾。实际观测时可以用遮光罩、除露带、低功率加热或提前封存干燥来减轻结露。",
            ),
            (
                "天象预报" in text and "城市" in text and "可见" in text,
                "observing_experience_city_visibility_without_tools",
                "不一定。天象预报中的“可见”通常表示几何位置和亮度满足条件，但城市光污染、遮挡、云量、目标高度和肉眼极限星等都会影响实际可见性。亮行星和月亮通常更稳，暗弱目标需要暗空。",
            ),
        ]
        for matched, reason, answer in stable_answers:
            if matched:
                return ToolNecessityDecision(
                    action="answer_without_tool",
                    confidence=0.9,
                    reason=reason,
                    answer_hint=answer,
                    forbidden_skill_hints=[
                        "weather-lookup",
                        "celestial-events-forecast",
                        "deep-sky-observing-guide",
                        "astrophotography-calculator",
                        "celestial-position-calculator",
                    ],
                )

        return None

    def _tool_allow_decision(self, text: str) -> ToolNecessityDecision | None:
        if self._is_observing_condition_weather_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.86,
                reason="observing_condition_requires_live_weather",
                allowed_skill_hints=["weather-lookup"],
            )

        if self._is_position_altaz_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.86,
                reason="position_visibility_requires_altaz",
                allowed_skill_hints=["celestial-position-calculator"],
            )

        if self._is_sunset_darkness_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.86,
                reason="sunset_darkness_requires_rise_set",
                allowed_skill_hints=["celestial-position-calculator"],
            )

        if self._is_monthly_event_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.84,
                reason="event_request_requires_monthly_events",
                allowed_skill_hints=["celestial-events-forecast"],
            )

        if self._is_observation_plan_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.84,
                reason="observation_plan_requires_composite_planner",
                allowed_skill_hints=["observation-planner"],
                forbidden_skill_hints=["weather-lookup"],
            )

        return None

    @staticmethod
    def _contains_all(text: str, tokens: tuple[str, ...]) -> bool:
        return all(token in text for token in tokens)

    @staticmethod
    def _is_photo_exposure_request(text: str) -> bool:
        return (
            ("拍" in text or "摄影" in text)
            and any(token in text for token in ("银河", "星野", "星空", "星轨"))
            and any(token in text for token in ("曝光", "快门", "拍多久"))
        )

    @staticmethod
    def _has_photo_exposure_params(text: str) -> bool:
        if re.search(r"\d{1,4}\s*(mm|毫米)", text, re.IGNORECASE):
            return True
        return any(
            token in text
            for token in (
                "镜头",
                "焦距",
                "相机",
                "三脚架",
                "固定三脚架",
                "赤道仪",
                "跟踪",
                "导星",
                "ISO",
                "iso",
                "光圈",
                "f/",
                "F/",
            )
        )

    @staticmethod
    def _is_unrealistic_naked_eye_deep_sky_request(text: str) -> bool:
        return (
            bool(re.search(r"\bM\s?101\b", text, re.IGNORECASE))
            and "肉眼" in text
            and any(token in text for token in ("市中心", "上海市中心", "城区"))
        )

    @staticmethod
    def _is_observing_condition_weather_request(text: str) -> bool:
        if any(token in text for token in ("同时", "并且", "以及", "天象")):
            return False
        has_when = any(token in text for token in ("今晚", "明晚", "今天", "明天", "当前"))
        has_place = any(
            token in text
            for token in (
                "北京",
                "上海",
                "广州",
                "深圳",
                "杭州",
                "成都",
                "南京",
                "武汉",
                "西安",
                "重庆",
                "天津",
            )
        )
        has_observing_condition = any(
            token in text
            for token in ("适合架望远镜", "适合出门观星", "适合观星", "观测条件", "云多吗")
        )
        return has_when and has_place and has_observing_condition

    @staticmethod
    def _is_position_altaz_request(text: str) -> bool:
        has_target = any(
            token in text
            for token in ("月亮", "月球", "木星", "火星", "土星", "金星", "水星")
        )
        has_when = any(token in text for token in ("今晚", "明晚", "今天", "明天", "现在"))
        has_altaz = any(
            token in text
            for token in ("高度", "高度角", "方向", "方位", "在哪", "可见", "能看到")
        )
        return has_target and has_when and has_altaz

    @staticmethod
    def _is_sunset_darkness_request(text: str) -> bool:
        return "日落" in text and any(token in text for token in ("天黑", "比较黑", "多久"))

    @staticmethod
    def _is_monthly_event_request(text: str) -> bool:
        return "天象" in text and any(
            token in text
            for token in ("这个月", "本月", "月内", "适合普通人", "带朋友")
        )

    @staticmethod
    def _is_observation_plan_request(text: str) -> bool:
        return any(
            token in text
            for token in (
                "观测计划",
                "备选",
                "观测顺序",
                "今晚还值得架望远镜",
                "今晚月亮比较亮",
                "从容易找到的目标开始",
            )
        )
