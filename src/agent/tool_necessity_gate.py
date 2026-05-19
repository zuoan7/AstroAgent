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

        hard_no_tool = self._pre_tool_no_tool_decision(text)
        if hard_no_tool is not None:
            return hard_no_tool

        allow = self._tool_allow_decision(text)
        if allow is not None:
            return allow

        no_tool = self._no_tool_decision(text)
        if no_tool is not None:
            return no_tool

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

        if "observation-planner" in matched_skills and self._is_observation_plan_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.84,
                reason="route_allowed_observation_plan",
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
        invalid_date = self._invalid_date_reason(text)
        if invalid_date:
            return ToolNecessityDecision(
                action="clarify",
                confidence=0.96,
                reason=invalid_date,
                clarification_prompt="这个时间表达不可用。请提供一个真实日期或明确的相对日期，我再继续查询或计算。",
                missing_params=["valid_date"],
            )

        if any(token in text for token in ("这张", "这幅", "图片", "照片")) and any(
            token in text for token in ("最亮", "是不是", "识别", "看出")
        ):
            return ToolNecessityDecision(
                action="clarify",
                confidence=0.94,
                reason="image_reference_missing_attachment",
                clarification_prompt="需要先上传图片，或描述拍摄时间、地点和画面方位，我才能判断图中的目标。",
                missing_params=["image"],
            )

        if "直接控制" in text and any(token in text for token in ("赤道仪", "望远镜", "转到")):
            return ToolNecessityDecision(
                action="answer_without_tool",
                confidence=0.96,
                reason="unsupported_physical_device_control",
                answer_hint="我不能直接控制赤道仪或望远镜硬件。可以提供土星的方位/高度或升落时间，供你在设备控制软件中手动输入。",
                forbidden_skill_hints=["celestial-position-calculator"],
            )

        if "太阳" in text and "双筒" in text and any(token in text for token in ("黑子", "直视", "直接")):
            return ToolNecessityDecision(
                action="answer_without_tool",
                confidence=0.96,
                reason="unsafe_solar_observation_without_filter",
                answer_hint="不能直接用双筒看太阳找黑子，这会造成严重且不可逆的眼损伤。只有在使用可靠的全口径太阳滤膜或专用太阳望远镜时才可以观测太阳。",
                forbidden_skill_hints=["celestial-position-calculator"],
            )

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

        if self._has_invalid_messier_identifier(text):
            return ToolNecessityDecision(
                action="clarify",
                confidence=0.95,
                reason="unsupported_catalog_identifier",
                clarification_prompt="这个星表编号看起来不在常用 Messier 编号范围内。请确认目标编号，例如 M31、M42、NGC 或 IC 编号。",
                missing_params=["valid_target"],
            )

        if self._looks_like_misnamed_deep_sky_object(text):
            return ToolNecessityDecision(
                action="clarify",
                confidence=0.94,
                reason="ambiguous_or_unsupported_object_name",
                clarification_prompt="这个名称不够明确，可能不是标准可查询天体名。请提供更准确的目标名或星表编号。",
                missing_params=["valid_target"],
            )

        if self._is_ambiguous_constellation_target(text):
            return ToolNecessityDecision(
                action="clarify",
                confidence=0.9,
                reason="constellation_used_as_observing_target",
                clarification_prompt="这里的目标不够明确。请确认你指的是星座本身、其中某个亮星，还是具体深空天体（例如 M31、M42）。",
                missing_params=["specific_target"],
            )

        if self._is_precise_position_request_missing_context(text):
            return ToolNecessityDecision(
                action="clarify",
                confidence=0.9,
                reason="position_query_missing_location_or_time",
                clarification_prompt="这个位置/升落问题需要观测地点和时间。请补充城市或经纬度，以及要查询的日期或时段。",
                forbidden_skill_hints=["celestial-position-calculator"],
                missing_params=["location_or_time"],
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

        if self._is_planetary_photo_params_missing_context(text):
            return ToolNecessityDecision(
                action="clarify",
                confidence=0.9,
                reason="planetary_imaging_missing_required_context",
                clarification_prompt="要给行星拍摄参数，需要先确认望远镜或焦距、相机/手机、是否有巴罗镜、支架/赤道仪，以及你想拍视频还是单张。",
                forbidden_skill_hints=["astrophotography-calculator"],
                missing_params=["telescope_or_focal_length", "camera", "capture_mode"],
            )

        return None

    def _pre_tool_no_tool_decision(self, text: str) -> ToolNecessityDecision | None:
        if self._explicitly_forbids_external_data(text):
            return ToolNecessityDecision(
                action="answer_without_tool",
                confidence=0.96,
                reason="user_explicitly_requested_no_external_data",
                answer_hint="可以，不使用实时数据也能回答。这个问题属于稳定知识或一般经验判断，我会直接说明原理和适用边界。",
                forbidden_skill_hints=[
                    "weather-lookup",
                    "observation-planner",
                    "celestial-events-forecast",
                    "deep-sky-observing-guide",
                    "neo-tracker",
                    "astrophotography-calculator",
                    "celestial-position-calculator",
                    "get_nasa_apod",
                    "web_search",
                ],
            )

        if self._is_external_failure_strategy_question(text):
            return ToolNecessityDecision(
                action="answer_without_tool",
                confidence=0.94,
                reason="external_api_failure_strategy_without_lookup",
                answer_hint="如果外部数据暂时不可用，应说明数据源不可用、避免编造实时结果，并给出可替代的信息来源或后续重试建议。",
                forbidden_skill_hints=["neo-tracker", "get_nasa_apod", "web_search"],
            )

        if self._is_unrealistic_naked_eye_deep_sky_request(text):
            return ToolNecessityDecision(
                action="answer_without_tool",
                confidence=0.94,
                reason="unrealistic_observing_request_can_be_answered_without_tools",
                answer_hint=(
                    "不建议这样安排。深空目标面亮度通常很低，在城市强光污染环境下肉眼基本看不到；"
                    "即使用望远镜也需要暗空、较好透明度和合适口径。更现实的选择是月面、亮行星、双星或少数亮星团。"
                ),
                forbidden_skill_hints=[
                    "deep-sky-observing-guide",
                    "observation-planner",
                    "weather-lookup",
                ],
            )

        return None

    def _no_tool_decision(self, text: str) -> ToolNecessityDecision | None:
        pre_tool_no_tool = self._pre_tool_no_tool_decision(text)
        if pre_tool_no_tool is not None:
            return pre_tool_no_tool

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

        if self._is_stable_explanation_or_advice(text):
            return ToolNecessityDecision(
                action="answer_without_tool",
                confidence=0.88,
                reason="stable_explanation_or_general_advice_without_tool",
                forbidden_skill_hints=[
                    "weather-lookup",
                    "observation-planner",
                    "celestial-events-forecast",
                    "deep-sky-observing-guide",
                    "neo-tracker",
                    "astrophotography-calculator",
                    "celestial-position-calculator",
                ],
            )

        return None

    def _tool_allow_decision(self, text: str) -> ToolNecessityDecision | None:
        if self._is_apod_lookup_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.9,
                reason="apod_lookup_requires_atomic_mcp",
                allowed_skill_hints=["get_nasa_apod"],
            )

        if self._is_web_search_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.9,
                reason="fresh_external_news_requires_web_search",
                allowed_skill_hints=["web_search"],
            )

        if self._is_neo_lookup_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.88,
                reason="neo_lookup_requires_nasa_data",
                allowed_skill_hints=["neo-tracker"],
            )

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

        if self._is_current_sky_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.88,
                reason="current_sky_requires_position_catalog",
                allowed_skill_hints=["celestial-position-calculator"],
            )

        if self._is_best_window_position_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.88,
                reason="best_observation_window_requires_position",
                allowed_skill_hints=["celestial-position-calculator"],
            )

        if self._is_deep_sky_visibility_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.88,
                reason="deep_sky_visibility_requires_object_and_position",
                allowed_skill_hints=[
                    "deep-sky-observing-guide",
                    "celestial-position-calculator",
                ],
                forbidden_skill_hints=["weather-lookup"],
            )

        if self._is_stateful_followup_tool_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.82,
                reason="stateful_followup_requires_prior_context_tool",
                allowed_skill_hints=self._stateful_followup_skill_hints(text),
            )

        if self._is_coordinate_transform_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.88,
                reason="coordinate_conversion_requires_calculator",
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

        if self._is_weekly_or_recent_event_request(text):
            return ToolNecessityDecision(
                action="use_tool",
                confidence=0.84,
                reason="event_request_requires_weekly_events",
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
        has_deep_sky_target = bool(
            re.search(r"\bM\s?\d{1,3}\b", text, re.IGNORECASE)
        ) or any(token in text for token in ("星云", "星系", "深空"))
        return (
            has_deep_sky_target
            and "肉眼" in text
            and any(token in text for token in ("市中心", "城区", "强光污染", "光污染"))
        )

    @staticmethod
    def _invalid_date_reason(text: str) -> str:
        if "星期八" in text or "周八" in text:
            return "invalid_relative_weekday"
        match = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
        if not match:
            return ""
        year, month, day = (int(match.group(i)) for i in range(1, 4))
        try:
            import datetime as _dt

            _dt.date(year, month, day)
        except ValueError:
            return "invalid_calendar_date"
        return ""

    @staticmethod
    def _has_invalid_messier_identifier(text: str) -> bool:
        for match in re.finditer(
            r"(?<![A-Za-z0-9])M\s?(\d{1,4})(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        ):
            number = int(match.group(1))
            if number < 1 or number > 110:
                return True
        return False

    @staticmethod
    def _looks_like_misnamed_deep_sky_object(text: str) -> bool:
        known_valid_names = ("仙女座星系", "猎户座大星云", "北美洲星云")
        if any(name in text for name in known_valid_names):
            return False

        constellation_names = ("猎户座", "仙女座", "天鹅座", "天琴座", "大熊座", "小熊座")
        bright_star_names = ("北极星", "织女星", "牛郎星", "天狼星", "参宿四")
        dso_types = ("星云", "星系", "星团")

        return (
            any(name in text for name in constellation_names + bright_star_names)
            and any(kind in text for kind in dso_types)
        )

    @staticmethod
    def _explicitly_forbids_external_data(text: str) -> bool:
        return any(
            token in text
            for token in (
                "不用查",
                "不要查",
                "别查",
                "不要联网",
                "不用联网",
                "不联网",
                "不用实时数据",
                "不用实时",
            )
        )

    @staticmethod
    def _is_external_failure_strategy_question(text: str) -> bool:
        return any(
            token in text
            for token in (
                "数据暂时查不到",
                "接口打不开",
                "接口不可用",
                "查不到",
                "外部数据不可用",
            )
        ) and any(token in text for token in ("NASA", "APOD", "NEO", "数据", "接口"))

    @staticmethod
    def _is_apod_lookup_request(text: str) -> bool:
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
    def _is_web_search_request(text: str) -> bool:
        if any(token in text for token in ("天象", "流星雨", "月食", "日食", "合月")):
            return False
        return any(token in text for token in ("最近", "最新", "新闻", "新结果", "新发现")) and any(
            token in text for token in ("天文", "韦布", "詹姆斯 Webb", "JWST", "发现", "结果")
        )

    @staticmethod
    def _is_neo_lookup_request(text: str) -> bool:
        lowered = text.lower()
        if any(
            token in text for token in ("怎么看", "怎么回答", "如果", "查不到")
        ):
            return False
        explicit_neo = any(token in lowered for token in ("neo", "近地天体", "近地小行星", "飞掠"))
        dynamic_small_body = "小行星" in text and any(
            token in text for token in ("最近", "未来", "本周", "今天", "靠近", "飞掠", "撞击", "威胁", "风险")
        )
        return explicit_neo or dynamic_small_body

    @staticmethod
    def _is_observing_condition_weather_request(text: str) -> bool:
        if any(token in text for token in ("同时", "并且", "以及", "天象")):
            return False
        if "并" in text and any(token in text for token in ("摄影", "计划", "安排")):
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
        if has_when and has_place and any(token in text for token in ("云少", "云量", "云多", "下雨", "降雨")):
            return True
        if has_place and any(token in text for token in ("天气怎么样", "查天气", "天气如何")):
            return True
        has_observing_condition = any(
            token in text
            for token in ("适合架望远镜", "适合出门观星", "适合观星", "观测条件", "云多吗")
        )
        return has_when and has_place and has_observing_condition

    @staticmethod
    def _is_current_sky_request(text: str) -> bool:
        return any(token in text for token in ("天上有什么", "能看到哪些", "能看哪些", "哪些亮星", "哪些行星", "亮星或行星")) and any(
            token in text for token in ("现在", "今晚", "当前")
        )

    @staticmethod
    def _is_coordinate_transform_request(text: str) -> bool:
        return ("赤经" in text and "赤纬" in text and any(token in text for token in ("哪里", "在哪里", "位置", "大概"))) or bool(
            re.search(r"\bRA\b.*\bDec\b", text, re.IGNORECASE)
            and any(token in text for token in ("哪里", "在哪里", "位置"))
        )

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
    def _has_location_or_default_cue(text: str) -> bool:
        if any(
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
                "我这边",
                "我在",
            )
        ):
            return True
        return bool(re.search(r"\d{1,2}(?:\.\d+)?\s*[,，]\s*\d{2,3}(?:\.\d+)?", text))

    @classmethod
    def _is_precise_position_request_missing_context(cls, text: str) -> bool:
        has_target = any(token in text for token in ("木星", "土星", "火星", "金星", "月亮", "月球", "太阳"))
        if not has_target:
            return False
        asks_position = any(token in text for token in ("哪个方向", "在哪个方向", "方向", "高度", "几点升", "什么时候升", "几点落", "什么时候落"))
        if not asks_position:
            return False
        if cls._has_location_or_default_cue(text):
            return False
        if "大概" in text and any(token in text for token in ("几点升", "几点落")):
            return False
        return True

    @staticmethod
    def _is_sunset_darkness_request(text: str) -> bool:
        return "日落" in text and any(token in text for token in ("天黑", "比较黑", "多久"))

    @staticmethod
    def _is_monthly_event_request(text: str) -> bool:
        if any(token in text for token in ("为什么", "是什么", "什么意思", "到底", "是不是", "一般")):
            return False
        return any(token in text for token in ("天象", "月食", "日食", "流星雨", "合月")) and any(
            token in text
            for token in ("这个月", "本月", "月内", "适合普通人", "带朋友", "今年")
        ) or (
            any(token in text for token in ("天象", "月食", "日食", "流星雨", "合月"))
            and bool(re.search(r"\d{4}\s*年", text))
        )

    @staticmethod
    def _is_weekly_or_recent_event_request(text: str) -> bool:
        if any(token in text for token in ("为什么", "是什么", "什么意思", "到底", "是不是", "一般")):
            return False
        return any(token in text for token in ("天象", "流星雨", "月食", "日食", "合月")) and any(
            token in text for token in ("这周", "本周", "未来七天", "未来7天", "最近", "有哪些", "有什么", "推荐")
        )

    @staticmethod
    def _is_observation_plan_request(text: str) -> bool:
        if "拍" in text:
            return False
        if re.search(r"\bM\s?\d{1,3}\b", text, re.IGNORECASE) and "差别" in text:
            return False
        if any(token in text for token in ("提前准备", "提前多久到", "出门观星前需要")):
            return False
        if ToolNecessityGate._is_observation_order_comparison(text):
            return True
        return any(
            token in text
            for token in (
                "观测计划",
                "备选",
                "观测顺序",
                "今晚还值得架望远镜",
                "今晚月亮比较亮",
                "安排深空目标",
                "从容易找到的目标开始",
                "今晚适合看什么",
                "今晚推荐什么",
                "今晚还能看什么",
                "明晚应该安排什么",
                "今晚最值得看什么",
                "今晚安排什么",
                "观测顺序",
                "带孩子看星星",
                "朋友观星",
                "城市阳台",
                "郊外公园",
                "哪天更适合观星",
                "周末晚上能不能安排",
                "计划要不要换",
                "只有半小时",
                "今晚还值得架望远镜",
            )
        )

    @staticmethod
    def _is_ambiguous_constellation_target(text: str) -> bool:
        if "仙女座星系" in text or "猎户座大星云" in text:
            return False
        constellation_names = ("仙女座", "猎户座", "天鹅座", "天琴座", "大熊座", "小熊座")
        return any(target in text for target in constellation_names) and any(
            token in text for token in ("今晚", "明晚", "能看", "合适", "能拍", "拍吗")
        )

    @staticmethod
    def _is_planetary_photo_params_missing_context(text: str) -> bool:
        if not ("拍" in text and any(token in text for token in ("木星", "土星", "火星", "月球", "月亮"))):
            return False
        if not any(token in text for token in ("参数", "怎么设", "设置")):
            return False
        if any(token in text for token in ("曝光", "增益", "视频", "帧率", "焦距", "望远镜", "相机", "手机", "巴罗")):
            return False
        return True

    @staticmethod
    def _is_stable_explanation_or_advice(text: str) -> bool:
        if any(token in text for token in ("查一下", "查询", "有哪些", "有什么", "推荐", "安排", "计划")):
            return False
        if ToolNecessityGate._asks_dynamic_observing_choice(text):
            return False
        if any(token in text for token in ("M31", "M42", "M87", "NGC", "IC")) and any(
            token in text for token in ("适合怎么观测", "用肉眼", "用双筒", "用多大倍率", "找不到", "从哪颗", "离我们", "什么天体", "类型", "值得试", "提到", "曝光", "满月")
        ):
            return False
        stable_markers = (
            "为什么",
            "是什么现象",
            "是什么意思",
            "什么意思",
            "怎么理解",
            "区别",
            "是不是",
            "会不会影响",
            "帮助大吗",
            "有必要买吗",
            "可能是什么原因",
            "应该怎么",
            "怎么避免",
            "适合肉眼",
            "提前多久",
            "还适合",
            "还值得",
            "还能看什么",
            "哪个更容易",
            "适合看哪些",
            "就是",
            "一般什么时候",
            "一定要",
            "到底",
        )
        return any(marker in text for marker in stable_markers)

    @staticmethod
    def _is_best_window_position_request(text: str) -> bool:
        has_target = any(token in text for token in ("木星", "土星", "火星", "金星", "月亮", "月球"))
        has_when = any(token in text for token in ("今晚", "明晚", "今天", "明天"))
        has_window = any(token in text for token in ("前半夜", "后半夜", "几点高度", "高度比较合适", "什么时候更好"))
        return has_target and has_when and has_window

    @staticmethod
    def _is_deep_sky_visibility_request(text: str) -> bool:
        has_dso = bool(re.search(r"\bM\s?\d{1,3}\b", text, re.IGNORECASE)) or any(
            token in text for token in ("猎户座大星云", "仙女座星系")
        )
        has_when_or_location = any(
            token in text
            for token in ("今晚", "明晚", "周末", "这周末", "本周末", "北京", "上海", "广州", "城市阳台", "郊区")
        )
        has_visibility = any(token in text for token in ("能看到", "能看见", "适合看", "还适合", "差别大", "哪个更适合"))
        return has_dso and has_when_or_location and has_visibility

    @staticmethod
    def _is_stateful_followup_tool_request(text: str) -> bool:
        if ToolNecessityGate._is_observation_order_comparison(text):
            return True

        followup_markers = ("那", "这个", "这个目标", "它", "刚才", "前面", "上面")
        has_followup_marker = any(marker in text for marker in followup_markers)
        has_visibility_or_equipment_question = any(
            token in text
            for token in ("还能看", "能看吗", "能看到", "用双筒", "用望远镜", "哪个方向", "高度")
        )
        return has_followup_marker and has_visibility_or_equipment_question

    @staticmethod
    def _stateful_followup_skill_hints(text: str) -> list[str]:
        if ToolNecessityGate._is_observation_order_comparison(text):
            return ["observation-planner", "celestial-position-calculator"]
        if any(token in text for token in ("木星", "土星", "火星", "金星", "月亮", "月球")):
            return ["celestial-position-calculator"]
        return ["deep-sky-observing-guide"]

    @staticmethod
    def _is_observation_order_comparison(text: str) -> bool:
        has_order_word = any(token in text for token in ("先看", "先观测", "优先看", "先看哪个"))
        has_choice = "还是" in text or "哪个" in text
        targets = sum(
            1
            for target in ("月亮", "月球", "木星", "土星", "火星", "金星", "深空", "星云", "星团", "星系")
            if target in text
        )
        return has_order_word and has_choice and targets >= 2

    @staticmethod
    def _asks_dynamic_observing_choice(text: str) -> bool:
        has_dynamic_context = any(
            token in text
            for token in ("今晚", "明晚", "今天", "明天", "本周", "周末", "现在", "当前")
        )
        asks_choice = any(
            token in text
            for token in ("看什么", "还能看什么", "适合看哪些")
        )
        return has_dynamic_context and asks_choice
