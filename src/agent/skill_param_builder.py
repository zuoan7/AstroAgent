"""SkillParamBuilder — 技能参数构建工具（Phase 9 提取）。

原位于 TaskOrchestrator._build_skill_params()，提取为独立类以消除
DirectExecutor / PlannedExecutor 对 TaskOrchestrator 的循环依赖。
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.skills import registry


DEFAULT_OBSERVER_CITY = "北京"

KNOWN_CITIES = (
    "北京",
    "上海",
    "广州",
    "深圳",
    "苏州",
    "杭州",
    "成都",
    "南京",
    "武汉",
    "西安",
    "重庆",
    "天津",
    "青岛",
    "厦门",
)


class SkillParamBuilder:
    """根据 skill_name 和自然语言 query 构建调用参数。"""

    def __init__(self, skill_manager: Any) -> None:
        self._skill_manager = skill_manager

    def build(self, skill_name: str, query: str) -> Dict[str, Any]:
        from src.agent.param_parser import ParamParser

        parsed = ParamParser.parse(query)
        if self._is_structured_skill_payload(parsed, query):
            return self._finalize(skill_name, parsed)

        if skill_name == "weather-lookup":
            return self._finalize(
                skill_name,
                {
                    "city": self._extract_location(query) or DEFAULT_OBSERVER_CITY,
                    "extensions": "all",
                },
            )
        if skill_name == "observation-planner":
            return self._finalize(
                skill_name,
                {
                    "location": self._extract_location(query) or DEFAULT_OBSERVER_CITY,
                    "date": self._extract_date(query),
                },
            )
        if skill_name == "deep-sky-observing-guide":
            return self._finalize(
                skill_name,
                {
                    "target": self._extract_target(query) or query.strip(),
                    "observer_location": self._extract_location(query),
                    "date": self._extract_date(query),
                    "equipment": self._extract_equipment(query),
                },
            )
        if skill_name == "celestial-events-forecast":
            start_date, end_date = self._extract_event_range(query)
            return self._finalize(
                skill_name,
                {
                    "event_type": self._extract_event_type(query),
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        if skill_name == "astrophotography-calculator":
            return self._finalize(
                skill_name,
                {
                    "target": self._extract_photo_target(query)
                    or self._extract_target(query)
                    or query.strip(),
                    "camera": self._extract_camera(query) or "未指定相机",
                    "telescope": self._extract_telescope(query),
                    "mount": self._extract_mount(query),
                    "location": self._extract_location(query),
                    "date": self._extract_date(query),
                    "iso": self._extract_iso(query),
                    "aperture": self._extract_aperture(query),
                },
            )
        if skill_name == "celestial-position-calculator":
            return self._finalize(
                skill_name,
                {
                    "target": self._extract_target(query) or query.strip(),
                    "location": self._extract_location(query) or DEFAULT_OBSERVER_CITY,
                    "datetime": self._extract_datetime(query),
                    "output_format": self._extract_output_format(query),
                },
            )

        spec = registry.get_skill_spec(skill_name)
        fallback = (
            {spec.param_names[0]: query.strip()}
            if len(spec.param_names) == 1
            else {}
        )
        return self._finalize(skill_name, fallback)

    def _is_structured_skill_payload(self, parsed: Dict[str, Any], query: str) -> bool:
        if not isinstance(parsed, dict) or not parsed:
            return False
        if set(parsed.keys()) != {"query"}:
            return True
        return str(parsed.get("query", "")).strip() != query.strip()

    def _finalize(self, skill_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        spec = registry.get_skill_spec(skill_name)
        normalized = dict(spec.defaults or {})
        candidate = dict(params or {})
        if spec.special_handling:
            candidate = spec.special_handling(candidate)
        for name in spec.param_names:
            value = candidate.get(name)
            if value is not None:
                normalized[name] = value
        return normalized

    def _extract_location(self, query: str) -> Optional[str]:
        for city in KNOWN_CITIES:
            if city in query:
                return city
        return None

    def _extract_target(self, query: str) -> Optional[str]:
        catalog_match = re.search(
            r"(?<![A-Za-z0-9])(M\s?\d{1,3}|NGC\s?\d{1,5}|IC\s?\d{1,5})(?![A-Za-z0-9])",
            query,
            re.IGNORECASE,
        )
        if catalog_match:
            return catalog_match.group(1).upper().replace(" ", "")
        for target in (
            "仙女座星系",
            "猎户座大星云",
            "北美洲星云",
            "昴星团",
            "银河",
            "星野",
            "星轨",
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
        ):
            if target in query:
                if target == "月亮":
                    return "月球"
                return target
        if "日落" in query or "天黑" in query:
            return "太阳"
        return None

    def _extract_photo_target(self, query: str) -> Optional[str]:
        for target in (
            "银河",
            "星野",
            "星轨",
            "星空",
            "M31",
            "M42",
            "猎户座大星云",
            "仙女座星系",
            "月球",
            "月亮",
            "太阳",
            "木星",
            "土星",
            "火星",
        ):
            if target in query:
                return "月球" if target == "月亮" else target
        return None

    def _extract_date(self, query: str) -> Optional[str]:
        for token in ("今晚", "明晚", "今天", "明天", "本周末", "这个周末", "周末", "下周一"):
            if token in query:
                if token in {"这个周末", "周末"}:
                    return "本周末"
                return token
        return None

    def _extract_event_range(self, query: str) -> tuple[Optional[str], Optional[str]]:
        month_match = re.search(r"(\d{4})年(\d{1,2})月", query)
        if month_match:
            year = int(month_match.group(1))
            month = int(month_match.group(2))
            start = datetime(year, month, 1)
            if month == 12:
                end = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(year, month + 1, 1) - timedelta(days=1)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        if "本月" in query or "这个月" in query:
            today = datetime.now()
            start = today.replace(day=1)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        if any(token in query for token in ("适合普通人", "带朋友看天象", "月内天象")):
            today = datetime.now()
            start = today.replace(day=1)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        if any(token in query for token in ("未来一周", "未来7天", "本周天象", "这周天象", "一周天象")):
            start = datetime.now()
            end = start + timedelta(days=7)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        date_value = self._extract_date(query)
        return date_value, None

    def _extract_datetime(self, query: str) -> Optional[str]:
        match = re.search(r"\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2})?", query)
        if match:
            return match.group(0)
        return self._extract_date(query)

    def _extract_equipment(self, query: str) -> Optional[str]:
        for equipment in ("双筒望远镜", "双筒", "小折射镜", "8寸望远镜", "赤道仪", "固定三脚架", "三脚架"):
            if equipment in query:
                return equipment
        return None

    def _extract_camera(self, query: str) -> Optional[str]:
        for camera in ("Sony", "Canon", "Nikon", "ZWO", "QHY", "相机"):
            if camera in query:
                return camera
        return None

    def _extract_telescope(self, query: str) -> Optional[str]:
        focal_match = re.search(
            r"(\d{1,4})\s*(?:mm|毫米)\s*(?:镜头|焦距)?",
            query,
            re.IGNORECASE,
        )
        if focal_match:
            return f"{focal_match.group(1)}mm 镜头"

        lens_match = re.search(
            r"(?:焦距|镜头)\s*(\d{1,4})\s*(?:mm|毫米)",
            query,
            re.IGNORECASE,
        )
        if lens_match:
            return f"{lens_match.group(1)}mm 镜头"

        for equipment in ("小折射镜", "8寸望远镜", "双筒望远镜", "双筒"):
            if equipment in query:
                return equipment
        return None

    def _extract_mount(self, query: str) -> Optional[str]:
        if "固定三脚架" in query:
            return "固定三脚架"
        if "三脚架" in query:
            return "三脚架"
        if "赤道仪" in query:
            return "赤道仪"
        if any(token in query for token in ("跟踪", "导星")):
            return "赤道仪/跟踪支架"
        return None

    def _extract_iso(self, query: str) -> Optional[str]:
        match = re.search(r"\bISO\s*([0-9]{2,6})\b", query, re.IGNORECASE)
        if match:
            return f"ISO {match.group(1)}"
        return None

    def _extract_aperture(self, query: str) -> Optional[str]:
        match = re.search(r"\bF/?\s*(\d+(?:\.\d+)?)\b", query, re.IGNORECASE)
        if match:
            return f"f/{match.group(1)}"
        match = re.search(r"光圈\s*(\d+(?:\.\d+)?)", query)
        if match:
            return f"f/{match.group(1)}"
        return None

    def _extract_output_format(self, query: str) -> Optional[str]:
        rise_set_terms = (
            "升起",
            "落下",
            "升落",
            "几点升",
            "几点落",
            "什么时候升",
            "什么时候落",
        )
        if any(term in query for term in rise_set_terms):
            return "rise_set"

        altaz_terms = (
            "方向",
            "方位",
            "方位角",
            "高度",
            "高度角",
            "地平高度",
            "地平坐标",
            "可见",
            "能看到",
            "能看见",
            "在哪",
            "哪里",
            "altaz",
        )
        if any(term in query for term in altaz_terms):
            return "altaz"

        if "日落" in query or "天黑" in query:
            return "rise_set"

        radec_terms = ("赤经", "赤纬", "RA", "Dec", "radec", "坐标")
        if any(term in query for term in radec_terms):
            return "radec"
        return None

    def _extract_event_type(self, query: str) -> Optional[str]:
        for event_type in ("流星雨", "月食", "日食", "行星合月", "冲日", "掩星"):
            if event_type in query:
                return event_type
        return None
