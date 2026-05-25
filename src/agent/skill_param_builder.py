"""SkillParamBuilder — 技能参数构建工具。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.agent.fast_answers import extract_latest_location, extract_latest_target
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

    def __init__(self, capability_provider: Any) -> None:
        self._capability_provider = capability_provider

    def build(
        self,
        skill_name: str,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
    ) -> Dict[str, Any]:
        from src.utils.param_parser import ParamParser

        context_text = self._context_text(chat_history, user_profile)
        query_location = self._extract_location(query)
        context_location = extract_latest_location(context_text)
        query_target = self._extract_target(query)
        context_target = extract_latest_target(context_text)

        parsed = ParamParser.parse(query)
        if self._is_structured_skill_payload(parsed, query):
            return self._finalize(skill_name, parsed)

        if skill_name == "weather-lookup":
            return self._finalize(
                skill_name,
                {
                    "city": query_location or context_location or DEFAULT_OBSERVER_CITY,
                    "extensions": "all",
                },
            )
        if skill_name == "observation-planner":
            return self._finalize(
                skill_name,
                {
                    "location": query_location
                    or context_location
                    or DEFAULT_OBSERVER_CITY,
                    "date": self._extract_date(query),
                },
            )
        if skill_name == "deep-sky-observing-guide":
            return self._finalize(
                skill_name,
                {
                    "target": query_target or context_target or query.strip(),
                    "observer_location": query_location or context_location,
                    "date": self._extract_date(query),
                    "equipment": self._extract_equipment(query),
                },
            )
        if skill_name == "neo-tracker":
            min_size, max_distance = self._extract_neo_filters(query)
            return self._finalize(
                skill_name,
                {
                    "time_range": self._extract_date(query)
                    or self._extract_neo_time_range(query),
                    "min_size": min_size,
                    "max_distance": max_distance,
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
                    "operation": self._extract_event_operation(
                        query, start_date, end_date
                    ),
                },
            )
        if skill_name == "astrophotography-calculator":
            return self._finalize(
                skill_name,
                {
                    "target": self._extract_photo_target(query)
                    or query_target
                    or context_target
                    or query.strip(),
                    "camera": self._extract_camera(query) or "未指定相机",
                    "telescope": self._extract_telescope(query),
                    "mount": self._extract_mount(query),
                    "location": query_location or context_location,
                    "date": self._extract_date(query),
                    "iso": self._extract_iso(query),
                    "aperture": self._extract_aperture(query),
                },
            )
        if skill_name == "celestial-position-calculator":
            ra, dec = self._extract_radec(query)
            return self._finalize(
                skill_name,
                {
                    "target": query_target or context_target or query.strip(),
                    "location": query_location
                    or context_location
                    or DEFAULT_OBSERVER_CITY,
                    "datetime": self._extract_datetime(query),
                    "output_format": self._extract_output_format(query),
                    "operation": self._extract_position_operation(query),
                    "ra": ra,
                    "dec": dec,
                    "epoch": "J2000" if ra is not None and dec is not None else None,
                    "target_system": (
                        "fk5" if ra is not None and dec is not None else None
                    ),
                },
            )

        definition = registry.get_skill_definition(skill_name)
        fields = definition.input_field_names
        fallback = {fields[0]: query.strip()} if len(fields) == 1 else {}
        return self._finalize(skill_name, fallback)

    @staticmethod
    def _context_text(chat_history: str, user_profile: str) -> str:
        return "\n".join(part for part in (chat_history, user_profile) if part)

    def _is_structured_skill_payload(self, parsed: Dict[str, Any], query: str) -> bool:
        if not isinstance(parsed, dict) or not parsed:
            return False
        if set(parsed.keys()) != {"query"}:
            return True
        return str(parsed.get("query", "")).strip() != query.strip()

    def _finalize(self, skill_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        definition = registry.get_skill_definition(skill_name)
        payload = definition.input_model.model_validate(params or {})
        return payload.model_dump(exclude_none=True)

    def _extract_location(self, query: str) -> Optional[str]:
        coord_match = re.search(
            r"(?<!\d)(-?\d{1,2}(?:\.\d+)?)\s*[,，]\s*(-?\d{2,3}(?:\.\d+)?)(?!\d)",
            query,
        )
        if coord_match:
            return f"{coord_match.group(1)},{coord_match.group(2)}"
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
            "银河系",
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
        for token in (
            "今晚",
            "明晚",
            "今天",
            "明天",
            "本周末",
            "这个周末",
            "这周末",
            "周末",
            "下周一",
        ):
            if token in query:
                if token in {"这个周末", "这周末", "周末"}:
                    return "本周末"
                return token
        return None

    def _extract_event_range(self, query: str) -> tuple[Optional[str], Optional[str]]:
        month_match = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", query)
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
                end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(
                    days=1
                )
            else:
                end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        if "今年" in query:
            today = datetime.now()
            return today.strftime("%Y-%m-%d"), datetime(today.year, 12, 31).strftime(
                "%Y-%m-%d"
            )

        if any(token in query for token in ("适合普通人", "带朋友看天象", "月内天象")):
            today = datetime.now()
            start = today.replace(day=1)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(
                    days=1
                )
            else:
                end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        if any(
            token in query
            for token in ("未来一周", "未来7天", "本周天象", "这周天象", "一周天象")
        ):
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

    def _extract_iso_date(self, query: str) -> Optional[str]:
        match = re.search(r"\d{4}-\d{2}-\d{2}", query)
        if match:
            return match.group(0)
        if any(token in query for token in ("今天", "今日", "今晚", "每日天文图")):
            return datetime.now().strftime("%Y-%m-%d")
        return None

    def _extract_equipment(self, query: str) -> Optional[str]:
        for equipment in (
            "双筒望远镜",
            "双筒",
            "小折射镜",
            "8寸望远镜",
            "赤道仪",
            "固定三脚架",
            "三脚架",
        ):
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
        radec_terms = ("赤经", "赤纬", "RA", "Dec", "radec")
        if any(term in query for term in radec_terms):
            return "radec"

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
            "能看",
            "能看到",
            "能看见",
            "在哪",
            "哪里",
            "更好",
            "前半夜",
            "后半夜",
            "altaz",
        )
        if any(term in query for term in altaz_terms):
            return "altaz"

        if "日落" in query or "天黑" in query:
            return "rise_set"

        if "坐标" in query:
            return "radec"
        return None

    def _extract_event_type(self, query: str) -> Optional[str]:
        for event_type in ("流星雨", "月食", "日食", "行星合月", "冲日", "掩星"):
            if event_type in query:
                return event_type
        return None

    def _extract_event_operation(
        self,
        query: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Optional[str]:
        if end_date:
            return "monthly"
        if any(token in query for token in ("本月", "这个月", "适合普通人", "带朋友")):
            return "monthly"
        if any(
            token in query for token in ("未来一周", "未来7天", "本周", "这周", "一周")
        ):
            return "weekly"
        return None

    def _extract_position_operation(self, query: str) -> Optional[str]:
        ra, dec = self._extract_radec(query)
        if self._is_current_sky_query(query):
            return "current_sky"
        if ra is not None and dec is not None:
            return "coordinate_transformation"
        output_format = self._extract_output_format(query)
        if output_format == "altaz":
            return "altaz"
        if output_format == "rise_set":
            return "rise_set"
        if output_format == "radec":
            return "planet_position"
        return None

    @staticmethod
    def _is_current_sky_query(query: str) -> bool:
        return any(
            token in query
            for token in (
                "天上有什么",
                "当前天空",
                "现在天空",
                "今晚天空",
                "能看到哪些",
                "能看哪些",
                "哪些亮星",
                "亮星或行星",
            )
        )

    def _extract_radec(self, query: str) -> tuple[Optional[float], Optional[float]]:
        return self._extract_ra_hours(query), self._extract_dec_degrees(query)

    @staticmethod
    def _extract_ra_hours(query: str) -> Optional[float]:
        match = re.search(
            r"(?:赤经|RA)\s*([0-9]{1,2}(?:\.\d+)?)\s*h(?:\s*([0-9]{1,2}(?:\.\d+)?)\s*m)?(?:\s*([0-9]{1,2}(?:\.\d+)?)\s*s)?",
            query,
            re.IGNORECASE,
        )
        if not match:
            return None
        hours = float(match.group(1))
        minutes = float(match.group(2) or 0)
        seconds = float(match.group(3) or 0)
        return hours + minutes / 60.0 + seconds / 3600.0

    @staticmethod
    def _extract_dec_degrees(query: str) -> Optional[float]:
        match = re.search(
            r"(?:赤纬|Dec)\s*([+-]?\d{1,2}(?:\.\d+)?)\s*[°d](?:\s*(\d{1,2}(?:\.\d+)?)\s*[′'m])?(?:\s*(\d{1,2}(?:\.\d+)?)\s*[″\"s])?",
            query,
            re.IGNORECASE,
        )
        if not match:
            return None
        degrees = float(match.group(1))
        sign = -1.0 if degrees < 0 else 1.0
        minutes = float(match.group(2) or 0)
        seconds = float(match.group(3) or 0)
        return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)

    @staticmethod
    def _extract_web_search_query(query: str) -> str:
        cleaned = query.strip()
        cleaned = re.sub(r"^帮我查一下", "", cleaned)
        return cleaned.strip(" ？?。") or query.strip()

    @staticmethod
    def _extract_neo_time_range(query: str) -> Optional[str]:
        if any(token in query for token in ("未来一周", "最近", "靠近", "飞掠")):
            return "未来7天"
        if "本月" in query:
            return "本月"
        return None

    @staticmethod
    def _extract_neo_filters(query: str) -> tuple[Optional[float], Optional[float]]:
        min_size = None
        max_distance = None
        size_match = re.search(r"(?:超过|大于|直径超过)\s*(\d+(?:\.\d+)?)\s*米", query)
        if size_match:
            min_size = float(size_match.group(1))
        elif any(token in query for token in ("个头也不算小", "比较大", "值得关注")):
            min_size = 50.0

        distance_match = re.search(r"(\d+(?:\.\d+)?)\s*个?地月距离(?:以内|内)?", query)
        if distance_match:
            max_distance = float(distance_match.group(1))
        elif any(token in query for token in ("比较近", "靠近")):
            max_distance = 20.0
        return min_size, max_distance
