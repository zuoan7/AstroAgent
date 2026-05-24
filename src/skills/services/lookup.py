"""Lookup tables and parsing helpers shared by astronomy skills."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

CITY_COORDS: Dict[str, tuple[float, float]] = {
    "北京": (39.9, 116.4),
    "上海": (31.23, 121.47),
    "广州": (23.13, 113.26),
    "深圳": (22.54, 114.06),
    "苏州": (31.3, 120.58),
    "杭州": (30.27, 120.16),
    "成都": (30.57, 104.07),
    "南京": (32.06, 118.79),
    "武汉": (30.59, 114.3),
    "西安": (34.34, 108.94),
    "重庆": (29.56, 106.55),
    "天津": (39.12, 117.2),
    "青岛": (36.07, 120.38),
    "厦门": (24.48, 118.09),
}


PLANET_NAME_ALIASES: Dict[str, str] = {
    "水星": "mercury",
    "金星": "venus",
    "火星": "mars",
    "木星": "jupiter",
    "土星": "saturn",
    "天王星": "uranus",
    "海王星": "neptune",
}


def parse_location(location: str) -> tuple[Optional[float], Optional[float]]:
    """Parse a city name or 'latitude,longitude' text."""
    if not location:
        return None, None
    text = str(location).strip()
    if text in CITY_COORDS:
        return CITY_COORDS[text]
    text = text.replace("，", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        return None, None
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None


def parse_time_range(time_range: Optional[str]) -> tuple[str, str]:
    """Parse a natural language NEO time range into start/end dates."""
    today = datetime.now().date()
    if not time_range:
        return today.strftime("%Y-%m-%d"), (today + timedelta(days=7)).strftime(
            "%Y-%m-%d"
        )

    text = str(time_range)
    if "30" in text and "天" in text:
        start = today
        end = today + timedelta(days=30)
    elif "本月" in text:
        start = today.replace(day=1)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
    else:
        start = today
        end = today + timedelta(days=7)

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
