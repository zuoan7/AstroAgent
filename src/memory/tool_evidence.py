"""Shared helpers for classifying short-term tool evidence."""

from __future__ import annotations

import re


_TOOL_TYPE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "visibility",
        (
            "visibility",
            "seeing",
            "transparency",
            "透明度",
            "能见度",
            "视宁度",
        ),
    ),
    (
        "weather",
        (
            "weather",
            "weatherlookup",
            "weather-lookup",
            "get-weather",
            "getweather",
            "天气",
        ),
    ),
    (
        "position",
        (
            "celestial-position",
            "celestialposition",
            "position",
            "get-altaz",
            "getaltaz",
            "altaz",
            "get-rise-set-times",
            "getrisesettimes",
            "rise-set",
            "riseset",
            "get-current-sky-objects",
            "getcurrentskyobjects",
            "current-sky",
            "currentsky",
            "位置",
        ),
    ),
    (
        "ephemeris",
        (
            "ephemeris",
            "星历",
        ),
    ),
    (
        "catalog",
        (
            "catalog",
            "simbad",
            "messier",
            "ngc",
            "gaia",
            "get-astrophysical-object-info",
            "getastrophysicalobjectinfo",
            "astrophysical-object",
            "get-galaxy-data",
            "getgalaxydata",
            "galaxy-data",
        ),
    ),
    (
        "photo",
        (
            "astrophotography-calculator",
            "astrophotographycalculator",
            "astrophotography",
            "astrophoto",
            "photo",
            "exposure",
            "calculator",
            "摄影",
            "拍摄",
        ),
    ),
    (
        "neo",
        (
            "neo",
            "asteroid",
            "小行星",
            "近地",
        ),
    ),
    (
        "event",
        (
            "event",
            "forecast",
            "calendar",
            "meteor",
            "天象",
            "流星雨",
        ),
    ),
)


def infer_tool_evidence_type(tool_name: str) -> str:
    """Infer the configured evidence TTL bucket from a skill or MCP tool name."""

    raw = (tool_name or "").strip().lower()
    normalized = re.sub(r"[\s_]+", "-", raw)
    compact = re.sub(r"[\s_-]+", "", raw)
    for tool_type, aliases in _TOOL_TYPE_ALIASES:
        for alias in aliases:
            alias_lower = alias.lower()
            alias_normalized = re.sub(r"[\s_]+", "-", alias_lower)
            alias_compact = re.sub(r"[\s_-]+", "", alias_lower)
            if (
                alias_lower in raw
                or alias_normalized in normalized
                or alias_compact in compact
            ):
                return tool_type
    return "generic"


__all__ = ["infer_tool_evidence_type"]
