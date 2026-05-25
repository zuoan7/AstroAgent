"""Helpers for formatting structured ToolResult values inside skills."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.utils.param_parser import ParamParser
from src.tools.results import ToolResult


def format_tool_display_text(data: Any) -> str:
    """Normalize a tool payload into display text."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, (int, float, bool)):
        return str(data)
    if isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            return "\n".join(item for item in data if item)
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(data)
    if isinstance(data, dict):
        for key in (
            "summary",
            "description",
            "content",
            "text",
            "message",
            "answer",
            "result",
        ):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(data)
    return str(data)


def tool_payload_and_text(result: ToolResult) -> tuple[Any, str]:
    """Extract payload and display text from one ToolResult."""
    if not result.ok:
        message = result.error.message if result.error else ""
        return None, message
    return result.data, format_tool_display_text(result.data)


def tool_source_entry(
    result: ToolResult,
    *,
    snippet_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standard SkillResult source entry."""
    text = snippet_text
    if text is None:
        _, text = tool_payload_and_text(result)
    snippet = ParamParser.shorten_text(text or str(result.data), 240)
    return {"kind": "tool_output", "tool": result.tool_name, "snippet": snippet}


def summarize_weather(result: ToolResult) -> str:
    """Extract a user-facing weather summary from a weather ToolResult."""
    if not result.ok or not isinstance(result.data, dict):
        return ""

    data = result.data
    if data.get("error"):
        return ""

    parts = []
    live = data.get("live") or {}
    if live:
        city = live.get("city")
        weather = live.get("weather")
        temp = live.get("temperature")
        humidity = live.get("humidity")
        wind = live.get("windpower")
        if city:
            parts.append(
                f"{city} 当前天气：{weather}，气温约 {temp}°C，湿度 {humidity}%，风力 {wind} 级左右。"
            )
        else:
            parts.append(f"当前天气：{weather}，气温约 {temp}°C，湿度 {humidity}%。")

    forecast = data.get("forecast") or {}
    if forecast:
        casts = forecast.get("casts") or []
        if casts:
            first_day = casts[0]
            date = first_day.get("date", "")
            day_weather = first_day.get("dayweather", "")
            night_weather = first_day.get("nightweather", "")
            day_temp = first_day.get("daytemp", "")
            night_temp = first_day.get("nighttemp", "")
            city = forecast.get("city", "")
            if city:
                parts.append(
                    f"{city} ({date}) 白天：{day_weather}，{day_temp}°C；夜间：{night_weather}，{night_temp}°C。"
                )
            else:
                parts.append(
                    f"{date} 白天：{day_weather}，{day_temp}°C；夜间：{night_weather}，{night_temp}°C。"
                )

    for tip in data.get("observing_tips") or []:
        parts.append(f"- {tip}")
    return "\n".join(parts) if parts else ""


def weather_data(result: ToolResult) -> Dict[str, Any]:
    """Return structured weather data with legacy fallback shape."""
    if not result.ok:
        return {"error": True}
    return result.data if isinstance(result.data, dict) else {"raw": result.data}
