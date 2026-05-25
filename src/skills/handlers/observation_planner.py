"""Observation planner skill handler."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict

from src.skills.result import SkillResult
from src.utils.param_parser import ParamParser
from src.skills.context import SkillContext
from src.skills.inputs import ObservationPlannerInput
from src.skills.services.dates import is_date_like
from src.skills.services.tool_results import (
    summarize_weather,
    tool_payload_and_text,
    tool_source_entry,
    weather_data,
)


def observation_planner_handler(
    ctx: SkillContext,
    payload: ObservationPlannerInput,
) -> SkillResult:
    """Generate a night-sky observing plan for a date and location."""
    started = time.perf_counter()
    date, location = _normalize_date_location(payload.date, payload.location)
    obs_date = ParamParser.parse_date(date)
    display_location = ParamParser.normalize_location(location)
    query_city = _query_city(location, display_location)
    calls = _build_calls(query_city, obs_date)
    results = ctx.tool_kit.invoke_parallel(calls)

    sources = []
    weather_brief = ""
    weather_payload: Dict[str, Any] = {}
    weekly_events = ""
    tonight_best = ""
    for call, result in zip(calls, results):
        key = call["_key"]
        if key == "weather":
            weather_brief = summarize_weather(result)
            weather_payload = weather_data(result)
            sources.append(tool_source_entry(result, snippet_text=weather_brief))
        elif key == "weekly_events":
            _, weekly_events = tool_payload_and_text(result)
            sources.append(tool_source_entry(result, snippet_text=weekly_events))
        elif key == "tonight_best":
            _, tonight_best = tool_payload_and_text(result)
            sources.append(tool_source_entry(result, snippet_text=tonight_best))

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SkillResult(
        skill_name="observation-planner",
        success=all(result.ok for result in results),
        data={
            "obs_date": obs_date.strftime("%Y-%m-%d"),
            "location": display_location or "",
            "weather": weather_payload,
            "weekly_events": weekly_events,
            "tonight_best": tonight_best,
        },
        summary=_summary(
            obs_date,
            display_location,
            payload.duration,
            weather_brief,
            weekly_events,
            tonight_best,
        ),
        sources=sources,
        latency_ms=round(elapsed_ms, 2),
    )


def _normalize_date_location(date: Any, location: Any) -> tuple[Any, Any]:
    if not location and date:
        text = str(date).strip()
        if not is_date_like(text):
            return None, text
    return date, location or "北京"


def _query_city(location: Any, display_location: str) -> str:
    if isinstance(location, dict):
        return location.get("city") or display_location
    return display_location


def _build_calls(query_city: str, obs_date: datetime) -> list[dict]:
    calls = []
    if query_city:
        calls.append(
            {
                "tool_name": "get_weather",
                "kwargs": {"city": query_city, "extensions": "all"},
                "_key": "weather",
            }
        )
    calls.append(
        {
            "tool_name": "get_weekly_events",
            "kwargs": {"start_date": obs_date.strftime("%Y-%m-%d")},
            "_key": "weekly_events",
        }
    )
    if obs_date.strftime("%Y-%m-%d") == datetime.now().strftime("%Y-%m-%d"):
        calls.append(
            {"tool_name": "get_tonight_best", "kwargs": {}, "_key": "tonight_best"}
        )
    return calls


def _summary(
    obs_date: datetime,
    display_location: str,
    duration: str | None,
    weather_brief: str,
    weekly_events: str,
    tonight_best: str,
) -> str:
    lines = [f"📅 观测日期：{obs_date.strftime('%Y-%m-%d')}"]
    if display_location:
        lines.append(f"📍 观测地点：{display_location}")
    else:
        lines.append("📍 观测地点：未明确指定，本计划为一般性观测建议。")
    if duration:
        lines.append(f"⏱️ 观测时段：{duration}")
    lines.append("\n一、观测条件（天气概览）")
    lines.append(
        weather_brief
        or "暂时无法获取指定地点的天气信息，请根据当地实际情况或天气预报应用调整计划。"
    )
    lines.append("\n二、本周重要天象（摘要）")
    lines.append(ParamParser.shorten_text(weekly_events, 600))
    if tonight_best:
        lines.append('\n三、系统给出的"今晚最佳观测目标"参考')
        lines.append(ParamParser.shorten_text(tonight_best, 600))
    lines.append("\n四、实用建议")
    lines.append(
        "1. 根据云量与月相选择目标：若接近新月，可优先考虑深空天体；若接近满月，可多观测月面细节与行星。\n"
        "2. 提前抵达观测地，预留架台、极轴校准与试拍时间。\n"
        "3. 如湿度偏高或风力较大，请准备除露带、配重或防风措施。"
    )
    return "\n".join(lines)
