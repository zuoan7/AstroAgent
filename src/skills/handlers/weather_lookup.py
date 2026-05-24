"""Weather lookup skill handler."""

from __future__ import annotations

from src.agent.models.skill_result import SkillResult
from src.skills.context import SkillContext
from src.skills.inputs import WeatherLookupInput
from src.skills.services.tool_results import (
    summarize_weather,
    tool_source_entry,
    weather_data,
)


def weather_lookup_handler(
    ctx: SkillContext,
    payload: WeatherLookupInput,
) -> SkillResult:
    """Invoke the weather atomic tool and wrap it as a SkillResult."""
    result = ctx.tool_kit.invoke(
        "get_weather",
        city=payload.city,
        extensions=payload.extensions or "all",
    )
    source = tool_source_entry(result, snippet_text=summarize_weather(result))
    if not result.ok:
        message = result.error.message if result.error else "天气工具调用失败"
        failed = SkillResult.from_error(
            skill_name="weather-lookup",
            error_code=result.error.code if result.error else "TOOL_CALL_FAILED",
            error_message=message,
            latency_ms=result.latency_ms,
        )
        failed.logical_skill = "weather-lookup"
        failed.expected_mcp_tools = ["get_weather"]
        failed.allowed_child_tools = ["get_weather"]
        failed.sources = [source]
        return failed

    summary = summarize_weather(result)
    if not summary:
        summary = str(result.data)
    return SkillResult(
        skill_name="weather-lookup",
        success=True,
        data=weather_data(result),
        summary=summary,
        sources=[source],
        latency_ms=result.latency_ms,
        logical_skill="weather-lookup",
        expected_mcp_tools=["get_weather"],
        allowed_child_tools=["get_weather"],
    )
