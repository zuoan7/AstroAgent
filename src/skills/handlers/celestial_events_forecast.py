"""Celestial events forecast skill handler."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from src.agent.models.skill_result import SkillResult
from src.agent.param_parser import ParamParser
from src.core.config import settings
from src.skills import registry
from src.skills.context import SkillContext
from src.skills.inputs import CelestialEventsForecastInput
from src.skills.services.tool_results import tool_payload_and_text, tool_source_entry


def celestial_events_forecast_handler(
    ctx: SkillContext,
    payload: CelestialEventsForecastInput,
) -> SkillResult:
    """Query weekly or monthly celestial events and summarize them."""
    started = time.perf_counter()
    start_dt = _supported_date(ParamParser.parse_date(payload.start_date))
    end_dt = _parse_end_date(payload.end_date)
    operation_name = _resolve_operation(payload.operation, start_dt, end_dt)
    operation_spec = registry.get_operation_spec(
        "celestial-events-forecast",
        operation_name,
    )
    child_ctx = ctx.with_tool_policy(
        operation=operation_name,
        allowed_tools=list(operation_spec.allowed_child_tools),
        forbidden_tools=list(operation_spec.forbidden_child_tools),
        required_params=[],
    )

    results = _invoke_events(
        child_ctx, operation_spec.atomic_tool_name, start_dt, end_dt
    )
    payloads = []
    texts = []
    sources = []
    for result in results:
        body_data, body = tool_payload_and_text(result)
        payloads.append(body_data)
        texts.append(body)
        sources.append(tool_source_entry(result, snippet_text=body))

    body = "\n".join(text for text in texts if text)
    body_data = payloads[0] if len(payloads) == 1 else payloads
    summary = "\n".join(
        [
            *_description_prefix(
                start_dt, end_dt, payload.end_date, payload.event_type
            ),
            "\n下面是为你整理的天象预报：\n",
            ParamParser.shorten_text(body, 1200),
        ]
    )
    success = all(result.ok for result in results)
    first_error = next((result.error for result in results if result.error), None)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SkillResult(
        skill_name="celestial-events-forecast",
        success=success,
        data={
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d") if end_dt else None,
            "event_type": payload.event_type,
            "operation": operation_name,
            "events_body": body_data,
        },
        summary=summary,
        sources=sources,
        error_code=first_error.code if first_error else None,
        error_message=first_error.message if first_error else None,
        latency_ms=round(elapsed_ms, 2),
        logical_skill="celestial-events-forecast",
        operation=operation_name,
        expected_mcp_tools=[operation_spec.atomic_tool_name],
        allowed_child_tools=list(operation_spec.allowed_child_tools),
        forbidden_child_tools=list(operation_spec.forbidden_child_tools),
    )


def _supported_date(value: datetime) -> datetime:
    supported_min, supported_max = settings.SUPPORTED_YEAR_RANGE
    if not (supported_min <= value.year <= supported_max):
        return value.replace(year=supported_min)
    return value


def _parse_end_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return _supported_date(ParamParser.parse_date(value))
    except Exception:
        return None


def _resolve_operation(
    requested: Optional[str],
    start_dt: datetime,
    end_dt: Optional[datetime],
) -> str:
    operation_name = (requested or "").strip().lower()
    if operation_name in {"weekly", "monthly"}:
        return operation_name
    if end_dt:
        try:
            return "weekly" if (end_dt - start_dt).days <= 7 else "monthly"
        except Exception:
            return "monthly"
    return "weekly"


def _invoke_events(
    ctx: SkillContext,
    tool_name: str,
    start_dt: datetime,
    end_dt: Optional[datetime],
):
    if tool_name == "get_weekly_events":
        return [
            ctx.tool_kit.invoke(tool_name, start_date=start_dt.strftime("%Y-%m-%d"))
        ]
    if not end_dt:
        return [
            ctx.tool_kit.invoke(tool_name, year=start_dt.year, month=start_dt.month)
        ]

    calls = []
    current_dt = start_dt
    while current_dt <= end_dt:
        calls.append(
            {
                "tool_name": tool_name,
                "kwargs": {"year": current_dt.year, "month": current_dt.month},
            }
        )
        if current_dt.month == 12:
            current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
        else:
            current_dt = current_dt.replace(month=current_dt.month + 1)
    return ctx.tool_kit.invoke_parallel(calls)


def _description_prefix(
    start_dt: datetime,
    end_dt: Optional[datetime],
    raw_end_date: Optional[str],
    event_type: Optional[str],
) -> list[str]:
    if end_dt:
        lines = [
            f"天象预报范围：从 {start_dt.strftime('%Y-%m-%d')} 开始，直到 {end_dt.strftime('%Y-%m-%d')}"
        ]
    else:
        suffix = f"，直到 {raw_end_date}" if raw_end_date else "，未来一段时间内"
        lines = [f"天象预报范围：从 {start_dt.strftime('%Y-%m-%d')} 开始{suffix}"]
    if event_type:
        lines.append(
            f"用户关心的事件类型：{event_type}（当前版本为软筛选，仅供解释用）"
        )
    return lines
