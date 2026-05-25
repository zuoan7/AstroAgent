"""Celestial position calculator skill handler."""

from __future__ import annotations

import datetime as dt_mod
import time

from src.skills.result import SkillResult
from src.utils.param_parser import ParamParser
from src.skills import registry
from src.skills.context import SkillContext
from src.skills.inputs import CelestialPositionCalculatorInput
from src.skills.services.lookup import PLANET_NAME_ALIASES, parse_location
from src.skills.services.tool_results import tool_payload_and_text, tool_source_entry

_POSITION_OPERATIONS = {
    "altaz",
    "rise_set",
    "planet_position",
    "current_sky",
    "coordinate_transformation",
}


def celestial_position_calculator_handler(
    ctx: SkillContext,
    payload: CelestialPositionCalculatorInput,
) -> SkillResult:
    """Calculate sky position, rise/set time, current sky or coordinate transform."""
    started = time.perf_counter()
    operation_name = _resolve_operation(payload)
    if not payload.target and operation_name not in {
        "current_sky",
        "coordinate_transformation",
    }:
        return SkillResult.from_error(
            skill_name="celestial-position-calculator",
            error_code="VALIDATION_ERROR",
            error_message='天体位置计算技能需要提供目标名称（target），例如"mars""jupiter"等',
        )
    if operation_name == "coordinate_transformation" and (
        payload.ra is None or payload.dec is None
    ):
        return SkillResult.from_error(
            skill_name="celestial-position-calculator",
            error_code="VALIDATION_ERROR",
            error_message="坐标转换需要提供赤经 ra 和赤纬 dec。",
        )

    operation_spec = registry.get_operation_spec(
        "celestial-position-calculator",
        operation_name,
    )
    child_ctx = ctx.with_tool_policy(
        operation=operation_name,
        allowed_tools=list(operation_spec.allowed_child_tools),
        forbidden_tools=list(operation_spec.forbidden_child_tools),
        required_params=[],
    )

    obs_time = (
        ParamParser.parse_date(payload.datetime)
        if payload.datetime
        else dt_mod.datetime.now()
    )
    lat, lon = parse_location(payload.location or "")
    if lat is None or lon is None:
        lat, lon = 39.9, 116.4

    tool_name = operation_spec.atomic_tool_name
    result = child_ctx.tool_kit.invoke(
        tool_name,
        **_tool_kwargs(payload, operation_name, obs_time, lat, lon),
    )
    position_data, body = tool_payload_and_text(result)
    sources = [tool_source_entry(result, snippet_text=body)]
    body = ParamParser.shorten_text(body, 600)
    if not isinstance(position_data, dict):
        position_data = {"raw": position_data if position_data is not None else body}

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SkillResult(
        skill_name="celestial-position-calculator",
        success=result.ok,
        data={
            "target": payload.target,
            "observation_time": obs_time.isoformat(),
            "latitude": lat,
            "longitude": lon,
            "output_format": (payload.output_format or "radec").lower(),
            "operation": operation_name,
            "position": position_data,
        },
        summary=_summary(payload.target, obs_time, lat, lon, operation_name, body),
        sources=sources,
        error_code=result.error.code if result.error else None,
        error_message=result.error.message if result.error else None,
        latency_ms=round(elapsed_ms, 2),
        logical_skill="celestial-position-calculator",
        operation=operation_name,
        expected_mcp_tools=[tool_name],
        allowed_child_tools=list(operation_spec.allowed_child_tools),
        forbidden_child_tools=list(operation_spec.forbidden_child_tools),
    )


def _resolve_operation(payload: CelestialPositionCalculatorInput) -> str:
    requested = (payload.operation or "").strip().lower()
    if requested in _POSITION_OPERATIONS:
        return requested
    output_format = (payload.output_format or "radec").strip().lower()
    if output_format in {"rise_set", "rise-set", "riseset"}:
        return "rise_set"
    if output_format == "altaz":
        return "altaz"
    return "planet_position"


def _tool_kwargs(
    payload: CelestialPositionCalculatorInput,
    operation_name: str,
    obs_time: dt_mod.datetime,
    lat: float,
    lon: float,
) -> dict:
    mcp_target = PLANET_NAME_ALIASES.get(payload.target, payload.target).lower()
    if operation_name == "current_sky":
        return {
            "latitude": lat,
            "longitude": lon,
            "date": obs_time.strftime("%Y-%m-%d"),
        }
    if operation_name == "coordinate_transformation":
        return {
            "ra": float(payload.ra or 0.0),
            "dec": float(payload.dec or 0.0),
            "epoch": payload.epoch or "J2000",
            "target_system": payload.target_system or "fk5",
        }
    if operation_name == "rise_set":
        return {
            "body_name": mcp_target,
            "date": obs_time.strftime("%Y-%m-%d"),
            "latitude": lat,
            "longitude": lon,
        }
    return {
        "planet_name": mcp_target,
        "observation_time": obs_time.isoformat(),
        "latitude": lat,
        "longitude": lon,
    }


def _summary(
    target: str,
    obs_time: dt_mod.datetime,
    lat: float,
    lon: float,
    operation_name: str,
    body: str,
) -> str:
    coordinate_label = {
        "rise_set": "升起/落下时间",
        "altaz": "地平坐标（高度角/方位角）",
        "current_sky": "当前天空目标",
        "coordinate_transformation": "坐标转换",
    }.get(operation_name, "赤道坐标")
    header = (
        f"🪐 天体位置计算\n"
        f"- 目标：{target}\n"
        f"- 时间：{obs_time.isoformat()}\n"
        f"- 观测点：纬度 {lat}，经度 {lon}\n"
        f"- 输出坐标系：{coordinate_label}\n"
    )
    return header + "\n原始计算结果（来自底层工具）：\n" + body
