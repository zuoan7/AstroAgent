"""Deep-sky observing guide skill handler."""

from __future__ import annotations

import time
from datetime import datetime

from src.skills.result import SkillResult
from src.utils.param_parser import ParamParser
from src.skills.context import SkillContext
from src.skills.inputs import DeepSkyObservingGuideInput
from src.skills.services.tool_results import tool_payload_and_text, tool_source_entry

_GALAXY_TARGETS = {
    "m31",
    "m33",
    "m51",
    "m81",
    "m82",
    "m87",
    "m101",
    "m104",
    "ngc224",
    "ngc598",
}


def deep_sky_observing_guide_handler(
    ctx: SkillContext,
    payload: DeepSkyObservingGuideInput,
) -> SkillResult:
    """Generate observing advice for a deep-sky target."""
    started = time.perf_counter()
    if not payload.target:
        return SkillResult.from_error(
            skill_name="deep-sky-observing-guide",
            error_code="VALIDATION_ERROR",
            error_message='深空观测指导技能需要提供目标名称（target），例如"M31"或"猎户座大星云"',
        )

    obs_date = ParamParser.parse_date(payload.date) if payload.date else datetime.now()
    calls = [
        {
            "tool_name": "get_astrophysical_object_info",
            "kwargs": {"object_name": payload.target},
            "_key": "obj_info",
        }
    ]
    if _is_galaxy_target(payload.target):
        calls.append(
            {
                "tool_name": "get_galaxy_data",
                "kwargs": {"galaxy_name": payload.target},
                "_key": "galaxy_info",
            }
        )

    results = ctx.tool_kit.invoke_parallel(calls)
    obj_info_data = None
    obj_info_raw = ""
    galaxy_info_data = None
    galaxy_info_raw = None
    sources = []
    for call, result in zip(calls, results):
        data, text = tool_payload_and_text(result)
        sources.append(tool_source_entry(result, snippet_text=text))
        if call["_key"] == "obj_info":
            obj_info_data = data
            obj_info_raw = text
        elif call["_key"] == "galaxy_info":
            galaxy_info_data = data
            galaxy_info_raw = text

    lines = [
        f"🎯 深空目标：{payload.target}",
        f"📅 观测日期：{obs_date.strftime('%Y-%m-%d')}",
    ]
    if payload.observer_location:
        lines.append(f"📍 观测地点：{payload.observer_location}")
    if payload.equipment:
        lines.append(f"🔭 计划使用设备：{payload.equipment}")
    lines.append("\n一、目标基础信息（基于专业数据库）")
    lines.append(ParamParser.shorten_text(obj_info_raw, 600))
    if galaxy_info_raw:
        lines.append("\n补充：星系数据摘要")
        lines.append(ParamParser.shorten_text(galaxy_info_raw, 400))
    lines.append("\n三、观测建议")
    lines.append(
        "1. 建议选择无月光或月亮落下后 1-2 小时的时段进行深空观测。\n"
        '2. 若使用双筒或小口径望远镜，可优先寻找目标所在星座的亮星作"跳星"指引。\n'
        "3. 使用较低倍率（长焦距目镜）先锁定目标，再逐步提高放大倍率细看结构。"
    )

    success = all(result.ok for result in results)
    first_error = next((result.error for result in results if result.error), None)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SkillResult(
        skill_name="deep-sky-observing-guide",
        success=success,
        data={
            "target": payload.target,
            "obs_date": obs_date.strftime("%Y-%m-%d"),
            "observer_location": payload.observer_location,
            "equipment": payload.equipment,
            "obj_info": obj_info_data if obj_info_data is not None else obj_info_raw,
            "galaxy_info": (
                galaxy_info_data if galaxy_info_data is not None else galaxy_info_raw
            ),
        },
        summary="\n".join(lines),
        sources=sources,
        error_code=first_error.code if first_error else None,
        error_message=first_error.message if first_error else None,
        latency_ms=round(elapsed_ms, 2),
    )


def _is_galaxy_target(target: str) -> bool:
    normalized = (target or "").strip().lower().replace(" ", "")
    if "galaxy" in normalized or "星系" in normalized or "银河系" in normalized:
        return True
    return normalized in _GALAXY_TARGETS
