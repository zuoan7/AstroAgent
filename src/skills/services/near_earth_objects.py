"""Near-earth object tracking service logic."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from src.skills.context import SkillContext
from src.skills.inputs import NeoTrackerInput
from src.skills.result import SkillResult
from src.skills.services.lookup import parse_time_range
from src.skills.services.tool_results import tool_payload_and_text, tool_source_entry


def track_near_earth_objects(
    ctx: SkillContext,
    payload: NeoTrackerInput,
) -> SkillResult:
    """Track near-earth object close approaches."""
    started = time.perf_counter()
    start_date, end_date = parse_time_range(payload.time_range)
    warnings = _range_warnings(start_date, end_date)
    result = ctx.tool_kit.invoke(
        "get_neo_data",
        start_date=start_date,
        end_date=end_date,
        limit=50,
    )
    neo_payload, neo_text = tool_payload_and_text(result)
    sources = [tool_source_entry(result, snippet_text=neo_text)]

    if not result.ok:
        return SkillResult.from_error(
            skill_name="neo-tracker",
            error_code=result.error.code if result.error else "TOOL_CALL_FAILED",
            error_message=result.error.message if result.error else "NEO 工具调用失败",
            latency_ms=_elapsed_ms(started),
        )
    if not isinstance(neo_payload, dict):
        return SkillResult(
            skill_name="neo-tracker",
            success=True,
            data={"raw": neo_payload},
            summary=neo_text or str(neo_payload),
            sources=sources,
            latency_ms=_elapsed_ms(started),
        )
    if neo_payload.get("error"):
        return SkillResult.from_error(
            skill_name="neo-tracker",
            error_code="TOOL_CALL_FAILED",
            error_message=str(neo_payload.get("error")),
            latency_ms=_elapsed_ms(started),
        )

    neos = _flatten_neos(neo_payload)
    filtered = _filter_neos(
        neos,
        min_size=payload.min_size,
        max_distance=payload.max_distance,
        observable_only=payload.observable_only,
    )
    return _result(filtered, len(neos), warnings, sources, _elapsed_ms(started))


def _range_warnings(start_date: str, end_date: str) -> list[str]:
    lines = []
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        delta_days = (end_dt - start_dt).days
        if delta_days > 7:
            lines.extend(
                [
                    "⚠️ 注意：NASA NEO API 最多只支持查询7天的数据。",
                    f"请求范围：{start_date} 至 {end_date}（{delta_days}天）",
                    f"将返回最近7天的数据：{start_date} 至 {(start_dt + timedelta(days=7)).strftime('%Y-%m-%d')}",
                    "",
                ]
            )
    except Exception:
        pass
    return lines


def _flatten_neos(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    neos = []
    for day, objs in (data.get("near_earth_objects") or {}).items():
        for obj in objs:
            neos.append((day, obj))
    return neos


def _filter_neos(
    neos: list[tuple[str, dict[str, Any]]],
    *,
    min_size: float | None,
    max_distance: float | None,
    observable_only: bool | None,
) -> list[dict[str, Any]]:
    filtered = []
    for day, obj in neos:
        size = (
            obj.get("estimated_diameter", {})
            .get("meters", {})
            .get("estimated_diameter_max")
        )
        ca_list = obj.get("close_approach_data") or []
        if not ca_list:
            continue
        miss_lunar = _float_or_none(ca_list[0].get("miss_distance", {}).get("lunar"))
        if min_size is not None and size is not None and size < min_size:
            continue
        if max_distance is not None and miss_lunar is not None:
            if miss_lunar > max_distance:
                continue
        abs_mag = _float_or_none(obj.get("absolute_magnitude_h"))
        if observable_only and abs_mag is not None and abs_mag > 25:
            continue
        filtered.append(
            {
                "date": day,
                "name": obj.get("name"),
                "size_m": size,
                "miss_distance_lunar": miss_lunar,
                "hazardous": obj.get("is_potentially_hazardous_asteroid"),
            }
        )
    return filtered


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


def _result(
    filtered: list[dict[str, Any]],
    total_raw: int,
    warnings: list[str],
    sources: list[dict[str, Any]],
    latency_ms: float,
) -> SkillResult:
    if not filtered:
        warning_text = "\n".join(warnings) if warnings else ""
        summary = (
            warning_text
            + "\n在给定的时间范围和筛选条件下，没有找到明显具有观测价值的近地天体飞掠事件。"
        )
        return SkillResult(
            skill_name="neo-tracker",
            success=True,
            data={"filtered_neos": [], "total_raw": total_raw},
            summary=summary.strip(),
            sources=sources,
            latency_ms=latency_ms,
        )

    lines = warnings.copy() if warnings else []
    if lines:
        lines.extend(["", "📡 近地天体飞掠列表（按时间排序）："])
    else:
        lines.append("📡 近地天体飞掠列表（按时间排序）：")
    filtered.sort(
        key=lambda item: (item["date"], item.get("miss_distance_lunar") or 1e9)
    )
    for item in filtered[:20]:
        hazard = "⚠️ 潜在威胁小行星" if item["hazardous"] else ""
        size = f"{item['size_m']:.0f} m" if item["size_m"] is not None else "未知"
        dist = (
            f"{item['miss_distance_lunar']:.2f} 个地月距离"
            if item["miss_distance_lunar"] is not None
            else "未知"
        )
        lines.append(
            f"- 日期 {item['date']}，目标 {item['name']}，估计直径约 {size}，最近距离约 {dist} {hazard}"
        )
    lines.append(
        "\n注：以上数据来自 NASA NEO 数据接口，是否实际可见还与亮度、天空背景和观测设备有关。"
    )
    return SkillResult(
        skill_name="neo-tracker",
        success=True,
        data={"filtered_neos": filtered[:20], "total_raw": total_raw},
        summary="\n".join(lines),
        sources=sources,
        latency_ms=latency_ms,
    )
