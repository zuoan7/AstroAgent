"""Astrophotography calculator skill handler."""

from __future__ import annotations

import time
from datetime import datetime

from src.skills.result import SkillResult
from src.utils.param_parser import ParamParser
from src.skills.context import SkillContext
from src.skills.inputs import AstrophotographyCalculatorInput


def astrophotography_calculator_handler(
    ctx: SkillContext,
    payload: AstrophotographyCalculatorInput,
) -> SkillResult:
    """Generate exposure, stacking and mount recommendations."""
    del ctx
    started = time.perf_counter()
    if not payload.target or not payload.camera:
        return SkillResult.from_error(
            skill_name="astrophotography-calculator",
            error_code="VALIDATION_ERROR",
            error_message="天文摄影参数技能需要提供拍摄目标（target）和相机（camera）。",
        )

    obs_date = ParamParser.parse_date(payload.date) if payload.date else datetime.now()
    lines = [
        "📷 天文摄影参数建议",
        f"🎯 拍摄目标：{payload.target}",
        f"📅 拍摄日期：{obs_date.strftime('%Y-%m-%d')}",
        f"📸 相机：{payload.camera}",
    ]
    if payload.telescope:
        lines.append(f"🔭 望远镜/镜头：{payload.telescope}")
    if payload.mount:
        lines.append(f"🗜 赤道仪/支架：{payload.mount}")
    if payload.location:
        lines.append(f"📍 拍摄地点：{payload.location}")
    if payload.aperture:
        lines.append(f"🔆 光圈：{payload.aperture}")
    if payload.iso:
        lines.append(f"🎚 ISO：{payload.iso}")

    lines.append("\n一、曝光时间估算（星点不拖尾的经验值）")
    lines.append(
        "若使用广角/标准镜头并在赤道仪跟踪下：\n"
        "- 星野/银河：单张 20-60 秒，ISO 1600-6400，光圈尽量开大。\n"
        "- 星云/星团：根据目标亮度，单张 120-300 秒，ISO 800-3200。\n"
        '若非跟踪（固定三脚架），可按"500 规则"粗略估计：曝光秒数 ≈ 500 / 焦距（全画幅等效）。'
    )
    lines.append("\n二、总曝光时间与叠加")
    lines.append(
        "为了获得较低噪点和更丰富细节，建议：\n"
        "- 星云/星系：累计曝光时间 1–3 小时以上（例如 120s × 30–90 张）。\n"
        "- 银河/星野：累计曝光 20 分钟以上即可有明显提升。\n"
        "请务必拍摄暗场/平场/偏置帧，以便后期校正。"
    )
    lines.append("\n三、赤道仪与极轴校准建议")
    lines.append(
        "若使用赤道仪：\n"
        "- 极轴误差越小，可用的单张曝光时间越长。\n"
        "- 建议使用极轴镜/电子极轴校准工具，将极轴误差控制在 1–2 角分以内。"
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SkillResult(
        skill_name="astrophotography-calculator",
        success=True,
        data={
            "target": payload.target,
            "camera": payload.camera,
            "telescope": payload.telescope,
            "mount": payload.mount,
            "location": payload.location,
            "obs_date": obs_date.strftime("%Y-%m-%d"),
            "iso": payload.iso,
            "aperture": payload.aperture,
        },
        summary="\n".join(lines),
        sources=[],
        latency_ms=round(elapsed_ms, 2),
    )
