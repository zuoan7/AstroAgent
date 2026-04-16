from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set


RouteType = Literal["simple", "handler"]


@dataclass(frozen=True)
class SkillSpec:
    skill_name: str
    langchain_tool_name: str
    summary: str
    description: str
    route_type: RouteType
    mcp_tool_name: Optional[str] = None
    param_names: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    type_conversions: Dict[str, type] = field(default_factory=dict)
    param_mapping: Dict[str, str] = field(default_factory=dict)
    special_handling: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


def normalize_weather_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """天气查询的特殊处理：合并 city 和 location 参数"""
    normalized = dict(kwargs)
    target = normalized.get("city") or normalized.get("location")
    if target:
        normalized["city"] = target
        normalized.pop("location", None)
    return normalized


_ASTRONOMY_SKILL_SPECS: tuple[SkillSpec, ...] = (
    SkillSpec(
        skill_name="weather-lookup",
        langchain_tool_name="WeatherLookup",
        summary="查询指定城市的观测相关天气信息",
        description=(
            "查询指定城市的观测相关天气信息（skill: weather-lookup，对应 MCP 工具 get_weather）。\n"
            "参数：city（城市名称或adcode，可选），"
            "location（城市名称，和 city 等价，可选），"
            "extensions（\"base\" 实时 或 \"all\" 预报，默认 all）。"
        ),
        route_type="simple",
        mcp_tool_name="get_weather",
        param_names=["city", "location", "extensions"],
        defaults={"extensions": "all"},
        param_mapping={"city": "city", "location": "city", "extensions": "extensions"},
        special_handling=normalize_weather_params,
    ),
    SkillSpec(
        skill_name="observation-planner",
        langchain_tool_name="ObservationPlanner",
        summary="生成指定日期和地点的天文观测计划",
        description=(
            "生成指定日期和地点的天文观测计划（skill: observation-planner）。\n"
            '参数：date（观测日期，可为"今天""明天"或YYYY-MM-DD，可选），'
            'location（观测地点，城市名或"纬度,经度"，必填），'
            'duration（观测时段，如"整夜""前半夜""后半夜"，可选）。'
        ),
        route_type="handler",
        param_names=["date", "location", "duration"],
    ),
    SkillSpec(
        skill_name="celestial-events-forecast",
        langchain_tool_name="CelestialEventsForecast",
        summary="查询指定时间段的天象事件",
        description=(
            "查询指定时间段的天象事件（skill: celestial-events-forecast）。\n"
            "参数：start_date（开始日期YYYY-MM-DD，可选），"
            "end_date（结束日期YYYY-MM-DD，可选），"
            "event_type（事件类型，如'流星雨''行星合月''月食'，可选，用于意图说明）。"
        ),
        route_type="handler",
        param_names=["start_date", "end_date", "event_type"],
    ),
    SkillSpec(
        skill_name="deep-sky-observing-guide",
        langchain_tool_name="DeepSkyObservingGuide",
        summary="为指定深空天体提供观测指导",
        description=(
            "为指定深空天体提供观测指导（skill: deep-sky-observing-guide）。\n"
            "参数：target（天体名称，如'M31''猎户座大星云'，必填），"
            "observer_location（观测者位置，可选），"
            "date（观测日期，可选），"
            "equipment（设备描述，如'裸眼''双筒''8寸望远镜'，可选）。"
        ),
        route_type="handler",
        param_names=["target", "observer_location", "date", "equipment"],
    ),
    SkillSpec(
        skill_name="neo-tracker",
        langchain_tool_name="NEOTracker",
        summary="追踪近地天体飞掠事件",
        description=(
            "追踪近地天体飞掠事件（skill: neo-tracker）。\n"
            "参数：time_range（时间范围，如'未来30天''本月'，可选），"
            "min_size（最小直径，单位米，可选），"
            "max_distance（最大距离，单位地月距离倍数，可选），"
            "observable_only（是否只返回具有观测价值的目标，布尔值，可选）。"
        ),
        route_type="handler",
        param_names=["time_range", "min_size", "max_distance", "observable_only"],
        type_conversions={"min_size": float, "max_distance": float, "observable_only": bool},
    ),
    SkillSpec(
        skill_name="astrophotography-calculator",
        langchain_tool_name="AstrophotographyCalculator",
        summary="计算天文摄影参数与建议",
        description=(
            "计算天文摄影参数与拍摄建议（skill: astrophotography-calculator）。\n"
            "参数：target（拍摄目标，必填），"
            "camera（相机型号，必填），"
            "telescope（望远镜型号或焦距，可选），"
            "mount（赤道仪型号，可选），"
            "location（拍摄地点，可选），"
            "date（拍摄日期，可选）。"
        ),
        route_type="handler",
        param_names=["target", "camera", "telescope", "mount", "location", "date"],
    ),
    SkillSpec(
        skill_name="celestial-position-calculator",
        langchain_tool_name="CelestialPositionCalculator",
        summary="计算天体在指定时间的位置",
        description=(
            "计算天体在指定时间的位置（skill: celestial-position-calculator）。\n"
            "参数：target（目标名称，如'mars''jupiter'等，必填），"
            "datetime（观测时间，建议YYYY-MM-DD HH:MM 格式，可选，默认当前时间），"
            "location（观测地点，经纬度'纬度,经度'形式，可选），"
            "output_format（输出坐标格式，如'altaz''radec'，可选）。"
        ),
        route_type="handler",
        param_names=["target", "datetime", "location", "output_format"],
    ),
)


def get_skill_specs() -> List[SkillSpec]:
    return list(_ASTRONOMY_SKILL_SPECS)


def get_skill_spec(skill_name: str) -> SkillSpec:
    for spec in get_skill_specs():
        if spec.skill_name == skill_name:
            return spec
    raise KeyError(f"未知技能：{skill_name}")


def list_skill_descriptions() -> Dict[str, str]:
    return {spec.skill_name: spec.summary for spec in get_skill_specs()}


def list_langchain_tool_names() -> List[str]:
    return [spec.langchain_tool_name for spec in get_skill_specs()]


def validate_skill_registry(
    specs: Optional[Sequence[SkillSpec]] = None,
    handler_names: Optional[Set[str]] = None,
) -> None:
    specs = list(specs or get_skill_specs())

    seen_skill_names: Set[str] = set()
    seen_tool_names: Set[str] = set()
    handler_skill_names: Set[str] = set()

    for spec in specs:
        if not spec.skill_name:
            raise ValueError("Skill registry 校验失败：存在空 skill_name")
        if not spec.langchain_tool_name:
            raise ValueError(f"Skill registry 校验失败：{spec.skill_name} 缺少 langchain_tool_name")
        if not spec.summary or not spec.description:
            raise ValueError(f"Skill registry 校验失败：{spec.skill_name} 缺少说明文案")
        if spec.skill_name in seen_skill_names:
            raise ValueError(f"Skill registry 校验失败：重复的 skill_name: {spec.skill_name}")
        if spec.langchain_tool_name in seen_tool_names:
            raise ValueError(f"Skill registry 校验失败：重复的 LangChain Tool 名称: {spec.langchain_tool_name}")

        seen_skill_names.add(spec.skill_name)
        seen_tool_names.add(spec.langchain_tool_name)

        if spec.route_type == "simple":
            if not spec.mcp_tool_name:
                raise ValueError(f"Skill registry 校验失败：simple skill {spec.skill_name} 缺少 mcp_tool_name")
        elif spec.route_type == "handler":
            handler_skill_names.add(spec.skill_name)
        else:
            raise ValueError(f"Skill registry 校验失败：{spec.skill_name} 的 route_type 非法: {spec.route_type}")

    if handler_names is not None:
        missing_handlers = handler_skill_names - handler_names
        extra_handlers = handler_names - handler_skill_names
        if missing_handlers:
            raise ValueError(
                "Skill registry 校验失败：以下 handler skill 未在 SKILL_HANDLERS 中注册："
                + ", ".join(sorted(missing_handlers))
            )
        if extra_handlers:
            raise ValueError(
                "Skill registry 校验失败：以下 SKILL_HANDLERS 未在 registry 中声明："
                + ", ".join(sorted(extra_handlers))
            )
