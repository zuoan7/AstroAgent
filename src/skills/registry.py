"""技能注册表，声明高层技能、LangChain 工具名、参数和 operation 子工具策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set


RouteType = Literal["simple", "handler"]


@dataclass(frozen=True)
class SkillSpec:
    """高层技能的静态定义。"""

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


@dataclass(frozen=True)
class OperationSpec:
    """高层技能 operation 与底层原子工具之间的策略定义。"""

    logical_skill: str
    operation: str
    atomic_tool_name: str
    trigger_summary: str
    required_params: List[str] = field(default_factory=list)
    allowed_child_tools: List[str] = field(default_factory=list)
    forbidden_child_tools: List[str] = field(default_factory=list)


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
            "event_type（事件类型，如'流星雨''行星合月''月食'，可选，用于意图说明），"
            "operation（weekly 或 monthly，可选，用于约束底层 MCP 工具）。"
        ),
        route_type="handler",
        param_names=["start_date", "end_date", "event_type", "operation"],
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
            "date（拍摄日期，可选），"
            "iso（感光度，可选），"
            "aperture（光圈，可选）。"
        ),
        route_type="handler",
        param_names=[
            "target",
            "camera",
            "telescope",
            "mount",
            "location",
            "date",
            "iso",
            "aperture",
        ],
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
            "output_format（输出格式，如'altaz''radec''rise_set'，可选），"
            "operation（altaz、rise_set、planet_position、coordinate_transformation 或 current_sky，可选，用于约束底层 MCP 工具）。"
        ),
        route_type="handler",
        param_names=[
            "target",
            "datetime",
            "location",
            "output_format",
            "operation",
            "ra",
            "dec",
            "epoch",
            "target_system",
        ],
    ),
)


_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        logical_skill="celestial-position-calculator",
        operation="altaz",
        atomic_tool_name="get_altaz",
        trigger_summary="目标在指定时间地点的高度角、方位角或可见性判断",
        required_params=["target", "datetime", "location"],
        allowed_child_tools=["get_altaz"],
        forbidden_child_tools=[
            "get_planet_position",
            "get_rise_set_times",
            "get_current_sky_objects",
        ],
    ),
    OperationSpec(
        logical_skill="celestial-position-calculator",
        operation="rise_set",
        atomic_tool_name="get_rise_set_times",
        trigger_summary="目标升起、落下、日落或与地平相关的时间判断",
        required_params=["target", "datetime", "location"],
        allowed_child_tools=["get_rise_set_times"],
        forbidden_child_tools=[
            "get_altaz",
            "get_planet_position",
            "get_current_sky_objects",
        ],
    ),
    OperationSpec(
        logical_skill="celestial-position-calculator",
        operation="planet_position",
        atomic_tool_name="get_planet_position",
        trigger_summary="行星赤道坐标或通用位置查询",
        required_params=["target", "datetime", "location"],
        allowed_child_tools=["get_planet_position"],
        forbidden_child_tools=["get_altaz", "get_rise_set_times"],
    ),
    OperationSpec(
        logical_skill="celestial-position-calculator",
        operation="current_sky",
        atomic_tool_name="get_current_sky_objects",
        trigger_summary="指定时间地点当前可见天空目标列表",
        required_params=["datetime", "location"],
        allowed_child_tools=["get_current_sky_objects"],
        forbidden_child_tools=["get_altaz", "get_planet_position", "get_rise_set_times"],
    ),
    OperationSpec(
        logical_skill="celestial-position-calculator",
        operation="coordinate_transformation",
        atomic_tool_name="coordinate_transformation",
        trigger_summary="赤经赤纬坐标解析、坐标系统转换或位置格式解释",
        required_params=["ra", "dec"],
        allowed_child_tools=["coordinate_transformation"],
        forbidden_child_tools=[
            "get_altaz",
            "get_planet_position",
            "get_rise_set_times",
            "get_current_sky_objects",
        ],
    ),
    OperationSpec(
        logical_skill="celestial-events-forecast",
        operation="weekly",
        atomic_tool_name="get_weekly_events",
        trigger_summary="未来一周或不超过 7 天的天象事件",
        required_params=["start_date"],
        allowed_child_tools=["get_weekly_events"],
        forbidden_child_tools=["get_monthly_events"],
    ),
    OperationSpec(
        logical_skill="celestial-events-forecast",
        operation="monthly",
        atomic_tool_name="get_monthly_events",
        trigger_summary="本月、这个月或面向普通人的月度天象筛选",
        required_params=["start_date", "end_date"],
        allowed_child_tools=["get_monthly_events"],
        forbidden_child_tools=["get_weekly_events"],
    ),
)


def get_skill_specs() -> List[SkillSpec]:
    """返回所有高层技能定义。"""
    return list(_ASTRONOMY_SKILL_SPECS)


def get_skill_spec(skill_name: str) -> SkillSpec:
    """按技能名读取单个技能定义。"""
    for spec in get_skill_specs():
        if spec.skill_name == skill_name:
            return spec
    raise KeyError(f"未知技能：{skill_name}")


def get_operation_specs() -> List[OperationSpec]:
    """返回所有 operation 子工具策略定义。"""
    return list(_OPERATION_SPECS)


def get_operation_spec(logical_skill: str, operation: str) -> OperationSpec:
    """按高层技能和 operation 读取子工具策略定义。"""
    for spec in get_operation_specs():
        if spec.logical_skill == logical_skill and spec.operation == operation:
            return spec
    raise KeyError(f"未知 operation：{logical_skill}.{operation}")


def list_operations_for_skill(logical_skill: str) -> List[OperationSpec]:
    """列出指定高层技能支持的所有 operation。"""
    return [spec for spec in get_operation_specs() if spec.logical_skill == logical_skill]


def list_skill_descriptions() -> Dict[str, str]:
    """返回技能名到摘要文案的映射。"""
    return {spec.skill_name: spec.summary for spec in get_skill_specs()}


def list_langchain_tool_names() -> List[str]:
    """返回注册给 LangChain ReAct Agent 的工具名列表。"""
    return [spec.langchain_tool_name for spec in get_skill_specs()]


def validate_skill_registry(
    specs: Optional[Sequence[SkillSpec]] = None,
    handler_names: Optional[Set[str]] = None,
) -> None:
    """校验技能注册表、handler 注册关系和工具名唯一性。"""
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
