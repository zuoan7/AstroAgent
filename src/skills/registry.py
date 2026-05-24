"""Skill registry and legacy compatibility facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Set

from src.skills.definition import SkillDefinition
from src.skills.inputs import (
    AstrophotographyCalculatorInput,
    CelestialEventsForecastInput,
    CelestialPositionCalculatorInput,
    DeepSkyObservingGuideInput,
    NeoTrackerInput,
    ObservationPlannerInput,
    WeatherLookupInput,
)

RouteType = Literal["simple", "handler"]


@dataclass(frozen=True)
class SkillSpec:
    """Legacy static definition projected from SkillDefinition."""

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
    """High-level skill operation to atomic tool policy."""

    logical_skill: str
    operation: str
    atomic_tool_name: str
    trigger_summary: str
    required_params: List[str] = field(default_factory=list)
    allowed_child_tools: List[str] = field(default_factory=list)
    forbidden_child_tools: List[str] = field(default_factory=list)


def normalize_weather_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge city and location for legacy weather callers."""
    normalized = dict(kwargs)
    target = normalized.get("city") or normalized.get("location")
    if target:
        normalized["city"] = target
        normalized.pop("location", None)
    return normalized


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
        forbidden_child_tools=[
            "get_altaz",
            "get_planet_position",
            "get_rise_set_times",
        ],
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


_SKILL_METADATA: tuple[dict[str, Any], ...] = (
    {
        "name": "weather-lookup",
        "display_name": "WeatherLookup",
        "summary": "查询指定城市的观测相关天气信息",
        "description": (
            "查询指定城市的观测相关天气信息（skill: weather-lookup，对应 MCP 工具 get_weather）。\n"
            "参数：city（城市名称或adcode，可选），location（城市名称，和 city 等价，可选），"
            'extensions（"base" 实时 或 "all" 预报，默认 all）。'
        ),
        "input_model": WeatherLookupInput,
        "allowed_tools": ("get_weather",),
        "required_params": (),
        "defaults": {"extensions": "all"},
        "mcp_tool_name": "get_weather",
        "param_mapping": {
            "city": "city",
            "location": "city",
            "extensions": "extensions",
        },
        "special_handling": normalize_weather_params,
    },
    {
        "name": "observation-planner",
        "display_name": "ObservationPlanner",
        "summary": "生成指定日期和地点的天文观测计划",
        "description": (
            "生成指定日期和地点的天文观测计划（skill: observation-planner）。\n"
            '参数：date（观测日期，可为"今天""明天"或YYYY-MM-DD，可选），'
            'location（观测地点，城市名或"纬度,经度"，必填），'
            'duration（观测时段，如"整夜""前半夜""后半夜"，可选）。'
        ),
        "input_model": ObservationPlannerInput,
        "allowed_tools": ("get_weather", "get_weekly_events", "get_tonight_best"),
        "required_params": ("location",),
    },
    {
        "name": "celestial-events-forecast",
        "display_name": "CelestialEventsForecast",
        "summary": "查询指定时间段的天象事件",
        "description": (
            "查询指定时间段的天象事件（skill: celestial-events-forecast）。\n"
            "参数：start_date（开始日期YYYY-MM-DD，可选），end_date（结束日期YYYY-MM-DD，可选），"
            "event_type（事件类型，如'流星雨''行星合月''月食'，可选，用于意图说明），"
            "operation（weekly 或 monthly，可选，用于约束底层 MCP 工具）。"
        ),
        "input_model": CelestialEventsForecastInput,
        "allowed_tools": ("get_weekly_events", "get_monthly_events"),
        "required_params": (),
    },
    {
        "name": "deep-sky-observing-guide",
        "display_name": "DeepSkyObservingGuide",
        "summary": "为指定深空天体提供观测指导",
        "description": (
            "为指定深空天体提供观测指导（skill: deep-sky-observing-guide）。\n"
            "参数：target（天体名称，如'M31''猎户座大星云'，必填），"
            "observer_location（观测者位置，可选），date（观测日期，可选），"
            "equipment（设备描述，如'裸眼''双筒''8寸望远镜'，可选）。"
        ),
        "input_model": DeepSkyObservingGuideInput,
        "allowed_tools": ("get_astrophysical_object_info", "get_galaxy_data"),
        "required_params": ("target",),
    },
    {
        "name": "neo-tracker",
        "display_name": "NEOTracker",
        "summary": "追踪近地天体飞掠事件",
        "description": (
            "追踪近地天体飞掠事件（skill: neo-tracker）。\n"
            "参数：time_range（时间范围，如'未来30天''本月'，可选），"
            "min_size（最小直径，单位米，可选），max_distance（最大距离，单位地月距离倍数，可选），"
            "observable_only（是否只返回具有观测价值的目标，布尔值，可选）。"
        ),
        "input_model": NeoTrackerInput,
        "allowed_tools": ("get_neo_data",),
        "required_params": (),
        "type_conversions": {
            "min_size": float,
            "max_distance": float,
            "observable_only": bool,
        },
    },
    {
        "name": "astrophotography-calculator",
        "display_name": "AstrophotographyCalculator",
        "summary": "计算天文摄影参数与建议",
        "description": (
            "计算天文摄影参数与拍摄建议（skill: astrophotography-calculator）。\n"
            "参数：target（拍摄目标，必填），camera（相机型号，必填），"
            "telescope（望远镜型号或焦距，可选），mount（赤道仪型号，可选），"
            "location（拍摄地点，可选），date（拍摄日期，可选），iso（感光度，可选），"
            "aperture（光圈，可选）。"
        ),
        "input_model": AstrophotographyCalculatorInput,
        "allowed_tools": (),
        "required_params": ("target", "camera"),
    },
    {
        "name": "celestial-position-calculator",
        "display_name": "CelestialPositionCalculator",
        "summary": "计算天体在指定时间的位置",
        "description": (
            "计算天体在指定时间的位置（skill: celestial-position-calculator）。\n"
            "参数：target（目标名称，如'mars''jupiter'等，必填），"
            "datetime（观测时间，建议YYYY-MM-DD HH:MM 格式，可选，默认当前时间），"
            "location（观测地点，经纬度'纬度,经度'形式，可选），"
            "output_format（输出格式，如'altaz''radec''rise_set'，可选），"
            "operation（altaz、rise_set、planet_position、coordinate_transformation 或 current_sky，可选）。"
        ),
        "input_model": CelestialPositionCalculatorInput,
        "allowed_tools": (
            "get_altaz",
            "get_rise_set_times",
            "get_planet_position",
            "get_current_sky_objects",
            "coordinate_transformation",
        ),
        "required_params": ("target",),
    },
)


class SkillRegistry:
    """Read-only registry for high-level skill definitions."""

    def __init__(
        self,
        definitions: Optional[Iterable[SkillDefinition]] = None,
    ) -> None:
        self._definitions = {
            definition.name: definition
            for definition in (definitions or _build_skill_definitions())
        }

    def register(self, definition: SkillDefinition) -> None:
        """Register or replace one skill definition."""
        self._definitions[definition.name] = definition

    def list(self) -> list[SkillDefinition]:
        """Return all registered skill definitions."""
        return list(self._definitions.values())

    def list_names(self) -> list[str]:
        """Return all registered skill names."""
        return list(self._definitions.keys())

    def has_skill(self, name: str) -> bool:
        """Return whether a skill name is registered."""
        return name in self._definitions

    def get(self, name: str) -> SkillDefinition:
        """Return one skill definition or raise KeyError."""
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"未知技能：{name}") from exc


def _build_skill_definitions() -> list[SkillDefinition]:
    from src.skills.handlers.astrophotography_calculator import (
        astrophotography_calculator_handler,
    )
    from src.skills.handlers.celestial_events_forecast import (
        celestial_events_forecast_handler,
    )
    from src.skills.handlers.celestial_position_calculator import (
        celestial_position_calculator_handler,
    )
    from src.skills.handlers.deep_sky_observing_guide import (
        deep_sky_observing_guide_handler,
    )
    from src.skills.handlers.neo_tracker import neo_tracker_handler
    from src.skills.handlers.observation_planner import observation_planner_handler
    from src.skills.handlers.weather_lookup import weather_lookup_handler

    handlers = {
        "weather-lookup": weather_lookup_handler,
        "observation-planner": observation_planner_handler,
        "celestial-events-forecast": celestial_events_forecast_handler,
        "deep-sky-observing-guide": deep_sky_observing_guide_handler,
        "neo-tracker": neo_tracker_handler,
        "astrophotography-calculator": astrophotography_calculator_handler,
        "celestial-position-calculator": celestial_position_calculator_handler,
    }
    operations_by_skill: dict[str, list[OperationSpec]] = {}
    for operation in _OPERATION_SPECS:
        operations_by_skill.setdefault(operation.logical_skill, []).append(operation)

    return [
        SkillDefinition(
            name=meta["name"],
            display_name=meta["display_name"],
            summary=meta["summary"],
            description=meta["description"],
            input_model=meta["input_model"],
            handler=handlers[meta["name"]],
            allowed_tools=tuple(meta.get("allowed_tools", ())),
            operations=tuple(operations_by_skill.get(meta["name"], ())),
            required_params=tuple(meta.get("required_params", ())),
        )
        for meta in _SKILL_METADATA
    ]


def get_default_skill_registry() -> SkillRegistry:
    """Construct the default high-level skill registry."""
    return SkillRegistry()


def get_skill_definitions() -> List[SkillDefinition]:
    """Return all high-level skill definitions."""
    return get_default_skill_registry().list()


def get_skill_definition(skill_name: str) -> SkillDefinition:
    """Return one high-level skill definition."""
    return get_default_skill_registry().get(skill_name)


def get_skill_specs() -> List[SkillSpec]:
    """Return legacy SkillSpec projections for all skills."""
    definitions = {
        definition.name: definition for definition in get_skill_definitions()
    }
    return [_legacy_spec(meta, definitions[meta["name"]]) for meta in _SKILL_METADATA]


def get_skill_spec(skill_name: str) -> SkillSpec:
    """Return one legacy SkillSpec projection."""
    for spec in get_skill_specs():
        if spec.skill_name == skill_name:
            return spec
    raise KeyError(f"未知技能：{skill_name}")


def get_operation_specs() -> List[OperationSpec]:
    """Return all operation policy specs."""
    return list(_OPERATION_SPECS)


def get_operation_spec(logical_skill: str, operation: str) -> OperationSpec:
    """Return one operation policy spec."""
    for spec in get_operation_specs():
        if spec.logical_skill == logical_skill and spec.operation == operation:
            return spec
    raise KeyError(f"未知 operation：{logical_skill}.{operation}")


def list_operations_for_skill(logical_skill: str) -> List[OperationSpec]:
    """List operations supported by one high-level skill."""
    return [
        spec for spec in get_operation_specs() if spec.logical_skill == logical_skill
    ]


def list_skill_descriptions() -> Dict[str, str]:
    """Return a skill name to summary mapping."""
    return {
        definition.name: definition.summary for definition in get_skill_definitions()
    }


def list_langchain_tool_names() -> List[str]:
    """Return LangChain tool names for all skills."""
    return [definition.display_name for definition in get_skill_definitions()]


def validate_skill_registry(
    specs: Optional[Sequence[SkillSpec]] = None,
    handler_names: Optional[Set[str]] = None,
) -> None:
    """Validate skill registry uniqueness and optional handler names."""
    specs = list(specs or get_skill_specs())
    seen_skill_names: Set[str] = set()
    seen_tool_names: Set[str] = set()
    handler_skill_names: Set[str] = set()

    for spec in specs:
        if not spec.skill_name:
            raise ValueError("Skill registry 校验失败：存在空 skill_name")
        if not spec.langchain_tool_name:
            raise ValueError(
                f"Skill registry 校验失败：{spec.skill_name} 缺少 langchain_tool_name"
            )
        if not spec.summary or not spec.description:
            raise ValueError(f"Skill registry 校验失败：{spec.skill_name} 缺少说明文案")
        if spec.skill_name in seen_skill_names:
            raise ValueError(
                f"Skill registry 校验失败：重复的 skill_name: {spec.skill_name}"
            )
        if spec.langchain_tool_name in seen_tool_names:
            raise ValueError(
                f"Skill registry 校验失败：重复的 LangChain Tool 名称: {spec.langchain_tool_name}"
            )
        seen_skill_names.add(spec.skill_name)
        seen_tool_names.add(spec.langchain_tool_name)
        if spec.route_type == "handler":
            handler_skill_names.add(spec.skill_name)
        elif spec.route_type == "simple" and not spec.mcp_tool_name:
            raise ValueError(
                f"Skill registry 校验失败：simple skill {spec.skill_name} 缺少 mcp_tool_name"
            )

    if handler_names is None:
        return
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


def _legacy_spec(meta: dict[str, Any], definition: SkillDefinition) -> SkillSpec:
    return SkillSpec(
        skill_name=definition.name,
        langchain_tool_name=definition.display_name,
        summary=definition.summary,
        description=definition.description,
        route_type="handler",
        mcp_tool_name=meta.get("mcp_tool_name"),
        param_names=definition.param_names,
        defaults=dict(meta.get("defaults", {})),
        type_conversions=dict(meta.get("type_conversions", {})),
        param_mapping=dict(meta.get("param_mapping", {})),
        special_handling=meta.get("special_handling"),
    )
