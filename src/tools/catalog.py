"""原子 MCP 工具目录，集中声明 Agent 工具层可识别的底层工具及参数名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class AtomicToolSpec:
    """单个原子 MCP 工具的静态描述。"""

    name: str
    summary: str = ""
    param_names: List[str] = field(default_factory=list)


_ATOMIC_TOOL_SPECS: tuple[AtomicToolSpec, ...] = (
    AtomicToolSpec(
        name="get_planet_position",
        summary="Planet equatorial position lookup",
        param_names=["planet_name", "observation_time", "latitude", "longitude"],
    ),
    AtomicToolSpec(
        name="get_altaz",
        summary="Planet altitude/azimuth lookup",
        param_names=["planet_name", "observation_time", "latitude", "longitude"],
    ),
    AtomicToolSpec(
        name="coordinate_transformation",
        summary="Coordinate system transformation",
        param_names=["ra", "dec", "epoch", "target_system"],
    ),
    AtomicToolSpec(
        name="get_rise_set_times",
        summary="Rise and set time calculation",
        param_names=["body_name", "latitude", "longitude", "date"],
    ),
    AtomicToolSpec(
        name="get_current_sky_objects",
        summary="Currently visible sky objects",
        param_names=["latitude", "longitude", "date"],
    ),
    AtomicToolSpec(
        name="get_astrophysical_object_info",
        summary="Astrophysical object database lookup",
        param_names=["object_name"],
    ),
    AtomicToolSpec(
        name="get_galaxy_data",
        summary="Galaxy database lookup",
        param_names=["galaxy_name"],
    ),
    AtomicToolSpec(
        name="get_nasa_apod",
        summary="NASA astronomy picture of the day",
        param_names=["date", "hd"],
    ),
    AtomicToolSpec(
        name="get_neo_data",
        summary="NASA near-earth object data",
        param_names=["start_date", "end_date", "limit"],
    ),
    AtomicToolSpec(
        name="get_weather",
        summary="Observation weather lookup",
        param_names=["city", "extensions"],
    ),
    AtomicToolSpec(
        name="web_search",
        summary="External web search",
        param_names=["query", "max_results"],
    ),
    AtomicToolSpec(name="get_tonight_best", summary="Tonight best observing targets"),
    AtomicToolSpec(
        name="get_weekly_events",
        summary="Weekly celestial events",
        param_names=["start_date"],
    ),
    AtomicToolSpec(
        name="get_monthly_events",
        summary="Monthly celestial events",
        param_names=["year", "month"],
    ),
)


class ToolCatalog:
    """Agent 工具层已知的原子 MCP 工具静态目录。"""

    def __init__(self, specs: Optional[Iterable[AtomicToolSpec]] = None) -> None:
        """初始化工具目录，并按工具名建立索引。"""
        self._specs: Dict[str, AtomicToolSpec] = {
            spec.name: spec for spec in (specs or _ATOMIC_TOOL_SPECS)
        }

    def list_specs(self) -> List[AtomicToolSpec]:
        """返回所有原子工具描述。"""
        return list(self._specs.values())

    def list_names(self) -> List[str]:
        """返回所有原子工具名称。"""
        return list(self._specs.keys())

    def has_tool(self, name: str) -> bool:
        """判断指定原子工具是否在目录中注册。"""
        return name in self._specs

    def get_tool(self, name: str) -> AtomicToolSpec:
        """读取指定原子工具描述，不存在时抛出 KeyError。"""
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown MCP atomic tool: {name}") from exc


def get_default_tool_catalog() -> ToolCatalog:
    """构造默认原子工具目录。"""
    return ToolCatalog()
