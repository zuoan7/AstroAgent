from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class AtomicToolSpec:
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
    """Static catalog for MCP atomic tools known to the agent layer."""

    def __init__(self, specs: Optional[Iterable[AtomicToolSpec]] = None) -> None:
        self._specs: Dict[str, AtomicToolSpec] = {
            spec.name: spec for spec in (specs or _ATOMIC_TOOL_SPECS)
        }

    def list_specs(self) -> List[AtomicToolSpec]:
        return list(self._specs.values())

    def list_names(self) -> List[str]:
        return list(self._specs.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._specs

    def get_tool(self, name: str) -> AtomicToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown MCP atomic tool: {name}") from exc


def get_default_tool_catalog() -> ToolCatalog:
    return ToolCatalog()
