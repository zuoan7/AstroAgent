"""Canonical registry for atomic tool definitions."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from src.tools.definition import ToolCostClass, ToolDefinition
from src.tools.schemas.astronomy import (
    AltAzData,
    AltAzInput,
    AstrophysicalObjectInfoInput,
    CoordinateTransformationData,
    CoordinateTransformationInput,
    CurrentSkyObjectsInput,
    GalaxyDataInput,
    PlanetPositionData,
    PlanetPositionInput,
    RiseSetTimesData,
    RiseSetTimesInput,
)
from src.tools.schemas.common import JsonObjectData
from src.tools.schemas.events import (
    MonthlyEventsInput,
    TonightBestInput,
    WeeklyEventsInput,
)
from src.tools.schemas.nasa import NASAApodInput, NeoDataInput
from src.tools.schemas.search import WebSearchInput
from src.tools.schemas.weather import WeatherInput

_TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="get_planet_position",
        summary="Planet equatorial position lookup",
        input_model=PlanetPositionInput,
        output_model=PlanetPositionData,
    ),
    ToolDefinition(
        name="get_altaz",
        summary="Planet altitude/azimuth lookup",
        input_model=AltAzInput,
        output_model=AltAzData,
    ),
    ToolDefinition(
        name="coordinate_transformation",
        summary="Coordinate system transformation",
        input_model=CoordinateTransformationInput,
        output_model=CoordinateTransformationData,
    ),
    ToolDefinition(
        name="get_rise_set_times",
        summary="Rise and set time calculation",
        input_model=RiseSetTimesInput,
        output_model=RiseSetTimesData,
    ),
    ToolDefinition(
        name="get_current_sky_objects",
        summary="Currently visible sky objects",
        input_model=CurrentSkyObjectsInput,
        output_model=JsonObjectData,
    ),
    ToolDefinition(
        name="get_astrophysical_object_info",
        summary="Astrophysical object database lookup",
        input_model=AstrophysicalObjectInfoInput,
        output_model=JsonObjectData,
    ),
    ToolDefinition(
        name="get_galaxy_data",
        summary="Galaxy database lookup",
        input_model=GalaxyDataInput,
        output_model=JsonObjectData,
    ),
    ToolDefinition(
        name="get_nasa_apod",
        summary="NASA astronomy picture of the day",
        input_model=NASAApodInput,
        output_model=JsonObjectData,
        tags=("react-exposed",),
    ),
    ToolDefinition(
        name="get_neo_data",
        summary="NASA near-earth object data",
        input_model=NeoDataInput,
        output_model=JsonObjectData,
    ),
    ToolDefinition(
        name="get_weather",
        summary="Observation weather lookup",
        input_model=WeatherInput,
        output_model=JsonObjectData,
        tags=("react-exposed",),
    ),
    ToolDefinition(
        name="web_search",
        summary="External web search",
        input_model=WebSearchInput,
        output_model=JsonObjectData,
        cost_class=ToolCostClass.EXPENSIVE,
        side_effect=True,
        tags=("react-exposed",),
    ),
    ToolDefinition(
        name="get_tonight_best",
        summary="Tonight best observing targets",
        input_model=TonightBestInput,
        output_model=str,
        cost_class=ToolCostClass.FAST,
    ),
    ToolDefinition(
        name="get_weekly_events",
        summary="Weekly celestial events",
        input_model=WeeklyEventsInput,
        output_model=str,
    ),
    ToolDefinition(
        name="get_monthly_events",
        summary="Monthly celestial events",
        input_model=MonthlyEventsInput,
        output_model=str,
    ),
)


class ToolRegistry:
    """Read-only registry for atomic MCP tool definitions."""

    def __init__(self, definitions: Optional[Iterable[ToolDefinition]] = None) -> None:
        self._definitions: Dict[str, ToolDefinition] = {
            definition.name: definition
            for definition in (definitions or _TOOL_DEFINITIONS)
        }

    def list_definitions(self) -> list[ToolDefinition]:
        """Return all tool definitions."""
        return list(self._definitions.values())

    def list_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._definitions.keys())

    def has_tool(self, name: str) -> bool:
        """Return whether a tool name is registered."""
        return name in self._definitions

    def get_tool(self, name: str) -> ToolDefinition:
        """Return one tool definition or raise KeyError."""
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"Unknown MCP atomic tool: {name}") from exc


def get_default_tool_registry() -> ToolRegistry:
    """Construct the default atomic tool registry."""
    return ToolRegistry()
