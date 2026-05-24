"""Canonical registry for atomic tool definitions and schemas."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Type

from pydantic import BaseModel, ConfigDict, RootModel

from src.tools.definition import ToolCostClass, ToolDefinition


class PlanetPositionInput(BaseModel):
    planet_name: str
    observation_time: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class AltAzInput(PlanetPositionInput):
    latitude: float
    longitude: float


class CoordinateTransformationInput(BaseModel):
    ra: float
    dec: float
    epoch: str = "J2000"
    target_system: str = "fk5"


class RiseSetTimesInput(BaseModel):
    body_name: str
    latitude: float
    longitude: float
    date: Optional[str] = None


class CurrentSkyObjectsInput(BaseModel):
    latitude: float
    longitude: float
    date: Optional[str] = None


class AstrophysicalObjectInfoInput(BaseModel):
    object_name: str


class GalaxyDataInput(BaseModel):
    galaxy_name: str


class NASAApodInput(BaseModel):
    date: Optional[str] = None
    hd: bool = False


class NeoDataInput(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = 10


class WeatherInput(BaseModel):
    city: Optional[str] = None
    extensions: str = "base"


class WebSearchInput(BaseModel):
    query: str
    max_results: int = 5


class TonightBestInput(BaseModel):
    pass


class WeeklyEventsInput(BaseModel):
    start_date: Optional[str] = None


class MonthlyEventsInput(BaseModel):
    year: Optional[int] = None
    month: Optional[int] = None


class PlanetPositionData(BaseModel):
    model_config = ConfigDict(extra="allow")

    ra_hours: Optional[float] = None
    ra_degrees: Optional[float] = None
    dec: Optional[float] = None
    distance_au: Optional[float] = None
    altitude: Optional[float] = None
    azimuth: Optional[float] = None


class AltAzData(BaseModel):
    model_config = ConfigDict(extra="allow")

    planet: str
    altitude: float
    azimuth: float
    distance_au: float


class CoordinateTransformationData(BaseModel):
    model_config = ConfigDict(extra="allow")

    ra_hours: float
    ra_degrees: float
    dec: float


class RiseSetTimesData(BaseModel):
    model_config = ConfigDict(extra="allow")

    rise_time: Any = None
    set_time: Any = None


class JsonObjectData(RootModel[Dict[str, Any]]):
    pass


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


TOOL_INPUT_MODELS: Dict[str, Type[BaseModel]] = {
    definition.name: definition.input_model
    for definition in _TOOL_DEFINITIONS
}


TOOL_OUTPUT_MODELS: Dict[str, Any] = {
    definition.name: definition.output_model
    for definition in _TOOL_DEFINITIONS
}

