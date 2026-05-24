"""
MCP tool response protocol helpers.

All MCP tools must return a JSON string in a fixed envelope:
- success: {"ok": true, "data": ..., "meta": ...}
- error: {"ok": false, "error": {...}, "meta": ...}
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ConfigDict, RootModel, ValidationError

from src.transport.mcp.envelope import (
    SCHEMA_VERSION,
    MCPToolError,
    MCPToolErrorEnvelope,
    MCPToolMeta,
    MCPToolSuccessEnvelope,
    build_meta,
    error_envelope,
    extract_tool_data,
    is_tool_error,
    parse_tool_response,
    serialize_envelope,
)

__all__ = [
    "SCHEMA_VERSION",
    "MCPToolMeta",
    "MCPToolError",
    "MCPToolSuccessEnvelope",
    "MCPToolErrorEnvelope",
    "TOOL_INPUT_MODELS",
    "TOOL_OUTPUT_MODELS",
    "build_meta",
    "validate_tool_input",
    "success_envelope",
    "error_envelope",
    "serialize_envelope",
    "wrap_tool_result",
    "parse_tool_response",
    "is_tool_error",
    "extract_tool_data",
]


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


TOOL_INPUT_MODELS: Dict[str, Type[BaseModel]] = {
    "get_planet_position": PlanetPositionInput,
    "get_altaz": AltAzInput,
    "coordinate_transformation": CoordinateTransformationInput,
    "get_rise_set_times": RiseSetTimesInput,
    "get_current_sky_objects": CurrentSkyObjectsInput,
    "get_astrophysical_object_info": AstrophysicalObjectInfoInput,
    "get_galaxy_data": GalaxyDataInput,
    "get_nasa_apod": NASAApodInput,
    "get_neo_data": NeoDataInput,
    "get_weather": WeatherInput,
    "web_search": WebSearchInput,
    "get_tonight_best": TonightBestInput,
    "get_weekly_events": WeeklyEventsInput,
    "get_monthly_events": MonthlyEventsInput,
}


TOOL_OUTPUT_MODELS: Dict[str, Any] = {
    "get_planet_position": PlanetPositionData,
    "get_altaz": AltAzData,
    "coordinate_transformation": CoordinateTransformationData,
    "get_rise_set_times": RiseSetTimesData,
    "get_current_sky_objects": JsonObjectData,
    "get_astrophysical_object_info": JsonObjectData,
    "get_galaxy_data": JsonObjectData,
    "get_nasa_apod": JsonObjectData,
    "get_neo_data": JsonObjectData,
    "get_weather": JsonObjectData,
    "web_search": JsonObjectData,
    "get_tonight_best": str,
    "get_weekly_events": str,
    "get_monthly_events": str,
}


def validate_tool_input(tool_name: str, payload: Dict[str, Any]) -> BaseModel:
    model_cls = TOOL_INPUT_MODELS.get(tool_name)
    if model_cls is None:
        raise KeyError(f"Unknown tool input model: {tool_name}")
    return model_cls.model_validate(payload)


def _validate_tool_output(tool_name: str, data: Any) -> Any:
    model_cls = TOOL_OUTPUT_MODELS.get(tool_name)
    if model_cls is None:
        return data
    if model_cls is str:
        if not isinstance(data, str):
            raise TypeError(f"{tool_name} expected str output, got {type(data).__name__}")
        return data
    validated = model_cls.model_validate(data)
    return validated.model_dump(mode="json") if isinstance(validated, BaseModel) else validated


def success_envelope(tool_name: str, data: Any, **meta: Any) -> MCPToolSuccessEnvelope:
    return MCPToolSuccessEnvelope(
        ok=True,
        data=_validate_tool_output(tool_name, data),
        meta=build_meta(tool_name, **meta),
    )


def _legacy_error_to_envelope(data: Dict[str, Any], tool_name: str) -> MCPToolErrorEnvelope:
    return error_envelope(
        tool_name=tool_name,
        code=str(data.get("code") or "UNKNOWN_ERROR"),
        message=str(data.get("message") or data.get("error") or "未知错误"),
        details=data.get("details") or {},
    )


def wrap_tool_result(result: Any, tool_name: str) -> str:
    from src.core.errors import ErrorCode

    def _is_agent_error_like(value: Any) -> bool:
        return (
            hasattr(value, "code")
            and hasattr(value, "message")
            and hasattr(value, "details")
        )

    if isinstance(result, str):
        trimmed = result.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                parsed = json.loads(trimmed)
                envelope = parse_tool_response(parsed)
                if envelope is not None:
                    return serialize_envelope(envelope)
                if isinstance(parsed, dict) and parsed.get("error") is True:
                    return serialize_envelope(_legacy_error_to_envelope(parsed, tool_name))
            except (json.JSONDecodeError, TypeError, ValidationError):
                pass
        if trimmed.startswith("错误："):
            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=ErrorCode.TOOL_CALL_FAILED.value,
                    message=trimmed[3:].strip() or trimmed,
                )
            )
        return serialize_envelope(success_envelope(tool_name, result))

    if _is_agent_error_like(result):
        code = getattr(result.code, "value", result.code)
        return serialize_envelope(
            error_envelope(
                tool_name=tool_name,
                code=str(code),
                message=result.message,
                details=getattr(result, "details", {}) or {},
            )
        )

    if isinstance(result, dict) and result.get("error") is True:
        return serialize_envelope(_legacy_error_to_envelope(result, tool_name))

    return serialize_envelope(success_envelope(tool_name, result))
