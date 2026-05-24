"""
Compatibility facade for MCP tool response protocol helpers.

The canonical tool schemas now live in ``src.tools.registry``. This module keeps
the historical import path stable for MCP server code and tests.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from pydantic import BaseModel, ValidationError

from src.tools.registry import (
    TOOL_INPUT_MODELS,
    TOOL_OUTPUT_MODELS,
    AltAzData,
    AltAzInput,
    AstrophysicalObjectInfoInput,
    CoordinateTransformationData,
    CoordinateTransformationInput,
    CurrentSkyObjectsInput,
    GalaxyDataInput,
    JsonObjectData,
    MonthlyEventsInput,
    NASAApodInput,
    NeoDataInput,
    PlanetPositionData,
    PlanetPositionInput,
    RiseSetTimesData,
    RiseSetTimesInput,
    TonightBestInput,
    WeatherInput,
    WebSearchInput,
    WeeklyEventsInput,
)
from src.tools.results import validate_tool_data
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
    "PlanetPositionInput",
    "AltAzInput",
    "CoordinateTransformationInput",
    "RiseSetTimesInput",
    "CurrentSkyObjectsInput",
    "AstrophysicalObjectInfoInput",
    "GalaxyDataInput",
    "NASAApodInput",
    "NeoDataInput",
    "WeatherInput",
    "WebSearchInput",
    "TonightBestInput",
    "WeeklyEventsInput",
    "MonthlyEventsInput",
    "PlanetPositionData",
    "AltAzData",
    "CoordinateTransformationData",
    "RiseSetTimesData",
    "JsonObjectData",
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


def validate_tool_input(tool_name: str, payload: Dict[str, Any]) -> BaseModel:
    model_cls = TOOL_INPUT_MODELS.get(tool_name)
    if model_cls is None:
        raise KeyError(f"Unknown tool input model: {tool_name}")
    return model_cls.model_validate(payload)


def _validate_tool_output(tool_name: str, data: Any) -> Any:
    model_cls = TOOL_OUTPUT_MODELS.get(tool_name)
    return validate_tool_data(model_cls, data)


def success_envelope(tool_name: str, data: Any, **meta: Any) -> MCPToolSuccessEnvelope:
    return MCPToolSuccessEnvelope(
        ok=True,
        data=_validate_tool_output(tool_name, data),
        meta=build_meta(tool_name, **meta),
    )


def _legacy_error_to_envelope(
    data: Dict[str, Any],
    tool_name: str,
) -> MCPToolErrorEnvelope:
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
                    return serialize_envelope(
                        _legacy_error_to_envelope(parsed, tool_name)
                    )
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
