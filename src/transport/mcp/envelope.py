"""
MCP tool response envelope helpers.

This module owns the transport-level response envelope shape. Tool input/output
schema helpers live in ``src.tools.protocol``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA_VERSION = "1.0"


class MCPToolMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool_name: str
    schema_version: str = SCHEMA_VERSION


class MCPToolError(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class MCPToolSuccessEnvelope(BaseModel):
    ok: Literal[True]
    data: Any
    meta: MCPToolMeta


class MCPToolErrorEnvelope(BaseModel):
    ok: Literal[False]
    error: MCPToolError
    meta: MCPToolMeta


def build_meta(tool_name: str, **extra: Any) -> MCPToolMeta:
    return MCPToolMeta(tool_name=tool_name, **extra)


def error_envelope(
    tool_name: str,
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    **meta: Any,
) -> MCPToolErrorEnvelope:
    return MCPToolErrorEnvelope(
        ok=False,
        error=MCPToolError(code=code, message=message, details=details or {}),
        meta=build_meta(tool_name, **meta),
    )


def serialize_envelope(envelope: MCPToolSuccessEnvelope | MCPToolErrorEnvelope) -> str:
    if isinstance(envelope, dict):
        return json.dumps(envelope, ensure_ascii=False)
    return json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False)


def parse_tool_response(
    payload: Any,
) -> Optional[MCPToolSuccessEnvelope | MCPToolErrorEnvelope]:
    try:
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            return None
        if payload.get("ok") is True:
            return MCPToolSuccessEnvelope.model_validate(payload)
        if payload.get("ok") is False:
            return MCPToolErrorEnvelope.model_validate(payload)
        return None
    except (json.JSONDecodeError, TypeError, ValidationError):
        return None


def is_tool_error(payload: Any) -> bool:
    envelope = parse_tool_response(payload)
    return isinstance(envelope, MCPToolErrorEnvelope)


def extract_tool_data(payload: Any) -> Any:
    envelope = parse_tool_response(payload)
    if isinstance(envelope, MCPToolSuccessEnvelope):
        return envelope.data
    if isinstance(envelope, MCPToolErrorEnvelope):
        return None
    return payload
