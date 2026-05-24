"""Structured result objects for atomic tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from src.transport.mcp.envelope import (
    MCPToolErrorEnvelope,
    MCPToolSuccessEnvelope,
    parse_tool_response,
)


@dataclass(frozen=True)
class ToolError:
    """Structured tool error details."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """Structured wrapper for one atomic tool call result."""

    ok: bool
    tool_name: str
    data: Any = None
    error: ToolError | None = None
    raw_envelope: str | None = None
    latency_ms: float = 0.0

    @classmethod
    def from_raw(
        cls,
        tool_name: str,
        raw: Any,
        *,
        output_model: Any = None,
        latency_ms: float = 0.0,
    ) -> "ToolResult":
        """Build a ToolResult from a raw envelope or legacy backend value."""
        envelope = parse_tool_response(raw)
        if isinstance(envelope, MCPToolErrorEnvelope):
            return cls(
                ok=False,
                tool_name=tool_name,
                error=ToolError(
                    code=envelope.error.code,
                    message=envelope.error.message,
                    details=dict(envelope.error.details or {}),
                ),
                raw_envelope=str(raw),
                latency_ms=latency_ms,
            )

        if isinstance(envelope, MCPToolSuccessEnvelope):
            try:
                data = validate_tool_data(output_model, envelope.data)
            except (TypeError, ValidationError) as exc:
                return cls(
                    ok=False,
                    tool_name=tool_name,
                    error=ToolError(
                        code="TOOL_OUTPUT_VALIDATION_ERROR",
                        message=str(exc),
                        details={"tool_name": tool_name},
                    ),
                    raw_envelope=str(raw),
                    latency_ms=latency_ms,
                )
            return cls(
                ok=True,
                tool_name=tool_name,
                data=data,
                raw_envelope=str(raw),
                latency_ms=latency_ms,
            )

        return cls(
            ok=True,
            tool_name=tool_name,
            data=raw,
            raw_envelope=str(raw) if raw is not None else None,
            latency_ms=latency_ms,
        )

    @classmethod
    def from_error(
        cls,
        tool_name: str,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        raw_envelope: str | None = None,
        latency_ms: float = 0.0,
    ) -> "ToolResult":
        """Build a failed ToolResult."""
        return cls(
            ok=False,
            tool_name=tool_name,
            error=ToolError(code=code, message=message, details=details or {}),
            raw_envelope=raw_envelope,
            latency_ms=latency_ms,
        )


def validate_tool_data(output_model: Any, data: Any) -> Any:
    """Validate and normalize tool output data with a registered output model."""
    if output_model is None:
        return data
    if output_model is str:
        if not isinstance(data, str):
            raise TypeError(f"expected str output, got {type(data).__name__}")
        return data
    if isinstance(output_model, type) and issubclass(output_model, BaseModel):
        validated = output_model.model_validate(data)
        return validated.model_dump(mode="json")
    return data

