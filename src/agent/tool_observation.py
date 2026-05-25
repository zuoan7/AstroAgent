"""Agent-local normalization for tool observations.

This module intentionally avoids importing MCP envelope helpers. Agent code only
needs a generic view of ToolResult, dict and JSON/string observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from src.core.errors import ErrorHandler


@dataclass(frozen=True)
class NormalizedObservation:
    """Generic agent-facing view of a tool observation."""

    data: Any = None
    text: str = ""
    is_error: bool = False
    error_code: Optional[str] = None
    error_message: str = ""


def normalize_observation(value: Any) -> NormalizedObservation:
    """Normalize ToolResult, dict and JSON/string observations without MCP parsing."""
    if _is_tool_result_like(value):
        if value.ok:
            return NormalizedObservation(
                data=value.data,
                text=_to_text(value.data),
                is_error=False,
            )
        error = getattr(value, "error", None)
        code = getattr(error, "code", None) or "TOOL_CALL_FAILED"
        message = getattr(error, "message", None) or "工具调用失败"
        details = getattr(error, "details", None) or {}
        return NormalizedObservation(
            data={
                "ok": False,
                "error": {
                    "code": code,
                    "message": message,
                    "details": details,
                },
            },
            text=message,
            is_error=True,
            error_code=str(code),
            error_message=str(message),
        )

    if isinstance(value, dict):
        return _normalize_mapping(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return NormalizedObservation(text="")
        parsed = _try_parse_json(text)
        if isinstance(parsed, dict):
            return _normalize_mapping(parsed)
        return NormalizedObservation(data=value, text=text)

    return NormalizedObservation(data=value, text=_to_text(value))


def observation_to_text(value: Any) -> str:
    """Return display text for an arbitrary observation."""
    return normalize_observation(value).text


def _normalize_mapping(value: dict[str, Any]) -> NormalizedObservation:
    if value.get("ok") is False and isinstance(value.get("error"), dict):
        error = value["error"]
        code = error.get("code")
        message = str(error.get("message") or "工具调用失败")
        return NormalizedObservation(
            data=value,
            text=message,
            is_error=True,
            error_code=str(code) if code else None,
            error_message=message,
        )

    if ErrorHandler.is_error_response(value):
        code = ErrorHandler.extract_error_code(value)
        message = str(value.get("message") or value.get("error") or "工具调用失败")
        return NormalizedObservation(
            data=value,
            text=message,
            is_error=True,
            error_code=code,
            error_message=message,
        )

    if value.get("ok") is True and "data" in value:
        data = value.get("data")
        return NormalizedObservation(data=data, text=_to_text(data))

    return NormalizedObservation(data=value, text=_to_text(value))


def _is_tool_result_like(value: Any) -> bool:
    return (
        hasattr(value, "ok")
        and hasattr(value, "tool_name")
        and hasattr(value, "data")
        and hasattr(value, "error")
    )


def _try_parse_json(text: str) -> Any:
    if not (text.startswith("{") or text.startswith("[")):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
