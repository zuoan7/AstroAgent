"""Tool definition contracts for atomic MCP tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Type

from pydantic import BaseModel


class ToolTransport(str, Enum):
    """Supported transport backends for atomic tools."""

    MCP = "mcp"


class ToolCostClass(str, Enum):
    """Planner-facing rough execution cost classification."""

    FAST = "fast"
    NORMAL = "normal"
    EXPENSIVE = "expensive"


@dataclass(frozen=True)
class ToolDefinition:
    """Static contract for one atomic tool."""

    name: str
    summary: str
    input_model: Type[BaseModel]
    output_model: Any
    description: str = ""
    transport: ToolTransport = ToolTransport.MCP
    cost_class: ToolCostClass = ToolCostClass.NORMAL
    side_effect: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def param_names(self) -> list[str]:
        """Return input field names for legacy callers."""
        return list(self.input_model.model_fields.keys())

