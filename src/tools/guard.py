from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.tools.catalog import ToolCatalog, get_default_tool_catalog


class ToolGuardViolation(ValueError):
    """Raised when a tool call violates the current runtime policy."""


@dataclass(frozen=True)
class ToolGuardContext:
    logical_skill: str = ""
    operation: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    enforce_allowed_tools: bool = False

    def with_policy(
        self,
        *,
        logical_skill: Optional[str] = None,
        operation: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        forbidden_tools: Optional[List[str]] = None,
        enforce_allowed_tools: Optional[bool] = None,
    ) -> "ToolGuardContext":
        return ToolGuardContext(
            logical_skill=self.logical_skill if logical_skill is None else logical_skill,
            operation=self.operation if operation is None else operation,
            allowed_tools=(
                list(self.allowed_tools)
                if allowed_tools is None
                else list(allowed_tools)
            ),
            forbidden_tools=(
                list(self.forbidden_tools)
                if forbidden_tools is None
                else list(forbidden_tools)
            ),
            enforce_allowed_tools=(
                self.enforce_allowed_tools
                if enforce_allowed_tools is None
                else bool(enforce_allowed_tools)
            ),
        )


class ToolGuard:
    """Validates MCP atomic tool calls before they leave the agent process."""

    def __init__(self, catalog: Optional[ToolCatalog] = None) -> None:
        self._catalog = catalog or get_default_tool_catalog()

    @property
    def catalog(self) -> ToolCatalog:
        return self._catalog

    def validate_tool_call(
        self,
        tool_name: str,
        *,
        context: Optional[ToolGuardContext] = None,
    ) -> None:
        if not self._catalog.has_tool(tool_name):
            raise ToolGuardViolation(f"Unknown MCP atomic tool: {tool_name}")

        ctx = context or ToolGuardContext()
        if tool_name in set(ctx.forbidden_tools or []):
            raise ToolGuardViolation(
                f"Tool {tool_name} is forbidden"
                + (f" for {ctx.logical_skill}" if ctx.logical_skill else "")
            )

        allowed = list(ctx.allowed_tools or [])
        if ctx.enforce_allowed_tools and tool_name not in set(allowed):
            raise ToolGuardViolation(
                f"Tool {tool_name} is not allowed"
                + (f" for {ctx.logical_skill}" if ctx.logical_skill else "")
            )
