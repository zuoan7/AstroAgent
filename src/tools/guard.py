from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.catalog import ToolCatalog, get_default_tool_catalog


class ToolGuardViolation(ValueError):
    """Raised when a tool call violates the current runtime policy."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "TOOL_GUARD_REJECTED",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class ToolGuardContext:
    logical_skill: str = ""
    operation: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    required_params: List[str] = field(default_factory=list)
    enforce_allowed_tools: bool = False

    def with_policy(
        self,
        *,
        logical_skill: Optional[str] = None,
        operation: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        forbidden_tools: Optional[List[str]] = None,
        required_params: Optional[List[str]] = None,
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
            required_params=(
                list(self.required_params)
                if required_params is None
                else list(required_params)
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
        params: Optional[Dict[str, Any]] = None,
        context: Optional[ToolGuardContext] = None,
    ) -> None:
        if not self._catalog.has_tool(tool_name):
            raise ToolGuardViolation(
                f"Unknown MCP atomic tool: {tool_name}",
                details={"tool_name": tool_name},
            )

        ctx = context or ToolGuardContext()
        if tool_name in set(ctx.forbidden_tools or []):
            raise ToolGuardViolation(
                f"Tool {tool_name} is forbidden"
                + (f" for {ctx.logical_skill}" if ctx.logical_skill else ""),
                details=self._policy_details(ctx, tool_name),
            )

        allowed = list(ctx.allowed_tools or [])
        if ctx.enforce_allowed_tools and tool_name not in set(allowed):
            raise ToolGuardViolation(
                f"Tool {tool_name} is not allowed"
                + (f" for {ctx.logical_skill}" if ctx.logical_skill else ""),
                details=self._policy_details(ctx, tool_name),
            )

        provided = params or {}
        missing = [
            name
            for name in ctx.required_params
            if provided.get(name) is None or provided.get(name) == ""
        ]
        if missing:
            raise ToolGuardViolation(
                f"Tool {tool_name} missing required params: {', '.join(missing)}",
                code="TOOL_INPUT_VALIDATION_ERROR",
                details={
                    **self._policy_details(ctx, tool_name),
                    "missing_params": missing,
                },
            )

    @staticmethod
    def _policy_details(ctx: ToolGuardContext, tool_name: str) -> Dict[str, Any]:
        return {
            "tool_name": tool_name,
            "logical_skill": ctx.logical_skill,
            "operation": ctx.operation,
            "allowed_tools": list(ctx.allowed_tools),
            "forbidden_tools": list(ctx.forbidden_tools),
            "required_params": list(ctx.required_params),
            "enforce_allowed_tools": ctx.enforce_allowed_tools,
        }
