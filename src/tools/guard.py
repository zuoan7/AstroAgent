"""工具调用防护层，在请求离开 Agent 进程前校验工具名、策略和必填参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.registry import ToolRegistry, get_default_tool_registry


class ToolGuardViolation(ValueError):
    """工具调用违反当前运行时策略时抛出的异常。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "TOOL_GUARD_REJECTED",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """初始化工具防护异常，保留错误码和结构化详情。"""
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class ToolGuardContext:
    """一次工具调用的策略上下文，描述允许工具、禁用工具和必填参数。"""

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
        """基于当前上下文派生一个带局部策略的新上下文。"""
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
    """在原子 MCP 工具调用离开 Agent 进程前执行校验。"""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        *,
        catalog: Optional[Any] = None,
    ) -> None:
        """初始化工具防护器和可用工具注册表。"""
        self._registry = registry or catalog or get_default_tool_registry()

    @property
    def registry(self) -> ToolRegistry:
        """返回当前工具防护器使用的工具注册表。"""
        return self._registry

    @property
    def catalog(self) -> ToolRegistry:
        """兼容旧属性名，返回当前工具注册表。"""
        return self._registry

    def validate_tool_call(
        self,
        tool_name: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[ToolGuardContext] = None,
    ) -> None:
        """校验工具名、允许/禁用策略和必填参数。"""
        if not self._registry.has_tool(tool_name):
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
        """生成工具防护拒绝时附带的策略详情。"""
        return {
            "tool_name": tool_name,
            "logical_skill": ctx.logical_skill,
            "operation": ctx.operation,
            "allowed_tools": list(ctx.allowed_tools),
            "forbidden_tools": list(ctx.forbidden_tools),
            "required_params": list(ctx.required_params),
            "enforce_allowed_tools": ctx.enforce_allowed_tools,
        }
