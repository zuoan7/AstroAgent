"""受防护的原子工具运行时门面，统一校验参数并转发到 MCPClient 或兼容后端。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.core.mcp_protocol import TOOL_INPUT_MODELS, error_envelope, serialize_envelope
from src.tools.guard import ToolGuard, ToolGuardContext, ToolGuardViolation


class ToolRuntime:
    """带防护的 MCP 原子工具运行时门面。

    backend 采用鸭子类型，方便测试替身和 MCPClient 共用同一调用形状。
    """

    def __init__(
        self,
        backend: Any,
        *,
        guard: Optional[ToolGuard] = None,
        context: Optional[ToolGuardContext] = None,
    ) -> None:
        """初始化运行时后端、防护器和当前策略上下文。"""
        self._backend = backend
        self._guard = guard or ToolGuard()
        self._context = context or ToolGuardContext()

    @property
    def guard(self) -> ToolGuard:
        """返回当前使用的工具防护器。"""
        return self._guard

    @property
    def context(self) -> ToolGuardContext:
        """返回当前工具调用策略上下文。"""
        return self._context

    def with_context(
        self,
        *,
        logical_skill: Optional[str] = None,
        operation: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        forbidden_tools: Optional[List[str]] = None,
        required_params: Optional[List[str]] = None,
        enforce_allowed_tools: Optional[bool] = None,
    ) -> "ToolRuntime":
        """派生一个带新策略上下文的 ToolRuntime。"""
        return ToolRuntime(
            self._backend,
            guard=self._guard,
            context=self._context.with_policy(
                logical_skill=logical_skill,
                operation=operation,
                allowed_tools=allowed_tools,
                forbidden_tools=forbidden_tools,
                required_params=required_params,
                enforce_allowed_tools=enforce_allowed_tools,
            ),
        )

    def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        """同步校验并调用单个原子工具。"""
        violation = self._validate(tool_name, kwargs)
        if violation is not None:
            return self._guard_error(tool_name, violation)
        return self._backend.call_tool(tool_name, **kwargs)

    async def async_call_tool(self, tool_name: str, **kwargs: Any) -> str:
        """异步校验并调用单个原子工具。"""
        violation = self._validate(tool_name, kwargs)
        if violation is not None:
            return self._guard_error(tool_name, violation)
        if hasattr(self._backend, "async_call_tool"):
            return await self._backend.async_call_tool(tool_name, **kwargs)
        return await asyncio.to_thread(self._backend.call_tool, tool_name, **kwargs)

    def call_tools_parallel(self, calls: List[Dict[str, Any]]) -> List[str]:
        """批量校验并并行调用多个原子工具，失败项返回标准错误 envelope。"""
        if not calls:
            return []

        violations: Dict[int, ToolGuardViolation] = {}
        valid_calls: List[Dict[str, Any]] = []
        valid_indices: List[int] = []

        for index, call in enumerate(calls):
            tool_name = str(call.get("tool_name", ""))
            violation = self._validate(tool_name, call.get("kwargs", {}) or {})
            if violation is not None:
                violations[index] = violation
                continue
            valid_calls.append(call)
            valid_indices.append(index)

        results: List[Optional[str]] = [None] * len(calls)
        if valid_calls:
            if hasattr(self._backend, "call_tools_parallel"):
                valid_results = self._backend.call_tools_parallel(valid_calls)
            else:
                valid_results = [
                    self._backend.call_tool(
                        call["tool_name"],
                        **call.get("kwargs", {}),
                    )
                    for call in valid_calls
                ]
            for index, result in zip(valid_indices, valid_results):
                results[index] = result

        for index, violation in violations.items():
            tool_name = str(calls[index].get("tool_name", "unknown_tool"))
            results[index] = self._guard_error(tool_name, violation)

        return [result or "" for result in results]

    def prewarm(self) -> bool:
        """预热底层后端连接。"""
        if hasattr(self._backend, "prewarm"):
            return bool(self._backend.prewarm())
        return True

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        """读取底层后端运行时指标快照。"""
        if hasattr(self._backend, "get_runtime_metrics_snapshot"):
            return self._backend.get_runtime_metrics_snapshot()
        return {}

    def shutdown(self) -> None:
        """关闭底层后端连接。"""
        if hasattr(self._backend, "shutdown"):
            self._backend.shutdown()

    def _validate(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
    ) -> Optional[ToolGuardViolation]:
        """执行工具目录、防护策略和 pydantic 输入模型校验。"""
        try:
            self._guard.validate_tool_call(
                tool_name,
                params=kwargs,
                context=self._context,
            )
            input_model = TOOL_INPUT_MODELS.get(tool_name)
            if input_model is not None:
                input_model(**kwargs)
        except ToolGuardViolation as exc:
            return exc
        except ValidationError as exc:
            return ToolGuardViolation(
                f"Invalid input for {tool_name}: {exc.errors()}",
                code="TOOL_INPUT_VALIDATION_ERROR",
                details={
                    "tool_name": tool_name,
                    "validation_errors": exc.errors(),
                },
            )
        return None

    def _guard_error(self, tool_name: str, violation: ToolGuardViolation) -> str:
        """把防护拒绝转换为统一 MCP 错误 envelope。"""
        details = {
            "logical_skill": self._context.logical_skill,
            "operation": self._context.operation,
            "allowed_tools": list(self._context.allowed_tools),
            "forbidden_tools": list(self._context.forbidden_tools),
            "required_params": list(self._context.required_params),
            "enforce_allowed_tools": self._context.enforce_allowed_tools,
        }
        details.update(getattr(violation, "details", {}) or {})
        return serialize_envelope(
            error_envelope(
                tool_name=tool_name,
                code=getattr(violation, "code", "TOOL_GUARD_REJECTED"),
                message=str(violation),
                details=details,
            )
        )
