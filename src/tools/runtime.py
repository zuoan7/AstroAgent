from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.core.mcp_protocol import error_envelope, serialize_envelope
from src.tools.guard import ToolGuard, ToolGuardContext, ToolGuardViolation


class ToolRuntime:
    """Guarded runtime facade for MCP atomic tools.

    The backend is intentionally duck-typed so existing MCPClient-like fakes keep
    working in tests and handlers can migrate without changing their call shape.
    """

    def __init__(
        self,
        backend: Any,
        *,
        guard: Optional[ToolGuard] = None,
        context: Optional[ToolGuardContext] = None,
    ) -> None:
        self._backend = backend
        self._guard = guard or ToolGuard()
        self._context = context or ToolGuardContext()

    @property
    def guard(self) -> ToolGuard:
        return self._guard

    @property
    def context(self) -> ToolGuardContext:
        return self._context

    def with_context(
        self,
        *,
        logical_skill: Optional[str] = None,
        operation: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        forbidden_tools: Optional[List[str]] = None,
        enforce_allowed_tools: Optional[bool] = None,
    ) -> "ToolRuntime":
        return ToolRuntime(
            self._backend,
            guard=self._guard,
            context=self._context.with_policy(
                logical_skill=logical_skill,
                operation=operation,
                allowed_tools=allowed_tools,
                forbidden_tools=forbidden_tools,
                enforce_allowed_tools=enforce_allowed_tools,
            ),
        )

    def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        violation = self._validate(tool_name)
        if violation is not None:
            return self._guard_error(tool_name, violation)
        return self._backend.call_tool(tool_name, **kwargs)

    async def async_call_tool(self, tool_name: str, **kwargs: Any) -> str:
        violation = self._validate(tool_name)
        if violation is not None:
            return self._guard_error(tool_name, violation)
        if hasattr(self._backend, "async_call_tool"):
            return await self._backend.async_call_tool(tool_name, **kwargs)
        return await asyncio.to_thread(self._backend.call_tool, tool_name, **kwargs)

    def call_tools_parallel(self, calls: List[Dict[str, Any]]) -> List[str]:
        if not calls:
            return []

        violations: Dict[int, ToolGuardViolation] = {}
        valid_calls: List[Dict[str, Any]] = []
        valid_indices: List[int] = []

        for index, call in enumerate(calls):
            tool_name = str(call.get("tool_name", ""))
            violation = self._validate(tool_name)
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
        if hasattr(self._backend, "prewarm"):
            return bool(self._backend.prewarm())
        return True

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        if hasattr(self._backend, "get_runtime_metrics_snapshot"):
            return self._backend.get_runtime_metrics_snapshot()
        return {}

    def shutdown(self) -> None:
        if hasattr(self._backend, "shutdown"):
            self._backend.shutdown()

    def _validate(self, tool_name: str) -> Optional[ToolGuardViolation]:
        try:
            self._guard.validate_tool_call(tool_name, context=self._context)
        except ToolGuardViolation as exc:
            return exc
        return None

    def _guard_error(self, tool_name: str, violation: ToolGuardViolation) -> str:
        return serialize_envelope(
            error_envelope(
                tool_name=tool_name,
                code="TOOL_GUARD_REJECTED",
                message=str(violation),
                details={
                    "logical_skill": self._context.logical_skill,
                    "operation": self._context.operation,
                    "allowed_tools": list(self._context.allowed_tools),
                    "forbidden_tools": list(self._context.forbidden_tools),
                    "enforce_allowed_tools": self._context.enforce_allowed_tools,
                },
            )
        )
