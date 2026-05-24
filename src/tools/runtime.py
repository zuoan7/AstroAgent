"""Guarded ToolKit for atomic tool invocation."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.tools.guard import ToolGuard, ToolGuardContext, ToolGuardViolation
from src.tools.registry import ToolRegistry
from src.tools.results import ToolResult
from src.transport.mcp.envelope import error_envelope, serialize_envelope


class ToolKit:
    """Guarded runtime facade for atomic MCP tools."""

    def __init__(
        self,
        backend: Any,
        *,
        guard: Optional[ToolGuard] = None,
        context: Optional[ToolGuardContext] = None,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self._backend = backend
        self._guard = guard or ToolGuard(registry=registry)
        self._context = context or ToolGuardContext()

    @property
    def guard(self) -> ToolGuard:
        """Return the active tool guard."""
        return self._guard

    @property
    def context(self) -> ToolGuardContext:
        """Return the active tool policy context."""
        return self._context

    @property
    def registry(self) -> ToolRegistry:
        """Return the active tool registry."""
        return self._guard.registry

    def list(self) -> list[Any]:
        """Return all registered tool definitions."""
        return self.registry.list_definitions()

    def get(self, name: str) -> Any:
        """Return one registered tool definition."""
        return self.registry.get_tool(name)

    def with_policy(
        self,
        *,
        logical_skill: Optional[str] = None,
        operation: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        forbidden_tools: Optional[List[str]] = None,
        required_params: Optional[List[str]] = None,
        enforce_allowed_tools: Optional[bool] = None,
    ) -> "ToolKit":
        """Derive a ToolKit with a refined policy context."""
        return self.__class__(
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

    def with_context(
        self,
        *,
        logical_skill: Optional[str] = None,
        operation: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        forbidden_tools: Optional[List[str]] = None,
        required_params: Optional[List[str]] = None,
        enforce_allowed_tools: Optional[bool] = None,
    ) -> "ToolKit":
        """Legacy alias for with_policy()."""
        return self.with_policy(
            logical_skill=logical_skill,
            operation=operation,
            allowed_tools=allowed_tools,
            forbidden_tools=forbidden_tools,
            required_params=required_params,
            enforce_allowed_tools=enforce_allowed_tools,
        )

    def invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Validate and synchronously invoke one atomic tool."""
        started = time.perf_counter()
        violation = self._validate(tool_name, kwargs)
        if violation is not None:
            raw = self._guard_error(tool_name, violation)
            return ToolResult.from_raw(
                tool_name,
                raw,
                output_model=self._output_model(tool_name),
                latency_ms=self._elapsed_ms(started),
            )

        raw = self._backend.call_tool(tool_name, **kwargs)
        return ToolResult.from_raw(
            tool_name,
            raw,
            output_model=self._output_model(tool_name),
            latency_ms=self._elapsed_ms(started),
        )

    async def ainvoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Validate and asynchronously invoke one atomic tool."""
        started = time.perf_counter()
        violation = self._validate(tool_name, kwargs)
        if violation is not None:
            raw = self._guard_error(tool_name, violation)
            return ToolResult.from_raw(
                tool_name,
                raw,
                output_model=self._output_model(tool_name),
                latency_ms=self._elapsed_ms(started),
            )

        if hasattr(self._backend, "async_call_tool"):
            raw = await self._backend.async_call_tool(tool_name, **kwargs)
        else:
            raw = await asyncio.to_thread(self._backend.call_tool, tool_name, **kwargs)
        return ToolResult.from_raw(
            tool_name,
            raw,
            output_model=self._output_model(tool_name),
            latency_ms=self._elapsed_ms(started),
        )

    def invoke_parallel(self, calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Validate and invoke a batch of atomic tools."""
        if not calls:
            return []

        started_by_index = [time.perf_counter() for _ in calls]
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

        results: List[Optional[ToolResult]] = [None] * len(calls)
        if valid_calls:
            raw_results = self._call_backend_parallel(valid_calls)
            for index, raw in zip(valid_indices, raw_results):
                tool_name = str(calls[index].get("tool_name", "unknown_tool"))
                results[index] = ToolResult.from_raw(
                    tool_name,
                    raw,
                    output_model=self._output_model(tool_name),
                    latency_ms=self._elapsed_ms(started_by_index[index]),
                )

        for index, violation in violations.items():
            tool_name = str(calls[index].get("tool_name", "unknown_tool"))
            raw = self._guard_error(tool_name, violation)
            results[index] = ToolResult.from_raw(
                tool_name,
                raw,
                output_model=self._output_model(tool_name),
                latency_ms=self._elapsed_ms(started_by_index[index]),
            )

        return [
            result
            or ToolResult.from_error(
                str(calls[index].get("tool_name", "unknown_tool")),
                code="TOOL_RESULT_MISSING",
                message="Tool result missing",
            )
            for index, result in enumerate(results)
        ]

    def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        """Legacy sync API returning raw envelope strings."""
        violation = self._validate(tool_name, kwargs)
        if violation is not None:
            return self._guard_error(tool_name, violation)
        return self._backend.call_tool(tool_name, **kwargs)

    async def async_call_tool(self, tool_name: str, **kwargs: Any) -> str:
        """Legacy async API returning raw envelope strings."""
        violation = self._validate(tool_name, kwargs)
        if violation is not None:
            return self._guard_error(tool_name, violation)
        if hasattr(self._backend, "async_call_tool"):
            return await self._backend.async_call_tool(tool_name, **kwargs)
        return await asyncio.to_thread(self._backend.call_tool, tool_name, **kwargs)

    def call_tools_parallel(self, calls: List[Dict[str, Any]]) -> List[str]:
        """Legacy batch API returning raw envelope strings."""
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
            raw_results = self._call_backend_parallel(valid_calls)
            for index, raw in zip(valid_indices, raw_results):
                results[index] = raw

        for index, violation in violations.items():
            tool_name = str(calls[index].get("tool_name", "unknown_tool"))
            results[index] = self._guard_error(tool_name, violation)

        return [result or "" for result in results]

    def prewarm(self) -> bool:
        """Prewarm the backend connection if supported."""
        if hasattr(self._backend, "prewarm"):
            return bool(self._backend.prewarm())
        return True

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        """Return backend runtime metrics if supported."""
        if hasattr(self._backend, "get_runtime_metrics_snapshot"):
            return self._backend.get_runtime_metrics_snapshot()
        return {}

    def shutdown(self) -> None:
        """Shutdown backend resources if supported."""
        if hasattr(self._backend, "shutdown"):
            self._backend.shutdown()

    def _call_backend_parallel(self, calls: List[Dict[str, Any]]) -> List[str]:
        if hasattr(self._backend, "call_tools_parallel"):
            return self._backend.call_tools_parallel(calls)
        return [
            self._backend.call_tool(
                call["tool_name"],
                **call.get("kwargs", {}),
            )
            for call in calls
        ]

    def _validate(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
    ) -> Optional[ToolGuardViolation]:
        """Validate registry membership, policy and pydantic input."""
        try:
            self._guard.validate_tool_call(
                tool_name,
                params=kwargs,
                context=self._context,
            )
            self.registry.get_tool(tool_name).input_model(**kwargs)
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
        except KeyError as exc:
            return ToolGuardViolation(
                str(exc),
                details={"tool_name": tool_name},
            )
        return None

    def _output_model(self, tool_name: str) -> Any:
        try:
            return self.registry.get_tool(tool_name).output_model
        except KeyError:
            return None

    def _guard_error(self, tool_name: str, violation: ToolGuardViolation) -> str:
        """Convert a guard rejection to a standard MCP error envelope."""
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

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000.0, 2)


class ToolRuntime(ToolKit):
    """Backward-compatible name for ToolKit."""
