from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Optional

from src.agent.models.skill_result import SkillResult
from src.capabilities.registry import CapabilityRegistry, get_default_capability_registry
from src.core.mcp_protocol import is_tool_error, parse_tool_response
from src.skills import registry
from src.tools.runtime import ToolRuntime


class _CallableToolBackend:
    def __init__(self, call_tool: Callable[..., str]) -> None:
        self._call_tool = call_tool

    def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        return self._call_tool(tool_name, **kwargs)


class SkillExecutor:
    """Executes high-level skills while preserving the legacy call contract."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime,
        handlers: Dict[str, Any],
        capability_registry: Optional[CapabilityRegistry] = None,
        simple_tool_caller: Optional[Callable[..., str]] = None,
    ) -> None:
        self._tool_runtime = tool_runtime
        self._handlers = dict(handlers)
        self._capabilities = capability_registry or get_default_capability_registry()
        self._simple_tool_caller = simple_tool_caller
        self._simple_skills: Dict[str, Dict[str, Any]] = {}
        self._register_registry_simple_skills()

    def call(self, name: str, **params: Any) -> SkillResult:
        spec = registry.get_skill_spec(name)
        normalized = self._normalize_skill_params(spec.skill_name, params)

        if spec.skill_name in self._handlers:
            runtime = self._runtime_for_skill(spec.skill_name)
            result = self._handlers[spec.skill_name](runtime, **normalized)
            self._attach_default_metadata(result, spec.skill_name)
            return result

        if spec.skill_name in self._simple_skills:
            return self._call_simple_skill(spec.skill_name, **normalized)

        raise ValueError(f"未知技能：{name}")

    def register_simple_skill(
        self,
        skill_name: str,
        tool_name: str,
        param_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self._simple_skills[skill_name] = {
            "tool_name": tool_name,
            "param_mapping": param_mapping or {},
        }

    def _register_registry_simple_skills(self) -> None:
        for spec in registry.get_skill_specs():
            if spec.route_type != "simple" or not spec.mcp_tool_name:
                continue
            self.register_simple_skill(
                skill_name=spec.skill_name,
                tool_name=spec.mcp_tool_name,
                param_mapping=spec.param_mapping,
            )

    def _call_simple_skill(self, name: str, **params: Any) -> SkillResult:
        started = time.perf_counter()
        cfg = self._simple_skills[name]
        tool_name = cfg["tool_name"]
        mapping: Dict[str, str] = cfg.get("param_mapping", {})
        tool_kwargs: Dict[str, Any] = {}
        for key, value in params.items():
            tool_key = mapping.get(key, key)
            tool_kwargs[tool_key] = value

        raw = self._runtime_for_skill(name, simple=True).call_tool(
            tool_name,
            **tool_kwargs,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if is_tool_error(raw):
            envelope = parse_tool_response(raw)
            error_msg = ""
            if envelope and hasattr(envelope, "error"):
                error_msg = getattr(envelope.error, "message", str(raw)[:500])
            result = SkillResult.from_error(
                skill_name=name,
                error_code="TOOL_CALL_FAILED",
                error_message=error_msg or str(raw)[:500],
                latency_ms=round(elapsed_ms, 2),
            )
            result.logical_skill = name
            result.expected_mcp_tools = [tool_name]
            result.allowed_child_tools = [tool_name]
            result.sources = [
                {"kind": "mcp_tool", "tool": tool_name, "snippet": str(raw)[:240]}
            ]
            return result

        data: Dict[str, Any] = {}
        summary: Any = raw
        envelope = parse_tool_response(raw)
        if envelope is not None and hasattr(envelope, "data"):
            raw_data = envelope.data
            if isinstance(raw_data, dict):
                data = raw_data
            else:
                data = {"raw": raw_data}
            summary = str(raw_data) if not isinstance(raw_data, str) else raw_data
        else:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    data = parsed
            except (json.JSONDecodeError, TypeError):
                data = {"raw": raw}

        return SkillResult(
            skill_name=name,
            success=True,
            data=data,
            summary=(
                summary
                if isinstance(summary, str)
                else json.dumps(summary, ensure_ascii=False)
            ),
            sources=[
                {"kind": "mcp_tool", "tool": tool_name, "snippet": str(raw)[:240]}
            ],
            latency_ms=round(elapsed_ms, 2),
            logical_skill=name,
            expected_mcp_tools=[tool_name],
            allowed_child_tools=[tool_name],
        )

    def _runtime_for_skill(self, skill_name: str, *, simple: bool = False) -> ToolRuntime:
        spec = self._capabilities.get_skill(skill_name)
        runtime = self._tool_runtime
        if simple and self._simple_tool_caller is not None:
            runtime = ToolRuntime(
                _CallableToolBackend(self._simple_tool_caller),
                guard=self._tool_runtime.guard,
            )
        return runtime.with_context(
            logical_skill=skill_name,
            allowed_tools=list(spec.allowed_tools),
            forbidden_tools=list(spec.forbidden_tools),
            enforce_allowed_tools=True,
        )

    def _normalize_skill_params(
        self,
        name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        spec = registry.get_skill_spec(name)
        normalized: Dict[str, Any] = dict(spec.defaults or {})
        candidate = dict(params or {})

        if spec.special_handling:
            candidate = spec.special_handling(candidate)

        for key in spec.param_names:
            value = candidate.get(key)
            if value is not None:
                normalized[key] = value

        if spec.route_type == "simple" and not normalized and candidate:
            for key, value in candidate.items():
                if value is not None:
                    normalized[key] = value

        return normalized

    @staticmethod
    def _attach_default_metadata(result: SkillResult, skill_name: str) -> None:
        if result.logical_skill is None:
            result.logical_skill = skill_name
