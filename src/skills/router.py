from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from src.core.logger import logger
from src.core.mcp_protocol import extract_tool_data, is_tool_error, parse_tool_response
from src.agent.models.skill_result import SkillResult
from src.skills.mcp_client import MCPClient
from src.skills import registry
from src.skills.skill_handlers import SKILL_HANDLERS


class AstronomySkillRouter:
    """
    Slim skill router that delegates to MCPClient and skill handlers.

    Responsibilities:
    - Maintain skill registry and dispatch calls
    - Manage simple skill passthrough to MCP tools
    - Delegate complex skill logic to dedicated handler classes
    - Return SkillResult for all skill calls
    """

    def __init__(self) -> None:
        registry.validate_skill_registry(handler_names=set(SKILL_HANDLERS.keys()))
        self._mcp = MCPClient()

        self._handlers: Dict[str, Any] = {}
        for skill_name, handler_cls in SKILL_HANDLERS.items():
            self._handlers[skill_name] = handler_cls()

        self._simple_skills: Dict[str, Dict[str, Any]] = {}
        self._register_registry_simple_skills()

        logger.info("✅ AstronomySkillRouter初始化完成（MCP延迟连接模式）")

    def list_skills(self) -> Dict[str, str]:
        return registry.list_skill_descriptions()

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

    def call(self, name: str, **params: Any) -> SkillResult:
        params = self._normalize_skill_params(name, params)
        if name in self._handlers:
            return self._handlers[name](self._mcp, **params)

        if name in self._simple_skills:
            return self._call_simple_skill(name, **params)

        raise ValueError(f"未知技能：{name}")

    def _call_simple_skill(self, name: str, **params: Any) -> SkillResult:
        started = time.perf_counter()
        cfg = self._simple_skills[name]
        tool_name = cfg["tool_name"]
        mapping: Dict[str, str] = cfg.get("param_mapping", {})
        tool_kwargs: Dict[str, Any] = {}
        for k, v in params.items():
            tool_key = mapping.get(k, k)
            tool_kwargs[tool_key] = v

        raw = self.call_mcp_tool(tool_name, **tool_kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if is_tool_error(raw):
            envelope = parse_tool_response(raw)
            error_msg = ""
            if envelope and hasattr(envelope, "error"):
                error_msg = getattr(envelope.error, "message", str(raw)[:500])
            return SkillResult.from_error(
                skill_name=name,
                error_code="TOOL_CALL_FAILED",
                error_message=error_msg or str(raw)[:500],
                latency_ms=round(elapsed_ms, 2),
            )

        data: Dict[str, Any] = {}
        summary = raw
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
            summary=summary if isinstance(summary, str) else json.dumps(summary, ensure_ascii=False),
            sources=[{"kind": "mcp_tool", "tool": tool_name, "snippet": str(raw)[:240]}],
            latency_ms=round(elapsed_ms, 2),
        )

    def _normalize_skill_params(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
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

    def call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        return self._mcp.call_tool(tool_name, **kwargs)

    def call_mcp_tools_parallel(self, calls: list[dict]) -> list[str]:
        return self._mcp.call_tools_parallel(calls)

    async def async_call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        return await self._mcp.async_call_tool(tool_name, **kwargs)

    def prewarm(self) -> bool:
        return self._mcp.prewarm()

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        return self._mcp.get_runtime_metrics_snapshot()

    def shutdown(self) -> None:
        self._mcp.shutdown()
        logger.info("✅ AstronomySkillRouter已关闭")
