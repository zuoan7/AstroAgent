from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.logger import logger
from src.agent.models.skill_result import SkillResult
from src.skills.executor import SkillExecutor
from src.skills.mcp_client import MCPClient
from src.skills import registry
from src.skills.skill_handlers import SKILL_HANDLERS
from src.tools.runtime import ToolRuntime


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
        self._tool_runtime = ToolRuntime(self._mcp)

        self._handlers: Dict[str, Any] = {}
        for skill_name, handler_cls in SKILL_HANDLERS.items():
            self._handlers[skill_name] = handler_cls()

        self._executor = SkillExecutor(
            tool_runtime=self._tool_runtime,
            handlers=self._handlers,
            simple_tool_caller=lambda tool_name, **kwargs: self.call_mcp_tool(
                tool_name,
                **kwargs,
            ),
        )

        logger.info("✅ AstronomySkillRouter初始化完成（MCP延迟连接模式）")

    def list_skills(self) -> Dict[str, str]:
        return registry.list_skill_descriptions()

    def register_simple_skill(
        self,
        skill_name: str,
        tool_name: str,
        param_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self._executor.register_simple_skill(skill_name, tool_name, param_mapping)

    def call(self, name: str, **params: Any) -> SkillResult:
        return self._executor.call(name, **params)

    def call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        return self._tool_runtime.call_tool(tool_name, **kwargs)

    def call_mcp_tools_parallel(self, calls: list[dict]) -> list[str]:
        return self._tool_runtime.call_tools_parallel(calls)

    async def async_call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        return await self._tool_runtime.async_call_tool(tool_name, **kwargs)

    def prewarm(self) -> bool:
        return self._tool_runtime.prewarm()

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        return self._tool_runtime.get_runtime_metrics_snapshot()

    def shutdown(self) -> None:
        self._tool_runtime.shutdown()
        logger.info("✅ AstronomySkillRouter已关闭")
