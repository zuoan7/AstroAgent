"""高层技能路由器，连接 SkillExecutor、ToolRuntime、MCPClient 和复杂技能 handler。
"""

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
    轻量技能路由器，将技能调用委托给 MCPClient 和专用 handler。

    职责：
    - 校验技能注册表并分发调用
    - 管理 simple skill 到 MCP 工具的透传
    - 将复杂技能逻辑交给专用 handler
    - 为所有技能调用返回 SkillResult
    """

    def __init__(self) -> None:
        """初始化 MCP 客户端、工具运行时、技能 handler 和执行器。"""
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
        """返回技能名到技能摘要的映射。"""
        return registry.list_skill_descriptions()

    def register_simple_skill(
        self,
        skill_name: str,
        tool_name: str,
        param_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """注册一个直接透传到底层 MCP 工具的 simple skill。"""
        self._executor.register_simple_skill(skill_name, tool_name, param_mapping)

    def call(self, name: str, **params: Any) -> SkillResult:
        """调用高层技能并返回结构化 SkillResult。"""
        return self._executor.call(name, **params)

    def call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        """同步调用单个原子 MCP 工具。"""
        return self._tool_runtime.call_tool(tool_name, **kwargs)

    def call_mcp_tools_parallel(self, calls: list[dict]) -> list[str]:
        """并行调用多个原子 MCP 工具。"""
        return self._tool_runtime.call_tools_parallel(calls)

    async def async_call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        """异步调用单个原子 MCP 工具。"""
        return await self._tool_runtime.async_call_tool(tool_name, **kwargs)

    def prewarm(self) -> bool:
        """预热 MCP 连接和底层运行时。"""
        return self._tool_runtime.prewarm()

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        """返回 MCP 和工具运行时指标快照。"""
        return self._tool_runtime.get_runtime_metrics_snapshot()

    def shutdown(self) -> None:
        """关闭工具运行时并释放 MCP 连接。"""
        self._tool_runtime.shutdown()
        logger.info("✅ AstronomySkillRouter已关闭")
