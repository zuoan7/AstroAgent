from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.core.logger import logger
from src.agent.param_parser import ParamParser
from src.skills.mcp_client import MCPClient
from src.skills.skill_handlers import SKILL_HANDLERS


class AstronomySkillRouter:
    """
    Slim skill router that delegates to MCPClient and skill handlers.

    Responsibilities:
    - Maintain skill registry and dispatch calls
    - Manage simple skill passthrough to MCP tools
    - Delegate complex skill logic to dedicated handler classes
    """

    def __init__(self) -> None:
        self._mcp = MCPClient()

        self._handlers: Dict[str, Any] = {}
        for skill_name, handler_cls in SKILL_HANDLERS.items():
            self._handlers[skill_name] = handler_cls()

        self._simple_skills: Dict[str, Dict[str, Any]] = {}

        self.register_simple_skill(
            skill_name="weather-lookup",
            tool_name="get_weather",
            param_mapping={
                "city": "city",
                "location": "city",
                "extensions": "extensions",
            },
        )

        logger.info("✅ AstronomySkillRouter初始化完成（MCP延迟连接模式）")

    def list_skills(self) -> Dict[str, str]:
        return {
            "observation-planner": "生成指定日期和地点的天文观测计划",
            "celestial-events-forecast": "查询指定时间段的天象事件",
            "deep-sky-observing-guide": "为指定深空天体提供观测指导",
            "neo-tracker": "追踪近地天体飞掠事件",
            "astrophotography-calculator": "计算天文摄影参数与建议",
            "celestial-position-calculator": "计算天体在指定时间的位置",
            "weather-lookup": "查询指定城市的观测相关天气信息",
        }

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

    def call(self, name: str, **params: Any) -> str:
        if name in self._handlers:
            return self._handlers[name](self._mcp, **params)

        if name in self._simple_skills:
            cfg = self._simple_skills[name]
            tool_name = cfg["tool_name"]
            mapping: Dict[str, str] = cfg.get("param_mapping", {})
            tool_kwargs: Dict[str, Any] = {}
            for k, v in params.items():
                tool_key = mapping.get(k, k)
                tool_kwargs[tool_key] = v
            raw = self.call_mcp_tool(tool_name, **tool_kwargs)
            return ParamParser.shorten_text(raw, 1200)

        raise ValueError(f"未知技能：{name}")

    def call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        return self._mcp.call_tool(tool_name, **kwargs)

    def call_mcp_tools_parallel(self, calls: list[dict]) -> list[str]:
        return self._mcp.call_tools_parallel(calls)

    async def async_call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        return await self._mcp.async_call_tool(tool_name, **kwargs)

    def shutdown(self) -> None:
        self._mcp.shutdown()
        logger.info("✅ AstronomySkillRouter已关闭")
