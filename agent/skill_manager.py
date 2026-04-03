"""
SkillManager - 统一的技能管理器（适配层）
整合了原有的 AgentTools 和 AstronomySkillRouter，提供统一的接口
"""

from typing import Any, Dict, List
from langchain_core.tools import Tool
from logger import logger

from skills import AstronomySkillRouter
from agent.tools import AgentTools
from agent.param_parser import ParamParser


class SkillManager:
    """
    统一的技能管理器，作为适配层整合原有的 AgentTools 和 AstronomySkillRouter。
    
    这样做的好处：
    1. 保持向后兼容，不破坏现有代码
    2. 提供统一的接口，简化上层调用
    3. 便于未来进一步重构
    """

    def __init__(self, rag_retriever: Any = None) -> None:
        self._rag = rag_retriever
        self._skill_router = AstronomySkillRouter()
        self._tools_manager = AgentTools(
            rag_retriever=rag_retriever,
            skill_router=self._skill_router,
        )
        logger.info("✅ SkillManager初始化完成（适配层模式）")

    def get_langchain_tools(self) -> List[Tool]:
        """获取 LangChain 工具列表"""
        return self._tools_manager.get_tools()

    def list_skills(self) -> Dict[str, str]:
        """返回可用技能名称及简要说明"""
        return self._skill_router.list_skills()

    def call_skill(self, name: str, **params: Any) -> str:
        """调用指定技能"""
        return self._skill_router.call(name, **params)

    def call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        """直接调用底层 MCP 工具"""
        return self._skill_router.call_mcp_tool(tool_name, **kwargs)
