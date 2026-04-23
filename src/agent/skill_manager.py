"""
SkillManager - 统一的技能管理器（重构版）
整合了原有的 AgentTools 和 AstronomySkillRouter，消除三层架构的冗余调用链。
"""

from typing import Any, Callable, Dict, List, Optional
from langchain_core.tools import Tool
from src.core.logger import logger
from src.skills.router import AstronomySkillRouter
from src.skills import registry
from src.skills.registry import SkillSpec
from src.agent.param_parser import ParamParser
from src.agent.models.skill_result import SkillResult


class SkillManager:
    """
    统一的技能管理器 - 重构后的单层入口。

    职责：
    1. 工具注册：创建 LangChain Tool 对象（原 AgentTools 的职责）
    2. 技能路由：分发技能请求到具体实现（原 AstronomySkillRouter 的接口）
    3. MCP通信：管理与底层 MCP 服务器的通信（委托给内部 router）

    Stage 2 改造：
    - call_skill() 返回 SkillResult 结构体
    - LangChain Tool 接口通过 to_legacy_str() 保持兼容
    """

    def __init__(self, rag_retriever: Optional[Any] = None) -> None:
        self._rag = rag_retriever
        self._skill_router = AstronomySkillRouter()
        self._tools: Optional[List[Tool]] = None
        logger.info("✅ SkillManager初始化完成（统一管理模式）")

    # ===== 公共接口 =====

    def get_langchain_tools(self) -> List[Tool]:
        if self._tools is None:
            self._tools = self._init_tools()
        return self._tools

    def list_skills(self) -> Dict[str, str]:
        return self._skill_router.list_skills()

    def call_skill(self, name: str, **params: Any) -> SkillResult:
        return self._skill_router.call(name, **params)

    def call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        return self._skill_router.call_mcp_tool(tool_name, **kwargs)

    def prewarm(self) -> bool:
        return self._skill_router.prewarm()

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        return self._skill_router.get_runtime_metrics_snapshot()

    # ===== 工具注册 =====

    def _init_tools(self) -> List[Tool]:
        registry.validate_skill_registry()
        tools = [
            Tool(
                name="RAGRetrieve",
                func=self._create_rag_func(),
                description="使用本地RAG知识库检索天文知识、概念解释、历史资料等。参数：query（查询语句，中文即可）。",
            )
        ]

        for spec in registry.get_skill_specs():
            tools.append(Tool(
                name=spec.langchain_tool_name,
                func=self._create_skill_func(spec),
                description=spec.description,
            ))

        logger.info(f"✅ 成功注册 {len(tools)} 个高层技能工具（含RAG）")
        return tools

    # ===== 通用工厂方法 =====

    def _create_skill_func(self, spec: SkillSpec) -> Callable:
        def skill_func(tool_input: Any) -> str:
            parsed_input = ParamParser.parse(tool_input)
            kwargs = parsed_input if isinstance(parsed_input, dict) else {}

            if spec.special_handling:
                kwargs = spec.special_handling(kwargs)

            expected_params = {name: spec.defaults.get(name, None) for name in spec.param_names} if spec.defaults else {}
            params = ParamParser.parse_tool_input(
                kwargs if isinstance(kwargs, dict) else {},
                expected_params=expected_params if expected_params else {name: None for name in spec.param_names},
            )

            if spec.type_conversions:
                for param_name, convert_func in spec.type_conversions.items():
                    value = params.get(param_name)
                    if value is not None:
                        params[param_name] = self._safe_convert(value, convert_func)

            result = self._skill_router.call(spec.skill_name, **params)
            return result.to_legacy_str()

        return skill_func

    def _create_rag_func(self) -> Callable:
        def rag_func(query: Any) -> str:
            params = ParamParser.parse_tool_input(query, primary_param="query")
            query_text = params.get("query", str(query))
            return self._rag.get_relevant_context(query_text)
        return rag_func

    # ===== 特殊处理函数 =====

    @staticmethod
    def _weather_param_handler(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return registry.normalize_weather_params(kwargs)

    @staticmethod
    def _safe_convert(value: Any, convert_func: type) -> Any:
        try:
            if isinstance(value, str):
                if convert_func == bool:
                    return value.lower() in ("true", "1", "yes")
                elif convert_func == float:
                    return float(value)
            return convert_func(value)
        except (ValueError, TypeError):
            return value
