"""统一技能管理器，注册 LangChain 工具并把高层技能、RAG 和原子 MCP 工具转发到实际路由器。
"""

from typing import Any, Callable, Dict, List, Optional
from langchain_core.tools import Tool
from src.core.logger import logger
from src.skills.router import AstronomySkillRouter
from src.skills import registry
from src.skills.registry import SkillSpec
from src.agent.param_parser import ParamParser
from src.agent.models.skill_result import SkillResult
from src.tools.catalog import AtomicToolSpec, get_default_tool_catalog
from src.tools.selector import AtomicToolParamAdapter


_REACT_ATOMIC_TOOL_NAMES = {"get_nasa_apod", "get_weather", "web_search"}


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
        """初始化 SkillManager 的依赖、配置和内部状态。"""
        self._rag = rag_retriever
        self._skill_router = AstronomySkillRouter()
        self._tools: Optional[List[Tool]] = None
        logger.info("✅ SkillManager初始化完成（统一管理模式）")

    # ===== 公共接口 =====

    def get_langchain_tools(self) -> List[Tool]:
        """返回注册给 ReAct Agent 使用的 LangChain Tool 列表。"""
        if self._tools is None:
            self._tools = self._init_tools()
        return self._tools

    def list_skills(self) -> Dict[str, str]:
        """列出当前可用高层技能。"""
        return self._skill_router.list_skills()

    def call_skill(self, name: str, **params: Any) -> SkillResult:
        """调用指定高层技能并返回 SkillResult。"""
        return self._skill_router.call(name, **params)

    def call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        """调用指定原子 MCP 工具并返回原始响应。"""
        return self._skill_router.call_mcp_tool(tool_name, **kwargs)

    def prewarm(self) -> bool:
        """预热底层技能路由器和 MCP 会话。"""
        return self._skill_router.prewarm()

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        """返回技能路由和 MCP 调用的运行时指标快照。"""
        return self._skill_router.get_runtime_metrics_snapshot()

    # ===== 工具注册 =====

    def _init_tools(self) -> List[Tool]:
        """初始化 RAG、高层技能和允许的原子工具 Tool 列表。"""
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

        for spec in get_default_tool_catalog().list_specs():
            if spec.name not in _REACT_ATOMIC_TOOL_NAMES:
                continue
            tools.append(
                Tool(
                    name=spec.name,
                    func=self._create_atomic_tool_func(spec),
                    description=self._atomic_tool_description(spec),
                )
            )

        logger.info(f"✅ 成功注册 {len(tools)} 个技能/原子工具（含RAG）")
        return tools

    # ===== 通用工厂方法 =====

    def _create_skill_func(self, spec: SkillSpec) -> Callable:
        """为高层技能创建 LangChain Tool 兼容函数。"""
        def skill_func(tool_input: Any) -> str:
            """解析 LangChain 输入并调用对应高层技能。"""
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
        """创建 RAG 检索 Tool 兼容函数。"""
        def rag_func(query: Any) -> str:
            """解析查询文本并调用本地 RAG 检索。"""
            params = ParamParser.parse_tool_input(query, primary_param="query")
            query_text = params.get("query", str(query))
            return self._rag.get_relevant_context(query_text)
        return rag_func

    def _create_atomic_tool_func(self, spec: AtomicToolSpec) -> Callable:
        """为原子 MCP 工具创建 LangChain Tool 兼容函数。"""
        def atomic_tool_func(tool_input: Any) -> str:
            """解析 LangChain 输入并调用对应原子 MCP 工具。"""
            params = self._parse_atomic_tool_input(spec, tool_input)
            result = self._skill_router.call(spec.name, **params)
            return result.to_legacy_str()

        return atomic_tool_func

    # ===== 特殊处理函数 =====

    @staticmethod
    def _weather_param_handler(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """规范化天气技能参数。"""
        return registry.normalize_weather_params(kwargs)

    @staticmethod
    def _atomic_tool_description(spec: AtomicToolSpec) -> str:
        """生成原子工具的 LangChain 描述文本。"""
        params = ", ".join(spec.param_names) if spec.param_names else "无参数"
        return f"{spec.summary or spec.name}（atomic MCP tool: {spec.name}）。参数：{params}。"

    @staticmethod
    def _parse_atomic_tool_input(spec: AtomicToolSpec, tool_input: Any) -> Dict[str, Any]:
        """解析原子工具输入并应用默认参数。"""
        if not isinstance(tool_input, dict):
            return AtomicToolParamAdapter.build(spec.name, str(tool_input or ""))

        defaults = {
            "get_nasa_apod": {"date": None, "hd": False},
            "get_weather": {"city": None, "extensions": "all"},
            "web_search": {"query": None, "max_results": 5},
        }.get(spec.name, {name: None for name in spec.param_names})
        params = ParamParser.parse_tool_input(tool_input, expected_params=defaults)
        if spec.name == "web_search" and params.get("max_results") is not None:
            params["max_results"] = SkillManager._safe_convert(
                params["max_results"],
                int,
            )
        if spec.name == "get_nasa_apod" and params.get("hd") is not None:
            params["hd"] = SkillManager._safe_convert(params["hd"], bool)
        return {key: value for key, value in params.items() if value is not None}

    @staticmethod
    def _safe_convert(value: Any, convert_func: type) -> Any:
        """安全执行类型转换，失败时保留原值。"""
        try:
            if isinstance(value, str):
                if convert_func == bool:
                    return value.lower() in ("true", "1", "yes")
                elif convert_func == float:
                    return float(value)
            return convert_func(value)
        except (ValueError, TypeError):
            return value
