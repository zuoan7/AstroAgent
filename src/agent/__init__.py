"""AstroAgent 应用门面，统一初始化 LLM、RAG、记忆、技能、执行引擎和流式服务，并为 API 会话创建运行时。"""

import traceback
from typing import List, Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.agent.adapters.langchain_adapter import to_langchain_tools
from src.agent.audit import RequestAuditLogger
from src.agent.capability_kit import CapabilityKit
from src.agent.execution.engine import ExecutionEngine
from src.agent.fallback_service import FallbackService
from src.agent.governance import (
    AgentExecutionPolicy,
    GovernanceMetricsRegistry,
    evaluate_router_benchmark,
    load_phase0_benchmark_cases,
)
from src.agent.llm_intent_classifier import LLMIntentClassifier
from src.agent.output_parser import LenientReActSingleInputOutputParser
from src.agent.planner import Planner
from src.agent.policies import FallbackPolicy, ModelPolicy
from src.agent.prompts import PromptRenderError, get_prompt_renderer
from src.agent.request_router import RequestRouter
from src.agent.response_synthesizer import ResponseSynthesizer
from src.agent.speech_service import SpeechService
from src.agent.streaming_service import StreamingService
from src.agent.vision_service import VisionService
from src.core.config import settings
from src.core.llm_factory import build_chat_model
from src.core.logger import logger
from src.core.model_catalog import model_selection_payload, resolve_model_config
from src.memory.api.memory_service import MemoryService
from src.rag.online_retriever import OnlineRetriever


class AstroAgent:
    """AstroAgent 总入口，聚合模型、RAG、记忆、技能和执行引擎能力。"""

    def __init__(
        self,
        user_id: Optional[str] = None,
        *,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """初始化 AstroAgent 的依赖、配置和内部状态。"""
        self.user_id = user_id or settings.DEFAULT_USER_ID
        self.model_provider = model_provider or getattr(
            settings, "DEFAULT_LLM_PROVIDER", "dashscope"
        )
        self.model_name = model_name or settings.MODEL_NAME

        resolve_model_config(self.model_provider, self.model_name)

        self.rag = OnlineRetriever()
        self.memory = MemoryService(
            db_path=settings.MEMORY_PERSISTENCE_PATH,
            session_id=f"mem_{self.user_id}",
            user_id=self.user_id,
        )
        from src.memory.long_term_memory.service import LongTermMemoryService

        self.long_term_memory = LongTermMemoryService(settings.LONG_TERM_MEMORY_PATH)

        self.capability_kit = CapabilityKit(rag_retriever=self.rag)
        self.request_router = RequestRouter()
        self.fallback_service = FallbackService(capability_kit=self.capability_kit)
        self.execution_policy = AgentExecutionPolicy.from_settings()
        self.governance_metrics = GovernanceMetricsRegistry()
        self.model_policy = ModelPolicy()
        self.audit_logger = RequestAuditLogger()
        self.vision_service = VisionService()
        self.speech_service = SpeechService()
        runtime = self.create_session_runtime(
            user_id=self.user_id,
            memory=self.memory,
            model_provider=self.model_provider,
            model_name=self.model_name,
        )
        self.llm = runtime["llm"]
        self.execution_engine = runtime["execution_engine"]
        self.task_orchestrator = runtime.get("task_orchestrator")
        self.streaming_service = runtime["streaming_service"]
        self._agent_executor = runtime["agent_executor"]
        self.model_provider = runtime["model_provider"]
        self.model_name = runtime["model_name"]
        self.model_label = runtime["model_label"]

        logger.info("✅ AstroAgent初始化完成，使用 CapabilityKit")

    def _init_llm(
        self, model_provider: Optional[str] = None, model_name: Optional[str] = None
    ):
        """初始化指定供应商和模型名称对应的聊天模型。"""
        try:
            resolved = resolve_model_config(model_provider, model_name)
            llm = build_chat_model(
                provider=resolved.provider,
                model=resolved.model_name,
                temperature=0.1,
            )
            logger.info(
                f"✅ 语言模型 {resolved.model_name} 初始化成功"
                f"（provider={resolved.provider}, base_url={resolved.base_url}）"
            )
            return llm
        except Exception as e:
            logger.error(f"❌ 语言模型初始化失败：{str(e)}")
            raise

    def _build_agent(self, llm=None):
        """构建 LangChain ReAct AgentExecutor 作为开放式 fallback 执行器。"""
        try:
            template = self._load_prompt_template()
        except Exception as e:
            logger.error(f"❌ 读取prompt模板文件失败：{str(e)}")
            raise

        prompt = PromptTemplate.from_template(template)

        tools = to_langchain_tools(self.capability_kit)

        agent = create_react_agent(
            llm=llm or self.llm,
            tools=tools,
            prompt=prompt,
            output_parser=LenientReActSingleInputOutputParser(),
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=60,
            early_stopping_method="force",
            return_intermediate_steps=True,
        )

        logger.info("✅ React Agent构建完成（最大迭代5次，超时60秒）")
        return agent_executor

    def _get_or_create_agent_executor(self, llm=None):
        """懒加载并复用当前会话的 ReAct AgentExecutor。"""
        if self._agent_executor is None:
            self._agent_executor = self._build_agent(llm=llm or self.llm)
        return self._agent_executor

    def _load_prompt_template(self) -> str:
        """从 PromptRegistry 或 legacy 文件加载 ReAct 主提示词模板。"""
        from src.core.config import resolve_path

        try:
            return get_prompt_renderer().render("react.main")
        except PromptRenderError as exc:
            logger.warning(f"⚠️  Prompt registry 读取失败，回退到 legacy 模板: {exc}")

        template_path = resolve_path(settings.PROMPT_TEMPLATE_PATH)
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        logger.info(f"✅ 成功从外部文件读取prompt模板: {template_path}")
        return template

    def create_session_runtime(
        self,
        *,
        user_id: str,
        memory: MemoryService,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """为指定用户会话创建 LLM、执行引擎和流式服务运行时。"""
        selection = model_selection_payload(model_provider, model_name)
        main_llm = self._init_llm(selection["model_provider"], selection["model_name"])
        synth_selection = self.model_policy.select("synthesizer")
        planner_selection = self.model_policy.select("planner")
        synth_llm = self._init_llm(
            synth_selection.provider,
            synth_selection.model_name,
        )
        planner_llm = self._init_llm(
            planner_selection.provider,
            planner_selection.model_name,
        )
        if getattr(settings, "ENABLE_LLM_INTENT_FALLBACK", False):
            self.request_router.configure_llm_fallback(
                LLMIntentClassifier(
                    planner_llm,
                    min_accept_confidence=float(
                        getattr(settings, "LLM_INTENT_MIN_ACCEPT_CONFIDENCE", 0.55)
                    ),
                ),
                enabled=True,
                confidence_threshold=float(
                    getattr(settings, "LLM_INTENT_CONFIDENCE_THRESHOLD", 0.8)
                ),
            )
        response_synthesizer = ResponseSynthesizer(llm=synth_llm)
        planner = Planner(llm=planner_llm)
        fallback_policy = FallbackPolicy()
        agent_executor_factory = lambda: self._get_or_create_agent_executor(
            llm=main_llm
        )
        execution_engine = ExecutionEngine(
            capability_kit=self.capability_kit,
            rag_retriever=self.rag,
            llm=main_llm,
            synthesizer=response_synthesizer,
            planner=planner,
            fallback_policy=fallback_policy,
            agent_executor_factory=agent_executor_factory,
        )
        streaming_service = StreamingService(
            agent_executor=None,
            memory=memory,
            long_term_memory=self.long_term_memory,
            user_id=user_id,
            fallback_service=self.fallback_service,
            request_router=self.request_router,
            capability_kit=self.capability_kit,
            rag_retriever=self.rag,
            execution_policy=self.execution_policy,
            governance_metrics=self.governance_metrics,
            audit_logger=self.audit_logger,
            agent_executor_factory=agent_executor_factory,
            execution_engine=execution_engine,
        )
        return {
            "llm": main_llm,
            "task_orchestrator": None,
            "execution_engine": execution_engine,
            "streaming_service": streaming_service,
            "agent_executor": None,
            **selection,
        }

    def generate_response(self, query: str):
        """同步生成完整文本响应。"""
        return self.streaming_service.generate_response(query)

    def generate_response_stream(self, query: str):
        """异步生成纯文本答案流。"""
        return self.streaming_service.generate_response_stream(query)

    async def generate_events(self, query: str, image_path: Optional[str] = None):
        """异步生成前端 JSON 事件流。"""
        if image_path:
            query = self.vision_service.build_vision_query(query, image_path)

        async for event in self.streaming_service.generate_events(query):
            yield event

    def describe_image(self, image_path: str, prompt: str) -> str:
        """调用视觉服务生成图片描述。"""
        return self.vision_service.describe_image(image_path, prompt)

    def add_astronomy_knowledge(self, knowledge: List[str]):
        """把人工补充的天文知识写入 RAG 检索库。"""
        if not knowledge:
            logger.warning("⚠️  无有效知识可添加")
            return {
                "added_count": 0,
                "updated_count": 0,
                "unchanged_count": 0,
                "stored_count": 0,
                "bm25_doc_count": 0,
            }

        try:
            from langchain.schema import Document

            documents = [Document(page_content=k) for k in knowledge]
            result = self.rag.add_documents(documents, source="manual")
            logger.info(f"✅ 成功添加 {len(knowledge)} 条知识到RAG系统")
            return result
        except Exception as e:
            logger.error(f"❌ 添加知识失败：{str(e)}")
            traceback.print_exc()
            return {
                "added_count": 0,
                "updated_count": 0,
                "unchanged_count": 0,
                "stored_count": 0,
                "bm25_doc_count": 0,
            }

    def clear_memory(self):
        """清空当前 Agent 会话的短期记忆。"""
        try:
            self.memory.clear()
            logger.info("✅ 记忆已清空")
        except Exception as e:
            logger.error(f"❌ 清空记忆失败：{str(e)}")
            traceback.print_exc()

    def get_governance_metrics_snapshot(self):
        """读取治理指标快照。"""
        return self.governance_metrics.snapshot()

    def evaluate_phase0_router_benchmark(self, path: Optional[str] = None):
        """运行 Phase0 路由基准评测并返回统计结果。"""
        cases = load_phase0_benchmark_cases(path)
        return evaluate_router_benchmark(self.request_router, cases)

    def __del__(self):
        """对象销毁时释放内部持有的外部连接或后台资源。"""
        try:
            if hasattr(self, "capability_kit") and self.capability_kit:
                self.capability_kit.shutdown()
                logger.info("✅ CapabilityKit已关闭")
        except Exception:
            pass
