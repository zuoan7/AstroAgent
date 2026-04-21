import traceback
from typing import List, Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.agent.fallback_service import FallbackService
from src.agent.request_router import RequestRouter
from src.agent.skill_manager import SkillManager
from src.agent.speech_service import SpeechService
from src.agent.streaming_service import StreamingService
from src.agent.task_orchestrator import TaskOrchestrator
from src.agent.vision_service import VisionService
from src.core.config import settings
from src.core.llm_factory import build_chat_model
from src.core.logger import logger
from src.core.model_catalog import model_selection_payload, resolve_model_config
from src.memory.api.memory_service import MemoryService
from src.memory.long_term_memory import LongTermMemoryManager
from src.rag.online_retriever import OnlineRetriever


class AstroAgent:
    def __init__(
        self,
        user_id: Optional[str] = None,
        *,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
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
        self.long_term_memory = LongTermMemoryManager(settings.LONG_TERM_MEMORY_PATH)

        self.skill_manager = SkillManager(rag_retriever=self.rag)
        self.request_router = RequestRouter()
        self.fallback_service = FallbackService(skill_manager=self.skill_manager)
        self.vision_service = VisionService()
        self.speech_service = SpeechService()
        runtime = self.create_session_runtime(
            user_id=self.user_id,
            memory=self.memory,
            model_provider=self.model_provider,
            model_name=self.model_name,
        )
        self.llm = runtime["llm"]
        self.task_orchestrator = runtime["task_orchestrator"]
        self.streaming_service = runtime["streaming_service"]
        self._agent_executor = runtime["agent_executor"]
        self.model_provider = runtime["model_provider"]
        self.model_name = runtime["model_name"]
        self.model_label = runtime["model_label"]

        logger.info("✅ AstroAgent初始化完成，使用统一的SkillManager")

    def _init_llm(
        self, model_provider: Optional[str] = None, model_name: Optional[str] = None
    ):
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
        try:
            template = self._load_prompt_template()
        except Exception as e:
            logger.error(f"❌ 读取prompt模板文件失败：{str(e)}")
            template = """
                    你是一个专业又亲切的天文助手，帮助用户解答天文问题。

                    **用户画像与偏好**：
                    {user_profile}

                    **对话历史**：
                    {chat_history}

                    **系统默认值**：
                    - 默认观测位置：北京（纬度 39.9°N，经度 116.4°E）
                    - 默认日期：当前日期
                    - 天象数据支持年份范围：2026–2030

                    **可用工具列表**：
                    {tools}

                    **问题类型与技能路由**：
                    - 科普知识类 → RAGRetrieve
                    - 实时数据类 → CelestialPositionCalculator / ObservationPlanner
                    - 观测条件/天气类 → ObservationPlanner
                    - 特殊天象类 → CelestialEventsForecast
                    - 深空观测类 → DeepSkyObservingGuide
                    - 近地天体类 → NEOTracker
                    - 天文摄影类 → AstrophotographyCalculator
                    - 闲聊类 → 直接礼貌回应，不调用工具

                    **回答风格**：
                    - 观测推荐：热情推荐，先说最佳目标，再给建议
                    - 科普知识：从简单概念讲起，用比喻帮助理解
                    - 实时数据：直接给出数据，简洁明了
                    - 专业术语附注英文原文（如"视星等 apparent magnitude"）

                    使用以下格式：

                    Question: {input}
                    Thought: 先判断问题类型，再选择合适的技能/工具
                    Action: 选择一个工具，应该是[{tool_names}]之一
                    Action Input: 工具参数，必须是有效的JSON格式（例如：{{"target": "mars"}}）
                    Observation: 工具返回结果
                    ... (Thought/Action/Action Input/Observation最多重复5次)
                    Thought: 我现在知道最终答案了
                    Final Answer: 最终答案

                    开始！

                    Question: {input}
                    Thought: {agent_scratchpad}
                    """
            logger.info("⚠️  使用默认prompt模板")

        prompt = PromptTemplate.from_template(template)

        tools = self.skill_manager.get_langchain_tools()

        agent = create_react_agent(llm=llm or self.llm, tools=tools, prompt=prompt)

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

    def _load_prompt_template(self) -> str:
        from src.core.config import resolve_path

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
        selection = model_selection_payload(model_provider, model_name)
        llm = self._init_llm(selection["model_provider"], selection["model_name"])
        task_orchestrator = TaskOrchestrator(
            skill_manager=self.skill_manager,
            rag_retriever=self.rag,
            llm=llm,
        )
        streaming_service = StreamingService(
            agent_executor=None,
            memory=memory,
            long_term_memory=self.long_term_memory,
            user_id=user_id,
            fallback_service=self.fallback_service,
            request_router=self.request_router,
            task_orchestrator=task_orchestrator,
            skill_manager=self.skill_manager,
            rag_retriever=self.rag,
        )
        agent_executor = self._build_agent(llm=llm)
        streaming_service._agent_executor = agent_executor
        return {
            "llm": llm,
            "task_orchestrator": task_orchestrator,
            "streaming_service": streaming_service,
            "agent_executor": agent_executor,
            **selection,
        }

    def generate_response(self, query: str):
        return self.streaming_service.generate_response(query)

    def generate_response_stream(self, query: str):
        return self.streaming_service.generate_response_stream(query)

    async def generate_events(self, query: str, image_path: Optional[str] = None):
        if image_path:
            query = self.vision_service.build_vision_query(query, image_path)

        async for event in self.streaming_service.generate_events(query):
            yield event

    def describe_image(self, image_path: str, prompt: str) -> str:
        return self.vision_service.describe_image(image_path, prompt)

    def add_astronomy_knowledge(self, knowledge: List[str]):
        if not knowledge:
            logger.warning("⚠️  无有效知识可添加")
            return

        try:
            from langchain.schema import Document

            documents = [Document(page_content=k) for k in knowledge]
            self.rag.add_documents(documents)
            logger.info(f"✅ 成功添加 {len(knowledge)} 条知识到RAG系统")
        except Exception as e:
            logger.error(f"❌ 添加知识失败：{str(e)}")
            traceback.print_exc()

    def clear_memory(self):
        try:
            self.memory.clear()
            logger.info("✅ 记忆已清空")
        except Exception as e:
            logger.error(f"❌ 清空记忆失败：{str(e)}")
            traceback.print_exc()

    def __del__(self):
        try:
            if hasattr(self, "skill_manager") and self.skill_manager:
                router = getattr(self.skill_manager, "_skill_router", None)
                if router and hasattr(router, "shutdown"):
                    router.shutdown()
                    logger.info("✅ MCP Router已关闭")
        except Exception:
            pass
