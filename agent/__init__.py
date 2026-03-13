from langchain_community.chat_models import ChatTongyi
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from config import settings
from rag.online_retriever import OnlineRetriever
from memory import ShortTermMemory
from skills import AstronomySkillRouter
from agent.tools import AgentTools
from agent.fallback_service import FallbackService
from agent.vision_service import VisionService
from agent.streaming_service import StreamingService
from typing import List, Optional
import traceback
from logger import logger


class AstroAgent:
    def __init__(self):
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("❌ DashScope API Key未配置！请在settings中设置DASHSCOPE_API_KEY")

        self.rag = OnlineRetriever()
        self.memory = ShortTermMemory()
        self.llm = self._init_llm()
        self.skill_router = AstronomySkillRouter()

        self.tools_manager = AgentTools(
            rag_retriever=self.rag,
            skill_router=self.skill_router,
        )

        self.fallback_service = FallbackService(skill_router=self.skill_router)
        self.vision_service = VisionService()

        self.streaming_service = StreamingService(
            agent_executor=None,
            memory=self.memory,
            fallback_service=self.fallback_service,
        )

        self._agent_executor = self._build_agent()
        self.streaming_service._agent_executor = self._agent_executor

        logger.info("✅ AstroAgent初始化完成，通过Skill层调用MCP工具")

    def _init_llm(self):
        try:
            llm = ChatTongyi(
                model=settings.MODEL_NAME,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
                temperature=0.1
            )
            logger.info(f"✅ 语言模型 {settings.MODEL_NAME} 初始化成功")
            return llm
        except Exception as e:
            logger.error(f"❌ 语言模型初始化失败：{str(e)}")
            raise

    def _build_agent(self):
        try:
            with open('prompt_template.txt', 'r', encoding='utf-8') as f:
                template = f.read()
            logger.info("✅ 成功从外部文件读取prompt模板")
        except Exception as e:
            logger.error(f"❌ 读取prompt模板文件失败：{str(e)}")
            template = '''
                    你是一个专业的天文助手，帮助用户解答天文问题。
                                
                    **可用工具列表**：
                    {tools}

                    **对话历史**：
                    {chat_history}

                    使用以下格式：

                    Question: {input}
                    Thought: 我需要思考如何回答这个问题
                    Action: 选择一个工具
                    Action Input: 工具参数
                    Observation: 工具返回结果
                    Thought: 现在我知道答案了
                    Final Answer: 最终答案

                    开始！

                    Question: {input}
                    Thought: {agent_scratchpad}
                    '''
            logger.info("⚠️  使用默认prompt模板")

        prompt = PromptTemplate.from_template(template)

        tools = self.tools_manager.get_tools()

        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
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
        if hasattr(self, 'http_client') and self.http_client:
            try:
                self.http_client.close()
                logger.info("✅ HTTP客户端已关闭")
            except:
                pass
