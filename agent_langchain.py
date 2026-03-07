from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import settings
from rag_langchain import RAGSystem
from memory import ShortTermMemory
from typing import Generator, List, Dict, Any
import time
import traceback
from logger import logger


class AstroAgent:
    """基于LangChain的天文Agent"""
    
    def __init__(self):
        # 验证API Key
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("❌ DashScope API Key未配置！请在settings中设置DASHSCOPE_API_KEY")
        
        # 初始化组件
        self.rag = RAGSystem()
        self.memory = ShortTermMemory()
        self.llm = self._init_llm()
        self.chain = self._build_chain()
        
        logger.info("✅ AstroAgent初始化完成，基于LangChain框架")
    
    def _init_llm(self):
        """初始化语言模型"""
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
    
    def _build_chain(self):
        """构建LangChain链"""
        # 系统提示
        system_prompt = """
        你是一个专业的天文知识助手，能够回答关于天文学的各种问题。
        请严格基于提供的上下文信息回答问题，确保专业、准确、详细。
        如果上下文没有相关信息，使用你的自有知识回答；如果不确定，如实告知。
        回答语言请使用中文，避免使用过于晦涩的术语，必要时给出解释。
        """
        
        # 提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "### 上下文信息：\n{context}\n\n### 对话历史：\n{history}\n\n### 用户问题：\n{query}")
        ])
        
        # 构建链
        chain = (
            {
                "context": lambda x: self.rag.get_relevant_context(x),
                "history": lambda x: self._get_history_text(),
                "query": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _get_history_text(self):
        """获取对话历史文本"""
        history = self.memory.get_recent_messages()
        history_text = ""
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            history_text += f"{role}：{msg['content']}\n"
        return history_text
    
    def generate_response(self, query: str) -> Generator[str, None, None]:
        """生成流式响应"""
        logger.info(f"\n=== 处理用户查询：{query} ===")
        
        try:
            # 调用LangChain链获取响应
            response = self.chain.invoke(query)
            
            # 模拟流式输出
            for i in range(0, len(response), 50):
                chunk = response[i:i+50]
                yield chunk
                time.sleep(0.1)  # 模拟流式效果
            
            # 保存对话到记忆
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", response, time.time())
            logger.info(f"✅ 对话已存入记忆 | 助手响应长度：{len(response)} 字符")
            
        except Exception as e:
            logger.error(f"❌ 生成响应失败：{str(e)}")
            traceback.print_exc()
            # 提供默认响应
            default_response = "抱歉，当前模型服务暂时不可用，无法回答你的问题。请检查API Key是否有效，或稍后再试。"
            yield default_response
            # 保存对话到记忆
            self.memory.add_message("user", query, time.time())
            self.memory.add_message("assistant", default_response, time.time())
    
    def add_astronomy_knowledge(self, knowledge: List[str]):
        """添加天文知识到RAG系统"""
        if not knowledge:
            logger.warning("⚠️  无有效知识可添加")
            return
        
        try:
            # 将字符串转换为Document对象
            from langchain.schema import Document
            documents = [Document(page_content=k) for k in knowledge]
            self.rag.add_documents(documents)
            logger.info(f"✅ 成功添加 {len(knowledge)} 条知识到RAG系统")
        except Exception as e:
            logger.error(f"❌ 添加知识失败：{str(e)}")
            traceback.print_exc()
    
    def clear_memory(self):
        """清空记忆"""
        try:
            self.memory.clear()
            logger.info("✅ 记忆已清空")
        except Exception as e:
            logger.error(f"❌ 清空记忆失败：{str(e)}")
            traceback.print_exc()
