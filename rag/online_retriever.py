"""
在线检索：只读打开 Chroma 向量库，提供相似度检索接口给 Agent 工具调用。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from config import settings
from logger import logger


class OnlineRetriever:
    def __init__(
        self,
        vector_db_path: str = settings.VECTOR_DB_PATH,
        collection_name: str = "astronomy_rag",
        top_k: int = 3,
    ):
        self.enabled = bool(settings.RAG_ENABLED)
        self.top_k = top_k

        if not self.enabled:
            self.db = None
            self.embeddings = None
            logger.warning("⚠️  RAG_ENABLED=False，在线检索已禁用")
            return

        self.embeddings = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )
        self.db = Chroma(
            embedding_function=self.embeddings,
            collection_name=collection_name,
            persist_directory=vector_db_path,
        )
        logger.info(f"✅ OnlineRetriever 已连接向量库：{vector_db_path}")

    def get_relevant_context(self, query: str, top_k: Optional[int] = None) -> str:
        if not self.enabled or not self.db:
            return ""

        k = top_k or self.top_k
        try:
            results = self.db.similarity_search(query, k=k)
            # 将元数据也拼进上下文，便于回答时引用来源
            parts: list[str] = []
            for doc in results:
                meta = doc.metadata or {}
                src = meta.get("source")
                rid = meta.get("record_id")
                prefix = []
                if src:
                    prefix.append(f"source={src}")
                if rid:
                    prefix.append(f"record_id={rid}")
                header = f"[{', '.join(prefix)}]" if prefix else ""
                parts.append((header + "\n" + doc.page_content).strip())
            context = "\n\n---\n\n".join(parts)
            logger.info(f"📄 RAG 检索上下文长度：{len(context)} 字符")
            return context
        except Exception as e:
            logger.error(f"❌ RAG 检索失败：{e}")
            return ""

