"""
BM25 检索器 - 用于混合检索
使用 rank-bm25 库实现基于关键词的全文检索
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from config import settings
from logger import logger


class BM25Retriever:
    """BM25 关键词检索器"""

    def __init__(
        self,
        index_path: str = None,
        top_k: int = 3,
    ):
        self.enabled = bool(settings.RAG_ENABLED)
        self.top_k = top_k

        # 默认索引路径
        if index_path is None:
            index_path = os.path.join(settings.VECTOR_DB_PATH, "bm25_index.pkl")

        self.index_path = index_path
        self.bm25: Optional[BM25Okapi] = None
        self.documents: list = []
        self.doc_metadata: list = []

        if self.enabled:
            self._load_index()

    def _load_index(self) -> None:
        """加载 BM25 索引"""
        if not os.path.exists(self.index_path):
            logger.warning(f"⚠️  BM25 索引文件不存在: {self.index_path}")
            logger.info("📝 需要先构建 BM25 索引，运行: python -m rag.build_bm25_index")
            return

        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data.get("documents", [])
                self.doc_metadata = data.get("metadata", [])

            if not self.documents:
                logger.warning("⚠️  BM25 索引为空")
                return

            # 使用文档内容创建 BM25 索引
            # 对中文进行简单的分词处理
            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(f"✅ BM25 索引已加载: {len(self.documents)} 个文档")

        except Exception as e:
            logger.error(f"❌ 加载 BM25 索引失败: {e}")
            self.bm25 = None
            self.documents = []

    def _tokenize(self, text: str) -> list[str]:
        """
        简单的中文分词
        对于中文，按字符分割（保留单字和双字组合）
        """
        import re
        tokens = []
        # 词组（2-4个字符）
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        tokens.extend(chinese_words)
        # 提取英文/数字词
        english_words = re.findall(r'[a-zA-Z0-9]{2,}', text.lower())
        tokens.extend(english_words)
        # 提取单字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens.extend(chinese_chars)
        # 去重但保持顺序
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        return unique_tokens if unique_tokens else [text.lower()]

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        执行 BM25 检索

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            检索结果列表，每个元素包含 document, metadata, score
        """
        if not self.enabled or not self.bm25:
            return []

        k = top_k or self.top_k

        try:
            # 对查询进行分词
            tokenized_query = self._tokenize(query)

            # 计算 BM25 分数
            scores = self.bm25.get_scores(tokenized_query)

            # 获取 top-k 结果的索引
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # 只返回有分数的结果
                    results.append({
                        "document": self.documents[idx],
                        "metadata": self.doc_metadata[idx] if idx < len(self.doc_metadata) else {},
                        "score": scores[idx],
                        "index": idx
                    })

            logger.info(f"📄 BM25 检索返回 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"❌ BM25 检索失败: {e}")
            return []

    def build_index(self, documents: list[str], metadata: list[dict]) -> None:
        """
        构建 BM25 索引

        Args:
            documents: 文档内容列表
            metadata: 对应的元数据列表
        """
        if not documents:
            logger.warning("⚠️  没有文档可索引")
            return

        self.documents = documents
        self.doc_metadata = metadata

        # 分词
        tokenized_docs = [self._tokenize(doc) for doc in documents]

        # 创建 BM25 索引
        self.bm25 = BM25Okapi(tokenized_docs)

        # 保存索引
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "documents": documents,
                "metadata": metadata
            }, f)

        logger.info(f"✅ BM25 索引已构建并保存: {len(documents)} 个文档")


def build_bm25_index_from_chroma(chroma_db, output_path: str = None) -> None:
    """
    从 Chroma 向量库构建 BM25 索引

    Args:
        chroma_db: Chroma 向量数据库实例
        output_path: 输出路径
    """
    if output_path is None:
        output_path = os.path.join(settings.VECTOR_DB_PATH, "bm25_index.pkl")

    try:
        # 获取所有文档
        results = chroma_db.get()

        documents = []
        metadata = []

        for i, doc in enumerate(results.get("documents", [])):
            if doc:  # 跳过空文档
                documents.append(doc)
                metadata.append(results.get("metadatas", [{}])[i] if results.get("metadatas") else {})

        if not documents:
            logger.warning("⚠️  Chroma 中没有文档")
            return

        # 构建 BM25 索引
        retriever = BM25Retriever(index_path=output_path)
        retriever.build_index(documents, metadata)

        logger.info(f"✅ 从 Chroma 构建 BM25 索引成功: {len(documents)} 个文档")

    except Exception as e:
        logger.error(f"❌ 构建 BM25 索引失败: {e}")


if __name__ == "__main__":
    # 测试用：直接构建 BM25 索引
    from langchain_chroma import Chroma
    from langchain_community.embeddings import DashScopeEmbeddings

    logger.info("开始构建 BM25 索引...")

    # 连接 Chroma
    embeddings = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )

    chroma_db = Chroma(
        embedding_function=embeddings,
        collection_name="astronomy_rag",
        persist_directory=settings.VECTOR_DB_PATH,
    )

    # 构建 BM25 索引
    build_bm25_index_from_chroma(chroma_db)
