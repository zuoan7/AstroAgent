"""
BM25 检索器 - 用于混合检索
使用 rank-bm25 库实现基于关键词的全文检索
支持 jieba 分词（可选依赖），回退到正则表达式分词
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from src.core.config import settings
from src.core.logger import logger

_jieba_available = False
try:
    import jieba
    _jieba_available = True
except ImportError:
    pass


class BM25Retriever:
    """BM25 关键词检索器"""

    def __init__(
        self,
        index_path: str = None,
        top_k: int = 3,
        use_jieba: Optional[bool] = None,
    ):
        self.enabled = bool(settings.RAG_ENABLED)
        self.top_k = top_k

        if use_jieba is None:
            self.use_jieba = _jieba_available
        else:
            self.use_jieba = use_jieba and _jieba_available

        if self.use_jieba and _jieba_available:
            logger.info("✅ BM25 使用 jieba 分词模式")
        else:
            logger.info("✅ BM25 使用正则表达式分词模式（jieba 未安装或已禁用）")

        if index_path is None:
            index_path = os.path.join(settings.VECTOR_DB_PATH, "bm25_index.pkl")

        self.index_path = index_path
        self.bm25: Optional[BM25Okapi] = None
        self.documents: list = []
        self.doc_metadata: list = []

        if self.enabled:
            self._load_index()

    def _load_index(self) -> None:
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

            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(f"✅ BM25 索引已加载: {len(self.documents)} 个文档")

        except Exception as e:
            logger.error(f"❌ 加载 BM25 索引失败: {e}")
            self.bm25 = None
            self.documents = []

    def _tokenize(self, text: str) -> list[str]:
        if self.use_jieba and _jieba_available:
            return self._tokenize_jieba(text)
        return self._tokenize_regex(text)

    @staticmethod
    def _tokenize_jieba(text: str) -> list[str]:
        tokens = []
        for word in jieba.cut(text):
            word = word.strip()
            if word:
                tokens.append(word)
        english_words = __import__("re").findall(r'[a-zA-Z0-9]{2,}', text.lower())
        tokens.extend(english_words)
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        return unique_tokens if unique_tokens else [text.lower()]

    @staticmethod
    def _tokenize_regex(text: str) -> list[str]:
        import re
        tokens = []
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        tokens.extend(chinese_words)
        english_words = re.findall(r'[a-zA-Z0-9]{2,}', text.lower())
        tokens.extend(english_words)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens.extend(chinese_chars)
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        return unique_tokens if unique_tokens else [text.lower()]

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        if not self.enabled or not self.bm25:
            return []

        k = top_k or self.top_k

        try:
            tokenized_query = self._tokenize(query)

            scores = self.bm25.get_scores(tokenized_query)

            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:
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
        if not documents:
            logger.warning("⚠️  没有文档可索引")
            return

        self.documents = documents
        self.doc_metadata = metadata

        tokenized_docs = [self._tokenize(doc) for doc in documents]

        self.bm25 = BM25Okapi(tokenized_docs)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "documents": documents,
                "metadata": metadata
            }, f)

        logger.info(f"✅ BM25 索引已构建并保存: {len(documents)} 个文档")


def build_bm25_index_from_chroma(chroma_db, output_path: str = None) -> None:
    if output_path is None:
        output_path = os.path.join(settings.VECTOR_DB_PATH, "bm25_index.pkl")

    try:
        results = chroma_db.get()

        documents = []
        metadata = []

        for i, doc in enumerate(results.get("documents", [])):
            if doc:
                documents.append(doc)
                metadata.append(results.get("metadatas", [{}])[i] if results.get("metadatas") else {})

        if not documents:
            logger.warning("⚠️  Chroma 中没有文档")
            return

        retriever = BM25Retriever(index_path=output_path)
        retriever.build_index(documents, metadata)

        logger.info(f"✅ 从 Chroma 构建 BM25 索引成功: {len(documents)} 个文档")

    except Exception as e:
        logger.error(f"❌ 构建 BM25 索引失败: {e}")


if __name__ == "__main__":
    from langchain_chroma import Chroma
    from langchain_community.embeddings import DashScopeEmbeddings

    logger.info("开始构建 BM25 索引...")

    embeddings = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )

    chroma_db = Chroma(
        embedding_function=embeddings,
        collection_name="astronomy_rag",
        persist_directory=settings.VECTOR_DB_PATH,
    )

    build_bm25_index_from_chroma(chroma_db)
