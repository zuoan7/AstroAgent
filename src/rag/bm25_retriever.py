"""
BM25 检索器 - 用于混合检索
使用 rank-bm25 库实现基于关键词的全文检索
支持 jieba 分词（可选依赖），回退到正则表达式分词
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Optional

from rank_bm25 import BM25Plus

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
        corpus_path: str = None,
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
            index_path = self.default_index_path(settings.VECTOR_DB_PATH)
        if corpus_path is None:
            corpus_path = self.default_corpus_path(settings.VECTOR_DB_PATH)

        self.index_path = index_path
        self.corpus_path = corpus_path
        self.bm25: Optional[BM25Plus] = None
        self.documents: list = []
        self.doc_metadata: list = []
        self.tokenized_documents: list[list[str]] = []

        if self.enabled:
            self._load_index()

    @staticmethod
    def default_index_path(base_path: str) -> str:
        return os.path.join(base_path, "bm25_index.pkl")

    @staticmethod
    def default_corpus_path(base_path: str) -> str:
        return os.path.join(base_path, "bm25_corpus.jsonl")

    def _load_index(self) -> None:
        try:
            data = self._load_index_payload()

            if data is None and os.path.exists(self.corpus_path):
                logger.info(f"📝 从 BM25 语料文件重建索引: {self.corpus_path}")
                self.documents, self.doc_metadata = self.load_corpus(self.corpus_path)
                if self.documents:
                    self._rebuild_runtime_index()
                    self._persist_index()
                    logger.info(f"✅ BM25 索引已从语料文件恢复: {len(self.documents)} 个文档")
                return

            if data is None:
                logger.warning(f"⚠️  BM25 索引文件不存在: {self.index_path}")
                logger.info("📝 需要先构建 BM25 索引，运行: python -m src.rag.build_bm25_index")
                return

            self.documents = data.get("documents", [])
            self.doc_metadata = data.get("metadata", [])

            if not self.documents:
                logger.warning("⚠️  BM25 索引为空")
                return

            tokenized_docs = data.get("tokenized_docs")
            if tokenized_docs:
                self.tokenized_documents = tokenized_docs
                self.bm25 = BM25Plus(tokenized_docs)
            else:
                self._rebuild_runtime_index()
            logger.info(f"✅ BM25 索引已加载: {len(self.documents)} 个文档")

        except Exception as e:
            logger.error(f"❌ 加载 BM25 索引失败: {e}")
            self.bm25 = None
            self.documents = []
            self.tokenized_documents = []

    def _load_index_payload(self) -> Optional[dict[str, Any]]:
        if not os.path.exists(self.index_path):
            return None
        with open(self.index_path, "rb") as f:
            return pickle.load(f)

    def _rebuild_runtime_index(self) -> None:
        self.tokenized_documents = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Plus(self.tokenized_documents)

    def _persist_index(self) -> None:
        parent_dir = os.path.dirname(self.index_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "metadata": self.doc_metadata,
                    "tokenized_docs": self.tokenized_documents,
                    "version": 2,
                },
                f,
            )

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
        return tokens if tokens else [text.lower()]

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
        return tokens if tokens else [text.lower()]

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        if not self.enabled or not self.bm25:
            return []

        k = top_k or self.top_k

        try:
            tokenized_query = self._tokenize(query)
            tokenized_query_set = set(tokenized_query)

            scores = self.bm25.get_scores(tokenized_query)

            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

            results = []
            for idx in top_indices:
                doc_tokens = (
                    self.tokenized_documents[idx]
                    if idx < len(self.tokenized_documents)
                    else self._tokenize(self.documents[idx])
                )
                if tokenized_query_set.intersection(doc_tokens):
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

        self._rebuild_runtime_index()
        self._persist_index()
        self.save_corpus(documents, metadata, self.corpus_path)

        logger.info(f"✅ BM25 索引已构建并保存: {len(documents)} 个文档")

    @staticmethod
    def load_corpus(corpus_path: str) -> tuple[list[str], list[dict[str, Any]]]:
        documents: list[str] = []
        metadata: list[dict[str, Any]] = []
        if not os.path.exists(corpus_path):
            return documents, metadata

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                content = item.get("content", "")
                if not content:
                    continue
                documents.append(content)
                metadata.append(item.get("metadata", {}) or {})
        return documents, metadata

    @staticmethod
    def save_corpus(
        documents: list[str],
        metadata: list[dict[str, Any]],
        corpus_path: str,
    ) -> None:
        parent_dir = os.path.dirname(corpus_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(corpus_path, "w", encoding="utf-8") as f:
            for index, content in enumerate(documents):
                if not content:
                    continue
                record = {
                    "content": content,
                    "metadata": metadata[index] if index < len(metadata) else {},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_bm25_index_from_chroma(
    chroma_db,
    output_path: str = None,
    corpus_path: str = None,
) -> None:
    if output_path is None:
        output_path = BM25Retriever.default_index_path(settings.VECTOR_DB_PATH)
    if corpus_path is None:
        corpus_path = BM25Retriever.default_corpus_path(settings.VECTOR_DB_PATH)

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

        retriever = BM25Retriever(index_path=output_path, corpus_path=corpus_path)
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
