"""
在线检索：支持混合检索（向量检索 + BM25），提供相似度检索接口给 Agent 工具调用。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from src.core.config import settings
from src.core.logger import logger


class OnlineRetriever:
    def __init__(
        self,
        vector_db_path: str = settings.VECTOR_DB_PATH,
        collection_name: str = "astronomy_rag",
        top_k: int = 3,
        use_hybrid: bool = True,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        self.enabled = bool(settings.RAG_ENABLED)
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        if not self.enabled or not settings.DASHSCOPE_API_KEY:
            self.db = None
            self.embeddings = None
            self.bm25_retriever = None
            if not settings.DASHSCOPE_API_KEY:
                logger.warning("⚠️  DASHSCOPE_API_KEY 未配置，在线检索已禁用")
            else:
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
        logger.info(f"✅ 向量检索已连接：{vector_db_path}")

        self.bm25_retriever = None
        if use_hybrid:
            try:
                from src.rag.bm25_retriever import BM25Retriever
                self.bm25_retriever = BM25Retriever(
                    index_path=vector_db_path + "/bm25_index.pkl",
                    top_k=top_k
                )
                if self.bm25_retriever.bm25 is None:
                    logger.warning("⚠️  BM25 索引未加载，将使用纯向量检索")
                    self.use_hybrid = False
                else:
                    logger.info("✅ BM25 检索已加载")
            except Exception as e:
                logger.warning(f"⚠️  BM25 检索初始化失败: {e}，将使用纯向量检索")
                self.use_hybrid = False

    @staticmethod
    def _min_max_normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [1.0] * len(scores) if max_s > 0 else [0.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def _merge_results(
        self,
        vector_results: list,
        vector_distances: list[float],
        bm25_results: list,
        top_k: int
    ) -> list:
        if not bm25_results:
            return vector_results[:top_k]
        if not vector_results:
            return [
                self._bm25_result_to_doc(r) for r in bm25_results[:top_k]
            ]

        vector_scores_raw = []
        for dist in vector_distances:
            vector_scores_raw.append(1.0 / (1.0 + dist))

        bm25_scores_raw = [r.get("score", 0.0) for r in bm25_results]

        vector_norm = self._min_max_normalize(vector_scores_raw)
        bm25_norm = self._min_max_normalize(bm25_scores_raw)

        all_results: dict[str, dict] = {}

        for i, doc in enumerate(vector_results):
            doc_id = doc.page_content[:100]
            all_results[doc_id] = {
                "normalized_score": vector_norm[i] * self.vector_weight,
                "doc": doc,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "vector",
            }

        for i, result in enumerate(bm25_results):
            doc_id = result.get("document", "")[:100]
            bm25_entry = {
                "normalized_score": bm25_norm[i] * self.bm25_weight,
                "content": result.get("document", ""),
                "metadata": result.get("metadata", {}),
                "source": "bm25",
            }
            if doc_id in all_results:
                all_results[doc_id]["normalized_score"] = (
                    all_results[doc_id]["normalized_score"] + bm25_entry["normalized_score"]
                )
                all_results[doc_id]["source"] = "hybrid"
            else:
                bm25_entry["doc"] = self._bm25_result_to_doc(result)
                all_results[doc_id] = bm25_entry

        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get("normalized_score", 0),
            reverse=True
        )[:top_k]

        merged = []
        for item in sorted_results:
            merged.append(item["doc"])

        return merged

    @staticmethod
    def _bm25_result_to_doc(result: dict):
        content = result.get("document", "")
        metadata = result.get("metadata", {})
        from langchain_core.documents import Document
        return Document(page_content=content, metadata=metadata)

    def get_relevant_context(self, query: str, top_k: Optional[int] = None) -> str:
        if not self.enabled or not self.db:
            return ""

        k = top_k or self.top_k

        try:
            results = []

            vector_results_with_scores = self.db.similarity_search_with_score(query, k=k)
            vector_results = [doc for doc, _ in vector_results_with_scores]
            vector_distances = [score for _, score in vector_results_with_scores]

            bm25_results = []
            if self.use_hybrid and self.bm25_retriever:
                bm25_results = self.bm25_retriever.search(query, top_k=k)

            if self.use_hybrid and bm25_results:
                results = self._merge_results(vector_results, vector_distances, bm25_results, k)
                logger.info(f"📄 混合检索：向量 {len(vector_results)} + BM25 {len(bm25_results)} -> 合并 {len(results)}")
            else:
                results = vector_results
                logger.info(f"📄 向量检索：{len(results)} 个结果")

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

    def get_vector_results(self, query: str, top_k: Optional[int] = None) -> list:
        if not self.enabled or not self.db:
            return []

        k = top_k or self.top_k
        return self.db.similarity_search(query, k=k)

    def get_bm25_results(self, query: str, top_k: Optional[int] = None) -> list:
        if not self.bm25_retriever:
            return []

        k = top_k or self.top_k
        return self.bm25_retriever.search(query, top_k=k)
