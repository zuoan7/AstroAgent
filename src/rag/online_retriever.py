"""
三级检索架构在线检索器

检索流程:
  Level 1 - 混合检索: 向量检索 + BM25 关键词检索并行执行
  Level 2 - RRF 融合: Reciprocal Rank Fusion 算法合并多路检索结果
  Level 3 - Rerank 重排序: qwen3-rerank 模型对候选文档精细化排序

性能优化:
  - TTLCache 缓存检索结果，避免重复计算
  - 检索候选数 > 最终返回数，确保 Rerank 有足够候选
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from cachetools import TTLCache

from src.core.config import settings
from src.core.logger import logger
from src.rag.bm25_retriever import BM25Retriever
from src.rag.reranker import DashScopeReranker, RerankResult
from src.rag.rrf_fusion import RankedDocument, reciprocal_rank_fusion


class OnlineRetriever:
    """三级检索架构在线检索器"""

    def __init__(
        self,
        vector_db_path: str = None,
        collection_name: str = "astronomy_rag",
        top_k: int = 3,
    ):
        self.enabled = bool(settings.RAG_ENABLED)
        self.top_k = top_k
        self.vector_db_path = vector_db_path or settings.VECTOR_DB_PATH

        self.vector_weight = settings.RAG_VECTOR_WEIGHT
        self.bm25_weight = settings.RAG_BM25_WEIGHT
        self.retrieval_candidates = settings.RAG_RETRIEVAL_CANDIDATES

        self.db = None
        self.embeddings = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.reranker: Optional[DashScopeReranker] = None
        self._cache: Optional[TTLCache] = None

        if not self.enabled or not settings.DASHSCOPE_API_KEY:
            if not settings.DASHSCOPE_API_KEY:
                logger.warning("⚠️  DASHSCOPE_API_KEY 未配置，在线检索已禁用")
            else:
                logger.warning("⚠️  RAG_ENABLED=False，在线检索已禁用")
            return

        self._init_vector_store(collection_name)
        self._init_bm25()
        self._init_reranker()
        self._init_cache()

        logger.info(
            f"✅ 三级检索架构初始化完成: "
            f"向量={'✓' if self.db else '✗'}, "
            f"BM25={'✓' if self.bm25_retriever and self.bm25_retriever.bm25 else '✗'}, "
            f"Rerank={'✓' if self.reranker and self.reranker.enabled else '✗'}, "
            f"缓存={'✓' if self._cache else '✗'}"
        )

    def _init_vector_store(self, collection_name: str) -> None:
        try:
            from langchain_chroma import Chroma
            from langchain_community.embeddings import DashScopeEmbeddings

            self.embeddings = DashScopeEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
            )
            self.db = Chroma(
                embedding_function=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.vector_db_path,
            )
            logger.info(f"✅ 向量检索已连接: {self.vector_db_path}")
        except Exception as e:
            logger.error(f"❌ 向量检索初始化失败: {e}")
            self.db = None

    def _init_bm25(self) -> None:
        try:
            self.bm25_retriever = BM25Retriever(
                index_path=self.vector_db_path + "/bm25_index.pkl",
                top_k=self.retrieval_candidates,
            )
            if self.bm25_retriever.bm25 is None:
                logger.warning("⚠️  BM25 索引未加载，将使用纯向量检索")
                self.bm25_retriever = None
            else:
                logger.info("✅ BM25 检索已加载")
        except Exception as e:
            logger.warning(f"⚠️  BM25 检索初始化失败: {e}，将使用纯向量检索")
            self.bm25_retriever = None

    def _init_reranker(self) -> None:
        try:
            self.reranker = DashScopeReranker()
        except Exception as e:
            logger.warning(f"⚠️  Reranker 初始化失败: {e}")
            self.reranker = None

    def _init_cache(self) -> None:
        if settings.RAG_CACHE_ENABLED:
            self._cache = TTLCache(
                maxsize=settings.RAG_CACHE_MAX_SIZE,
                ttl=settings.RAG_CACHE_TTL,
            )
            logger.info(f"✅ 检索缓存已启用: maxsize={settings.RAG_CACHE_MAX_SIZE}, ttl={settings.RAG_CACHE_TTL}s")

    @staticmethod
    def _cache_key(query: str, top_k: int) -> str:
        raw = f"{query}::{top_k}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[str]:
        if self._cache is None:
            return None
        return self._cache.get(key)

    def _set_cached(self, key: str, value: str) -> None:
        if self._cache is None:
            return
        self._cache[key] = value

    # ===== Level 1: 混合检索 =====

    def _vector_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.db:
            return []
        try:
            results_with_scores = self.db.similarity_search_with_score(query, k=k)
            formatted = []
            for doc, score in results_with_scores:
                formatted.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata or {},
                    "score": float(score),
                })
            logger.info(f"📄 向量检索返回 {len(formatted)} 个结果")
            return formatted
        except Exception as e:
            logger.error(f"❌ 向量检索失败: {e}")
            return []

    def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.bm25_retriever:
            return []
        try:
            raw_results = self.bm25_retriever.search(query, top_k=k)
            formatted = []
            for r in raw_results:
                formatted.append({
                    "content": r.get("document", ""),
                    "metadata": r.get("metadata", {}),
                    "score": r.get("score", 0.0),
                })
            logger.info(f"📄 BM25 检索返回 {len(formatted)} 个结果")
            return formatted
        except Exception as e:
            logger.error(f"❌ BM25 检索失败: {e}")
            return []

    def _hybrid_search(self, query: str, k: int) -> Dict[str, List[Dict[str, Any]]]:
        vector_results = self._vector_search(query, k)
        bm25_results = self._bm25_search(query, k)
        ranked_lists = {}
        if vector_results:
            ranked_lists["vector"] = vector_results
        if bm25_results:
            ranked_lists["bm25"] = bm25_results
        return ranked_lists

    # ===== Level 2: RRF 融合 =====

    def _rrf_merge(
        self,
        ranked_lists: Dict[str, List[Dict[str, Any]]],
        top_k: int,
    ) -> List[RankedDocument]:
        weights = {}
        if "vector" in ranked_lists:
            weights["vector"] = self.vector_weight
        if "bm25" in ranked_lists:
            weights["bm25"] = self.bm25_weight

        return reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k=settings.RRF_K,
            weights=weights,
            top_k=top_k,
        )

    # ===== Level 3: Rerank 重排序 =====

    def _rerank(
        self,
        query: str,
        rrf_results: List[RankedDocument],
        top_n: int,
    ) -> List[RerankResult]:
        if not self.reranker or not self.reranker.enabled:
            return self._rrf_to_rerank_results(rrf_results, top_n)

        documents = [doc.content for doc in rrf_results]
        if not documents:
            return []

        rerank_results = self.reranker.rerank(query, documents, top_n=top_n)

        final_results = []
        for rr in rerank_results:
            original_idx = rr.index
            if original_idx < len(rrf_results):
                final_results.append(RerankResult(
                    index=original_idx,
                    relevance_score=rr.relevance_score,
                    content=rr.content,
                    metadata=rrf_results[original_idx].metadata,
                ))
            else:
                final_results.append(rr)

        logger.info(f"✅ Rerank 重排序完成: {len(rrf_results)} -> {len(final_results)} 个结果")
        return final_results

    @staticmethod
    def _rrf_to_rerank_results(
        rrf_results: List[RankedDocument],
        top_n: int,
    ) -> List[RerankResult]:
        results = []
        for i, doc in enumerate(rrf_results[:top_n]):
            results.append(RerankResult(
                index=i,
                relevance_score=doc.rrf_score,
                content=doc.content,
                metadata=doc.metadata,
            ))
        return results

    # ===== 公共接口 =====

    def get_relevant_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        三级检索主入口

        流程: 混合检索 → RRF 融合 → Rerank 重排序 → 格式化输出

        Args:
            query: 查询文本
            top_k: 最终返回的文档数

        Returns:
            格式化后的上下文字符串
        """
        if not self.enabled or not self.db:
            return ""

        k = top_k or self.top_k

        cache_key = self._cache_key(query, k)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"📄 检索缓存命中: {cache_key[:8]}...")
            return cached

        try:
            start_time = time.time()

            ranked_lists = self._hybrid_search(query, self.retrieval_candidates)

            if not ranked_lists:
                logger.warning("⚠️  混合检索未返回任何结果")
                return ""

            rrf_results = self._rrf_merge(ranked_lists, self.retrieval_candidates)

            if not rrf_results:
                logger.warning("⚠️  RRF 融合未返回任何结果")
                return ""

            rerank_results = self._rerank(query, rrf_results, top_n=k)

            final_docs = rerank_results[:k]

            context = self._format_results(final_docs)

            elapsed = time.time() - start_time
            logger.info(
                f"📄 三级检索完成: 混合检索({len(ranked_lists)}路) → "
                f"RRF({len(rrf_results)}候选) → Rerank({len(final_docs)}最终), "
                f"耗时 {elapsed:.3f}s"
            )

            self._set_cached(cache_key, context)
            return context

        except Exception as e:
            logger.error(f"❌ 三级检索失败: {e}")
            return ""

    @staticmethod
    def _format_results(results: List[RerankResult]) -> str:
        parts: list[str] = []
        for result in results:
            meta = result.metadata or {}
            src = meta.get("source")
            rid = meta.get("record_id")
            prefix = []
            if src:
                prefix.append(f"source={src}")
            if rid:
                prefix.append(f"record_id={rid}")
            if result.relevance_score > 0:
                prefix.append(f"score={result.relevance_score:.4f}")
            header = f"[{', '.join(prefix)}]" if prefix else ""
            parts.append((header + "\n" + result.content).strip())

        return "\n\n---\n\n".join(parts)

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

    def get_pipeline_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "vector_store": self.db is not None,
            "bm25": self.bm25_retriever is not None and self.bm25_retriever.bm25 is not None,
            "reranker": self.reranker is not None and self.reranker.enabled,
            "cache": self._cache is not None,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "rrf_k": settings.RRF_K,
            "retrieval_candidates": self.retrieval_candidates,
            "rerank_top_n": settings.RERANK_TOP_N,
            "rerank_model": settings.RERANK_MODEL_NAME,
        }
