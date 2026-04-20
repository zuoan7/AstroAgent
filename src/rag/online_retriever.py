"""
增强型三级检索架构在线检索器

检索流程:
  Level 1 - 多路混合检索: 向量检索 + BM25 关键词检索 + 天文实体检索
  Level 2 - RRF 融合: Reciprocal Rank Fusion 算法合并多路检索结果
  Level 3 - 重排序: Rerank 模型 + 时效性/可信度/领域相关性综合重排序

增强功能:
  - 天文实体检索（第三路检索）
  - 多模态检索（图像/光谱）
  - 多级缓存（L1/L2/L3）
  - 结果过滤（可信度+时效性）
  - 检索质量监控
  - 知识更新机制
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger
from src.rag.bm25_retriever import BM25Retriever
from src.rag.cache import MultiLevelCache
from src.rag.entity_retriever import AstronomyEntityRetriever
from src.rag.knowledge_updater import KnowledgeUpdateManager
from src.rag.metrics import MetricsCollector, RetrievalMetrics, UserFeedback
from src.rag.multimodal_retriever import MultimodalRetriever
from src.rag.reranker import DashScopeReranker, RerankResult
from src.rag.result_filter import FilteredResult, ResultFilterAndReranker
from src.rag.rrf_fusion import RankedDocument, reciprocal_rank_fusion


class OnlineRetriever:
    """增强型三级检索架构在线检索器"""

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
        self.entity_weight = getattr(settings, "RAG_ENTITY_WEIGHT", 0.3)
        self.retrieval_candidates = settings.RAG_RETRIEVAL_CANDIDATES

        self.db = None
        self.embeddings = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.entity_retriever: Optional[AstronomyEntityRetriever] = None
        self.multimodal_retriever: Optional[MultimodalRetriever] = None
        self.reranker: Optional[DashScopeReranker] = None
        self.result_filter: Optional[ResultFilterAndReranker] = None
        self.cache: Optional[MultiLevelCache] = None
        self.metrics: Optional[MetricsCollector] = None
        self.knowledge_updater: Optional[KnowledgeUpdateManager] = None
        self._metrics_lock = threading.Lock()
        self._runtime_metrics: Dict[str, float] = {
            "rag_total_ms": 0.0,
            "rerank_ms": 0.0,
            "rag_call_count": 0.0,
        }

        if not self.enabled or not settings.DASHSCOPE_API_KEY:
            if not settings.DASHSCOPE_API_KEY:
                logger.warning("⚠️  DASHSCOPE_API_KEY 未配置，在线检索已禁用")
            else:
                logger.warning("⚠️  RAG_ENABLED=False，在线检索已禁用")
            return

        self._init_vector_store(collection_name)
        self._init_bm25()
        self._init_entity_retriever()
        self._init_multimodal()
        self._init_reranker()
        self._init_result_filter()
        self._init_cache()
        self._init_metrics()
        self._init_knowledge_updater()

        logger.info(
            f"✅ 增强型三级检索架构初始化完成: "
            f"向量={'✓' if self.db else '✗'}, "
            f"BM25={'✓' if self.bm25_retriever and self.bm25_retriever.bm25 else '✗'}, "
            f"实体={'✓' if self.entity_retriever else '✗'}, "
            f"多模态={'✓' if self.multimodal_retriever else '✗'}, "
            f"Rerank={'✓' if self.reranker and self.reranker.enabled else '✗'}, "
            f"过滤={'✓' if self.result_filter else '✗'}, "
            f"缓存={'✓' if self.cache else '✗'}, "
            f"监控={'✓' if self.metrics else '✗'}"
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

    def _init_entity_retriever(self) -> None:
        try:
            if self.bm25_retriever and self.bm25_retriever.documents:
                docs = []
                for i, doc_text in enumerate(self.bm25_retriever.documents):
                    meta = self.bm25_retriever.doc_metadata[i] if i < len(self.bm25_retriever.doc_metadata) else {}
                    docs.append({"content": doc_text, "metadata": meta})
                self.entity_retriever = AstronomyEntityRetriever(docs)
                logger.info("✅ 天文实体检索已加载")
            else:
                self.entity_retriever = AstronomyEntityRetriever()
                logger.info("✅ 天文实体检索已初始化（空索引）")
        except Exception as e:
            logger.warning(f"⚠️  天文实体检索初始化失败: {e}")
            self.entity_retriever = None

    def _init_multimodal(self) -> None:
        try:
            self.multimodal_retriever = MultimodalRetriever()
            logger.info("✅ 多模态检索已初始化")
        except Exception as e:
            logger.warning(f"⚠️  多模态检索初始化失败: {e}")
            self.multimodal_retriever = None

    def _init_reranker(self) -> None:
        try:
            self.reranker = DashScopeReranker()
        except Exception as e:
            logger.warning(f"⚠️  Reranker 初始化失败: {e}")
            self.reranker = None

    def _init_result_filter(self) -> None:
        try:
            self.result_filter = ResultFilterAndReranker()
            logger.info("✅ 结果过滤器已初始化")
        except Exception as e:
            logger.warning(f"⚠️  结果过滤器初始化失败: {e}")
            self.result_filter = None

    def _init_cache(self) -> None:
        try:
            if settings.RAG_CACHE_ENABLED:
                self.cache = MultiLevelCache(
                    l1_maxsize=64,
                    l1_ttl=60,
                    l2_maxsize=settings.RAG_CACHE_MAX_SIZE,
                    l2_ttl=settings.RAG_CACHE_TTL,
                )
                logger.info("✅ 多级缓存已启用")
        except Exception as e:
            logger.warning(f"⚠️  缓存初始化失败: {e}")
            self.cache = None

    def _init_metrics(self) -> None:
        try:
            self.metrics = MetricsCollector()
            logger.info("✅ 检索质量监控已启用")
        except Exception as e:
            logger.warning(f"⚠️  监控初始化失败: {e}")
            self.metrics = None

    def _init_knowledge_updater(self) -> None:
        try:
            self.knowledge_updater = KnowledgeUpdateManager(self.vector_db_path)
            logger.info("✅ 知识更新管理器已初始化")
        except Exception as e:
            logger.warning(f"⚠️  知识更新管理器初始化失败: {e}")
            self.knowledge_updater = None

    # ===== Level 1: 多路混合检索 =====

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

    def _entity_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.entity_retriever:
            return []
        try:
            results = self.entity_retriever.search_formatted(query, top_k=k)
            logger.info(f"📄 天文实体检索返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"❌ 天文实体检索失败: {e}")
            return []

    def _multimodal_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.multimodal_retriever:
            return []
        try:
            results = self.multimodal_retriever.search_formatted(query, top_k=k)
            logger.info(f"📄 多模态检索返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"❌ 多模态检索失败: {e}")
            return []

    def _hybrid_search(self, query: str, k: int) -> Dict[str, List[Dict[str, Any]]]:
        ranked_lists = {}

        vector_results = self._vector_search(query, k)
        if vector_results:
            ranked_lists["vector"] = vector_results

        bm25_results = self._bm25_search(query, k)
        if bm25_results:
            ranked_lists["bm25"] = bm25_results

        entity_results = self._entity_search(query, k)
        if entity_results:
            ranked_lists["entity"] = entity_results

        mm_results = self._multimodal_search(query, k)
        if mm_results:
            ranked_lists["multimodal"] = mm_results

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
        if "entity" in ranked_lists:
            weights["entity"] = self.entity_weight
        if "multimodal" in ranked_lists:
            weights["multimodal"] = 0.2

        return reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k=settings.RRF_K,
            weights=weights,
            top_k=top_k,
        )

    # ===== Level 3: 重排序 + 过滤 =====

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

    def _filter_results(
        self,
        results: List[RerankResult],
        query: str,
        top_k: int,
    ) -> List[FilteredResult]:
        if not self.result_filter:
            return self._rerank_to_filtered(results, top_k)

        return self.result_filter.filter_and_rerank(results, query, top_k=top_k)

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

    @staticmethod
    def _rerank_to_filtered(
        results: List[RerankResult],
        top_k: int,
    ) -> List[FilteredResult]:
        filtered = []
        for r in results[:top_k]:
            filtered.append(FilteredResult(
                content=r.content,
                relevance_score=r.relevance_score,
                timeliness_score=0.7,
                credibility_score=0.5,
                domain_score=0.5,
                final_score=r.relevance_score,
                metadata=r.metadata,
            ))
        return filtered

    # ===== 公共接口 =====

    def get_relevant_context(self, query: str, top_k: Optional[int] = None) -> str:
        return self.retrieve(query, top_k=top_k).get("context", "")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        增强型三级检索主入口

        流程: 多路混合检索 → RRF 融合 → Rerank 重排序 → 结果过滤 → 格式化输出

        Args:
            query: 查询文本
            top_k: 最终返回的文档数

        Returns:
            检索结果与阶段指标
        """
        if not self.enabled or not self.db:
            return {"context": "", "stages_ms": {}, "cache_hit": False, "rerank_used": False}

        k = top_k or self.top_k
        cache_hit = False
        overall_started = time.perf_counter()

        if self.cache:
            cached = self.cache.get(query, k)
            if cached is not None:
                cache_hit = True
                logger.debug(f"📄 检索缓存命中")
                self._record_metrics(query, 0.0, 0, 0.0, 0.0, cache_hit=True)
                self._add_runtime_metrics({"rag_total_ms": 0.0, "rag_call_count": 1.0})
                return {
                    "context": cached,
                    "stages_ms": {"rag_total_ms": 0.0},
                    "cache_hit": True,
                    "rerank_used": False,
                }

        try:
            start_time = time.time()
            stage_times = {}

            t0 = time.time()
            candidate_k = self.retrieval_candidates if not fast_mode else min(self.retrieval_candidates, max(k * 2, 6))
            ranked_lists = self._hybrid_search(query, candidate_k)
            stage_times["hybrid_search"] = time.time() - t0

            if not ranked_lists:
                logger.warning("⚠️  混合检索未返回任何结果")
                return {"context": "", "stages_ms": {}, "cache_hit": False, "rerank_used": False}

            t0 = time.time()
            rrf_results = self._rrf_merge(ranked_lists, candidate_k)
            stage_times["rrf_fusion"] = time.time() - t0

            if not rrf_results:
                logger.warning("⚠️  RRF 融合未返回任何结果")
                return {"context": "", "stages_ms": {}, "cache_hit": False, "rerank_used": False}

            rerank_used = self._should_use_rerank(query, rrf_results, fast_mode=fast_mode)
            if rerank_used:
                t0 = time.time()
                rerank_results = self._rerank(query, rrf_results, top_n=k * 2)
                stage_times["rerank"] = time.time() - t0
            else:
                rerank_results = self._rrf_to_rerank_results(rrf_results, top_n=k * 2)
                stage_times["rerank"] = 0.0

            t0 = time.time()
            filtered_results = self._filter_results(rerank_results, query, top_k=k)
            stage_times["filter"] = time.time() - t0

            final_docs = filtered_results[:k]

            context = self._format_filtered_results(final_docs)

            elapsed = time.time() - start_time
            latency_ms = elapsed * 1000

            top_score = final_docs[0].final_score if final_docs else 0.0
            avg_score = (
                sum(d.final_score for d in final_docs) / len(final_docs)
                if final_docs else 0.0
            )

            logger.info(
                f"📄 增强检索完成: 混合检索({len(ranked_lists)}路) → "
                f"RRF({len(rrf_results)}候选) → Rerank({len(rerank_results)}) → "
                f"过滤({len(final_docs)}最终), 耗时 {elapsed:.3f}s"
            )

            self._record_metrics(
                query, latency_ms, len(final_docs), top_score, avg_score,
                cache_hit=False, stages=stage_times,
            )
            runtime_stage_ms = {
                "rag_total_ms": round((time.perf_counter() - overall_started) * 1000.0, 2),
                "rerank_ms": round(stage_times.get("rerank", 0.0) * 1000.0, 2),
            }
            self._add_runtime_metrics({
                **runtime_stage_ms,
                "rag_call_count": 1.0,
            })

            if self.cache:
                self.cache.set(query, k, context)

            return {
                "context": context,
                "stages_ms": {
                    key: round(value * 1000.0, 2)
                    for key, value in stage_times.items()
                } | runtime_stage_ms,
                "cache_hit": False,
                "rerank_used": rerank_used,
                "result_count": len(final_docs),
            }

        except Exception as e:
            logger.error(f"❌ 增强检索失败: {e}")
            self._add_runtime_metrics({"rag_call_count": 1.0})
            return {"context": "", "stages_ms": {}, "cache_hit": False, "rerank_used": False}

    def _should_use_rerank(
        self,
        query: str,
        rrf_results: List[RankedDocument],
        *,
        fast_mode: bool,
    ) -> bool:
        if not self.reranker or not self.reranker.enabled:
            return False
        if not rrf_results:
            return False
        if not fast_mode:
            return True
        if len(query.strip()) > 24:
            return True
        top_score = rrf_results[0].rrf_score if rrf_results else 0.0
        return top_score < 0.2

    def _record_metrics(
        self,
        query: str,
        latency_ms: float,
        num_results: int,
        top_score: float,
        avg_score: float,
        cache_hit: bool = False,
        stages: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.metrics:
            return
        try:
            self.metrics.record_retrieval(RetrievalMetrics(
                query=query,
                latency_ms=latency_ms,
                num_results=num_results,
                top_score=top_score,
                avg_score=avg_score,
                cache_hit=cache_hit,
                pipeline_stages=stages or {},
            ))
        except Exception:
            pass

    def submit_feedback(
        self,
        query: str,
        relevance_rating: int,
        is_accurate: bool,
        comment: str = "",
    ) -> None:
        if not self.metrics:
            return
        try:
            self.metrics.record_feedback(UserFeedback(
                query=query,
                relevance_rating=relevance_rating,
                is_accurate=is_accurate,
                comment=comment,
            ))
        except Exception:
            pass

    @staticmethod
    def _format_filtered_results(results: List[FilteredResult]) -> str:
        parts: list[str] = []
        for result in results:
            meta = result.metadata or {}
            prefix_parts = []
            if meta.get("source"):
                prefix_parts.append(f"source={meta['source']}")
            if meta.get("doc_type"):
                prefix_parts.append(f"type={meta['doc_type']}")
            if meta.get("celestial_object"):
                prefix_parts.append(f"object={meta['celestial_object']}")
            if result.final_score > 0:
                prefix_parts.append(f"score={result.final_score:.4f}")
            if meta.get("observation_date"):
                prefix_parts.append(f"date={meta['observation_date']}")
            header = f"[{', '.join(prefix_parts)}]" if prefix_parts else ""
            parts.append((header + "\n" + result.content).strip())

        return "\n\n---\n\n".join(parts)

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
        stats = {
            "enabled": self.enabled,
            "vector_store": self.db is not None,
            "bm25": self.bm25_retriever is not None and self.bm25_retriever.bm25 is not None,
            "entity": self.entity_retriever is not None,
            "multimodal": self.multimodal_retriever is not None,
            "reranker": self.reranker is not None and self.reranker.enabled,
            "result_filter": self.result_filter is not None,
            "cache": self.cache is not None,
            "metrics": self.metrics is not None,
            "knowledge_updater": self.knowledge_updater is not None,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "entity_weight": self.entity_weight,
            "rrf_k": settings.RRF_K,
            "retrieval_candidates": self.retrieval_candidates,
            "rerank_top_n": settings.RERANK_TOP_N,
            "rerank_model": settings.RERANK_MODEL_NAME,
        }
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        if self.metrics:
            stats["metrics_summary"] = self.metrics.get_metrics_summary()
        return stats

    def update_knowledge(self, source: str = "online") -> Dict[str, int]:
        if not self.knowledge_updater:
            return {}
        new_docs = []
        if source in ("online", "nasa_apod"):
            new_docs.extend(self.knowledge_updater.fetch_nasa_apod())
        if source in ("online", "nasa_neo"):
            new_docs.extend(self.knowledge_updater.fetch_nasa_neo())
        if new_docs and self.knowledge_updater:
            return self.knowledge_updater.check_updates(new_docs, source=source)
        return {}

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        with self._metrics_lock:
            return dict(self._runtime_metrics)

    def _add_runtime_metrics(self, payload: Dict[str, float]) -> None:
        with self._metrics_lock:
            for key, value in payload.items():
                self._runtime_metrics[key] = self._runtime_metrics.get(key, 0.0) + value
