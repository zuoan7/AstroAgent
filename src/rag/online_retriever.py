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
        """
        初始化在线检索器

        Args:
            vector_db_path: 向量数据库路径
            collection_name: collection 名称
            top_k: 返回结果数量
            use_hybrid: 是否使用混合检索
            vector_weight: 向量检索权重 (0-1)
            bm25_weight: BM25 检索权重 (0-1)
        """
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

        # 初始化向量检索
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

        # 初始化 BM25 检索器
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

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """将分数归一化到 0-1 范围"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores) if max_score > 0 else [0.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _merge_results(
        self,
        vector_results: list,
        bm25_results: list,
        top_k: int
    ) -> list:
        """
        合并向量检索和 BM25 检索的结果

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果
            top_k: 返回结果数量

        Returns:
            合并后的结果列表
        """
        if not bm25_results:
            return vector_results[:top_k]
        if not vector_results:
            return bm25_results[:top_k]

        # 创建文档 ID 到分数的映射
        vector_scores = {}
        for i, doc in enumerate(vector_results):
            # 使用文档内容作为 key
            doc_id = doc.page_content[:100]  # 取前100字符作为 ID
            score = 1.0 - (i * 0.1)  # 简化的分数：排名越前分数越高
            vector_scores[doc_id] = {
                "score": score,
                "doc": doc,
                "source": "vector"
            }

        bm25_scores = {}
        for result in bm25_results:
            doc_id = result.get("document", "")[:100]
            bm25_scores[doc_id] = {
                "score": result.get("score", 0),
                "doc": result.get("document", ""),
                "metadata": result.get("metadata", {}),
                "source": "bm25"
            }

        # 归一化分数
        vector_score_values = [v["score"] for v in vector_scores.values()]
        bm25_score_values = [b["score"] for b in bm25_scores.values()]

        vector_norm = self._normalize_scores(vector_score_values)
        bm25_norm = self._normalize_scores(bm25_score_values)

        # 更新归一化后的分数
        for i, (doc_id, data) in enumerate(vector_scores.items()):
            data["normalized_score"] = vector_norm[i] * self.vector_weight
            data["content"] = data["doc"].page_content
            data["metadata"] = data["doc"].metadata

        for i, (doc_id, data) in enumerate(bm25_scores.items()):
            data["normalized_score"] = bm25_norm[i] * self.bm25_weight
            data["content"] = data["doc"]
            # 将 bm25 结果包装为类似向量结果的格式
            data["doc"] = type('obj', (object,), {
                "page_content": data["content"],
                "metadata": data.get("metadata", {})
            })()

        # 合并所有结果
        all_results = {}
        for doc_id, data in vector_scores.items():
            all_results[doc_id] = data

        for doc_id, data in bm25_scores.items():
            if doc_id in all_results:
                # 如果已存在，取较高分数
                all_results[doc_id]["normalized_score"] = max(
                    all_results[doc_id]["normalized_score"],
                    data["normalized_score"]
                )
                all_results[doc_id]["source"] = "hybrid"
            else:
                all_results[doc_id] = data

        # 按分数排序
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get("normalized_score", 0),
            reverse=True
        )[:top_k]

        # 转换为标准格式
        merged = []
        for item in sorted_results:
            doc = item["doc"]
            merged.append(doc)

        return merged

    def get_relevant_context(self, query: str, top_k: Optional[int] = None) -> str:
        if not self.enabled or not self.db:
            return ""

        k = top_k or self.top_k

        try:
            results = []

            # 向量检索
            vector_results = self.db.similarity_search(query, k=k)

            # BM25 检索（如果可用）
            bm25_results = []
            if self.use_hybrid and self.bm25_retriever:
                bm25_results = self.bm25_retriever.search(query, top_k=k)

            # 合并结果
            if self.use_hybrid and bm25_results:
                results = self._merge_results(vector_results, bm25_results, k)
                logger.info(f"📄 混合检索：向量 {len(vector_results)} + BM25 {len(bm25_results)} -> 合并 {len(results)}")
            else:
                results = vector_results
                logger.info(f"📄 向量检索：{len(results)} 个结果")

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

    def get_vector_results(self, query: str, top_k: Optional[int] = None) -> list:
        """仅获取向量检索结果"""
        if not self.enabled or not self.db:
            return []

        k = top_k or self.top_k
        return self.db.similarity_search(query, k=k)

    def get_bm25_results(self, query: str, top_k: Optional[int] = None) -> list:
        """仅获取 BM25 检索结果"""
        if not self.bm25_retriever:
            return []

        k = top_k or self.top_k
        return self.bm25_retriever.search(query, top_k=k)
