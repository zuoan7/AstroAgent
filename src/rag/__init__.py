"""
RAG 模块 - 增强型三级检索架构

Level 1: 多路混合检索（向量 + BM25 + 天文实体 + 多模态）
Level 2: RRF (Reciprocal Rank Fusion) 融合
Level 3: Rerank 模型重排序 + 时效性/可信度过滤

增强模块:
  - astronomy_chunker: 天文领域专用文档分块器
  - entity_retriever: 天文专业实体检索器
  - multimodal_retriever: 多模态检索器
  - result_filter: 时效性感知重排序与结果过滤
  - cache: 多级缓存系统
  - knowledge_updater: 知识更新机制
  - metrics: 检索质量监控与用户反馈
"""

from src.rag.rrf_fusion import RankedDocument, reciprocal_rank_fusion
from src.rag.reranker import DashScopeReranker, RerankResult
from src.rag.online_retriever import OnlineRetriever
from src.rag.result_filter import FilteredResult, ResultFilterAndReranker
from src.rag.astronomy_chunker import AstronomyChunker, AstronomyMetadata, AstronomyContentType
from src.rag.entity_retriever import AstronomyEntityRetriever, AstronomyEntityRecognizer
from src.rag.multimodal_retriever import MultimodalRetriever
from src.rag.cache import MultiLevelCache
from src.rag.knowledge_updater import KnowledgeUpdateManager
from src.rag.metrics import MetricsCollector, RetrievalMetrics, UserFeedback

__all__ = [
    "OnlineRetriever",
    "reciprocal_rank_fusion",
    "RankedDocument",
    "DashScopeReranker",
    "RerankResult",
    "FilteredResult",
    "ResultFilterAndReranker",
    "AstronomyChunker",
    "AstronomyMetadata",
    "AstronomyContentType",
    "AstronomyEntityRetriever",
    "AstronomyEntityRecognizer",
    "MultimodalRetriever",
    "MultiLevelCache",
    "KnowledgeUpdateManager",
    "MetricsCollector",
    "RetrievalMetrics",
    "UserFeedback",
]
