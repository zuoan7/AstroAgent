"""
三级检索架构单元测试

覆盖:
  - RRF 融合算法
  - DashScope Reranker
  - OnlineRetriever 三级流水线
  - 缓存机制
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.rag.rrf_fusion import RankedDocument, reciprocal_rank_fusion
from src.rag.reranker import DashScopeReranker, RerankResult


# ===== RRF 融合算法测试 =====

class TestReciprocalRankFusion:
    """RRF 算法核心逻辑测试"""

    def test_empty_input(self):
        result = reciprocal_rank_fusion({})
        assert result == []

    def test_single_retriever(self):
        ranked_lists = {
            "vector": [
                {"content": "doc1", "metadata": {"source": "a"}},
                {"content": "doc2", "metadata": {"source": "b"}},
            ]
        }
        result = reciprocal_rank_fusion(ranked_lists, k=60, top_k=10)
        assert len(result) == 2
        assert result[0].content == "doc1"
        assert result[0].rrf_score == pytest.approx(1.0 / (60 + 1), abs=1e-6)
        assert result[1].rrf_score == pytest.approx(1.0 / (60 + 2), abs=1e-6)

    def test_two_retrievers_no_overlap(self):
        ranked_lists = {
            "vector": [{"content": "doc_a"}],
            "bm25": [{"content": "doc_b"}],
        }
        result = reciprocal_rank_fusion(ranked_lists, k=60, top_k=10)
        assert len(result) == 2
        scores = {r.content: r.rrf_score for r in result}
        assert scores["doc_a"] == pytest.approx(1.0 / 61, abs=1e-6)
        assert scores["doc_b"] == pytest.approx(1.0 / 61, abs=1e-6)

    def test_two_retrievers_with_overlap(self):
        ranked_lists = {
            "vector": [
                {"content": "shared_doc"},
                {"content": "only_vector"},
            ],
            "bm25": [
                {"content": "only_bm25"},
                {"content": "shared_doc"},
            ],
        }
        result = reciprocal_rank_fusion(ranked_lists, k=60, top_k=10)
        assert len(result) == 3
        shared = next(r for r in result if r.content == "shared_doc")
        assert shared.rrf_score == pytest.approx(1.0 / 61 + 1.0 / 62, abs=1e-6)
        assert shared.source == "hybrid"

    def test_weighted_rrf(self):
        ranked_lists = {
            "vector": [{"content": "doc_a"}],
            "bm25": [{"content": "doc_b"}],
        }
        weights = {"vector": 2.0, "bm25": 1.0}
        result = reciprocal_rank_fusion(ranked_lists, k=60, weights=weights, top_k=10)
        scores = {r.content: r.rrf_score for r in result}
        assert scores["doc_a"] == pytest.approx(2.0 / 61, abs=1e-6)
        assert scores["doc_b"] == pytest.approx(1.0 / 61, abs=1e-6)

    def test_top_k_limit(self):
        ranked_lists = {
            "vector": [{"content": f"doc_{i}"} for i in range(10)],
        }
        result = reciprocal_rank_fusion(ranked_lists, k=60, top_k=3)
        assert len(result) == 3

    def test_rank_info_tracking(self):
        ranked_lists = {
            "vector": [
                {"content": "shared"},
                {"content": "v_only"},
            ],
            "bm25": [
                {"content": "b_only"},
                {"content": "shared"},
            ],
        }
        result = reciprocal_rank_fusion(ranked_lists, k=60, top_k=10)
        shared = next(r for r in result if r.content == "shared")
        assert shared.rank_info["vector"] == 1
        assert shared.rank_info["bm25"] == 2

    def test_custom_k_parameter(self):
        ranked_lists = {
            "vector": [{"content": "doc1"}],
        }
        result = reciprocal_rank_fusion(ranked_lists, k=10, top_k=10)
        assert result[0].rrf_score == pytest.approx(1.0 / 11, abs=1e-6)

    def test_skip_empty_content(self):
        ranked_lists = {
            "vector": [
                {"content": ""},
                {"content": "valid_doc"},
            ],
        }
        result = reciprocal_rank_fusion(ranked_lists, k=60, top_k=10)
        assert len(result) == 1
        assert result[0].content == "valid_doc"

    def test_document_key_field(self):
        ranked_lists = {
            "vector": [
                {"document": "via_document_field"},
            ],
        }
        result = reciprocal_rank_fusion(ranked_lists, k=60, top_k=10)
        assert len(result) == 1
        assert result[0].content == "via_document_field"


# ===== Reranker 测试 =====

class TestDashScopeReranker:
    """DashScope Reranker 测试"""

    def test_disabled_reranker(self):
        with patch("src.rag.reranker.settings") as mock_settings:
            mock_settings.RERANK_ENABLED = False
            mock_settings.RERANK_MODEL_NAME = "qwen3-rerank"
            mock_settings.RERANK_TOP_N = 3
            mock_settings.DASHSCOPE_API_KEY = "test-key"
            reranker = DashScopeReranker.__new__(DashScopeReranker)
            reranker.enabled = False
            reranker.model_name = "qwen3-rerank"
            reranker.api_key = "test-key"
            reranker.top_n = 3

            result = reranker.rerank("query", ["doc1", "doc2"])
            assert len(result) == 2
            assert result[0].content == "doc1"
            assert result[0].relevance_score == 0.0

    def test_no_api_key_disables_reranker(self):
        with patch("src.rag.reranker.settings") as mock_settings:
            mock_settings.RERANK_ENABLED = True
            mock_settings.RERANK_MODEL_NAME = "qwen3-rerank"
            mock_settings.RERANK_TOP_N = 3
            mock_settings.DASHSCOPE_API_KEY = ""
            reranker = DashScopeReranker.__new__(DashScopeReranker)
            reranker.enabled = False
            assert not reranker.enabled

    def test_empty_documents(self):
        reranker = DashScopeReranker.__new__(DashScopeReranker)
        reranker.enabled = True
        result = reranker.rerank("query", [])
        assert result == []

    @patch("requests.post")
    def test_http_rerank_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95, "document": {"text": "doc2"}},
                {"index": 0, "relevance_score": 0.80, "document": {"text": "doc1"}},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        reranker = DashScopeReranker.__new__(DashScopeReranker)
        reranker.enabled = True
        reranker.model_name = "qwen3-rerank"
        reranker.api_key = "test-key"
        reranker.top_n = 3
        reranker._use_sdk = False

        result = reranker.rerank("query", ["doc1", "doc2"], top_n=2)
        assert len(result) == 2
        assert result[0].relevance_score == 0.95
        assert result[0].content == "doc2"

    @patch("requests.post")
    def test_http_rerank_failure_fallback(self, mock_post):
        mock_post.side_effect = Exception("Network error")

        reranker = DashScopeReranker.__new__(DashScopeReranker)
        reranker.enabled = True
        reranker.model_name = "qwen3-rerank"
        reranker.api_key = "test-key"
        reranker.top_n = 3
        reranker._use_sdk = False

        result = reranker.rerank("query", ["doc1", "doc2"], top_n=2)
        assert len(result) == 2
        assert result[0].relevance_score == 0.0

    def test_fallback_passthrough(self):
        docs = ["doc1", "doc2", "doc3"]
        result = DashScopeReranker._fallback_passthrough(docs)
        assert len(result) == 3
        for i, r in enumerate(result):
            assert r.index == i
            assert r.content == docs[i]
            assert r.relevance_score == 0.0


# ===== OnlineRetriever 三级流水线测试 =====

class TestOnlineRetrieverPipeline:
    """三级检索流水线集成测试"""

    def test_disabled_rag(self):
        with patch("src.rag.online_retriever.settings") as mock_settings:
            mock_settings.RAG_ENABLED = False
            mock_settings.DASHSCOPE_API_KEY = "test"
            mock_settings.VECTOR_DB_PATH = "/tmp"
            mock_settings.RAG_VECTOR_WEIGHT = 0.5
            mock_settings.RAG_BM25_WEIGHT = 0.5
            mock_settings.RAG_RETRIEVAL_CANDIDATES = 20
            mock_settings.RRF_K = 60
            mock_settings.RERANK_TOP_N = 3
            mock_settings.RERANK_ENABLED = False
            mock_settings.RERANK_MODEL_NAME = "qwen3-rerank"
            mock_settings.RAG_CACHE_ENABLED = False
            mock_settings.RAG_CACHE_TTL = 300
            mock_settings.RAG_CACHE_MAX_SIZE = 256

            from src.rag.online_retriever import OnlineRetriever
            retriever = OnlineRetriever.__new__(OnlineRetriever)
            retriever.enabled = False
            retriever.db = None

            result = retriever.get_relevant_context("test query")
            assert result == ""

    def test_format_results(self):
        results = [
            RerankResult(
                index=0,
                relevance_score=0.95,
                content="Content A",
                metadata={"source": "wiki", "record_id": "r1"},
            ),
            RerankResult(
                index=1,
                relevance_score=0.80,
                content="Content B",
                metadata={},
            ),
        ]
        from src.rag.online_retriever import OnlineRetriever
        formatted = OnlineRetriever._format_results(results)
        assert "Content A" in formatted
        assert "Content B" in formatted
        assert "source=wiki" in formatted
        assert "record_id=r1" in formatted
        assert "score=0.9500" in formatted

    def test_rrf_to_rerank_conversion(self):
        rrf_docs = [
            RankedDocument(content="doc1", metadata={"s": "a"}, rrf_score=0.03),
            RankedDocument(content="doc2", metadata={"s": "b"}, rrf_score=0.02),
        ]
        from src.rag.online_retriever import OnlineRetriever
        result = OnlineRetriever._rrf_to_rerank_results(rrf_docs, 2)
        assert len(result) == 2
        assert result[0].relevance_score == 0.03
        assert result[0].content == "doc1"

    def test_cache_key_deterministic(self):
        from src.rag.cache import MultiLevelCache
        key1 = MultiLevelCache._make_key("test query", 3)
        key2 = MultiLevelCache._make_key("test query", 3)
        key3 = MultiLevelCache._make_key("other query", 3)
        assert key1 == key2
        assert key1 != key3


# ===== 缓存机制测试 =====

class TestCacheMechanism:
    """TTLCache 缓存测试"""

    def test_cache_hit(self):
        from cachetools import TTLCache
        cache = TTLCache(maxsize=10, ttl=300)
        cache["key1"] = "value1"
        assert cache.get("key1") == "value1"

    def test_cache_miss(self):
        from cachetools import TTLCache
        cache = TTLCache(maxsize=10, ttl=300)
        assert cache.get("nonexistent") is None

    def test_cache_max_size(self):
        from cachetools import TTLCache
        cache = TTLCache(maxsize=2, ttl=300)
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        cache["key3"] = "value3"
        assert "key1" not in cache
        assert "key3" in cache

    def test_cache_ttl_expiry(self):
        from cachetools import TTLCache
        cache = TTLCache(maxsize=10, ttl=1)
        cache["key1"] = "value1"
        time.sleep(1.1)
        assert cache.get("key1") is None


# ===== 端到端流水线模拟测试 =====

class TestEndToEndPipeline:
    """模拟完整三级检索流水线"""

    def test_full_pipeline_simulation(self):
        vector_results = [
            {"content": "木星是太阳系最大的行星", "metadata": {"source": "wiki"}},
            {"content": "土星拥有壮观的环系统", "metadata": {"source": "wiki"}},
            {"content": "火星被称为红色星球", "metadata": {"source": "wiki"}},
        ]
        bm25_results = [
            {"content": "木星的大红斑是巨型风暴", "metadata": {"source": "nasa"}},
            {"content": "木星是太阳系最大的行星", "metadata": {"source": "wiki"}},
            {"content": "金星是地球的姊妹星", "metadata": {"source": "wiki"}},
        ]

        ranked_lists = {"vector": vector_results, "bm25": bm25_results}
        weights = {"vector": 0.5, "bm25": 0.5}

        rrf_results = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k=60,
            weights=weights,
            top_k=10,
        )

        assert len(rrf_results) > 0
        shared_doc = next((r for r in rrf_results if r.content == "木星是太阳系最大的行星"), None)
        assert shared_doc is not None
        assert shared_doc.rrf_score > 0
        assert shared_doc.source == "hybrid"

        rrf_scores = [r.rrf_score for r in rrf_results]
        for i in range(len(rrf_scores) - 1):
            assert rrf_scores[i] >= rrf_scores[i + 1]

    def test_pipeline_with_rerank_simulation(self):
        vector_results = [
            {"content": "木星是太阳系最大的行星"},
            {"content": "土星拥有壮观的环系统"},
        ]
        bm25_results = [
            {"content": "木星的大红斑是巨型风暴"},
            {"content": "木星是太阳系最大的行星"},
        ]

        ranked_lists = {"vector": vector_results, "bm25": bm25_results}
        rrf_results = reciprocal_rank_fusion(ranked_lists, k=60, top_k=5)

        rerank_results = [
            RerankResult(index=0, relevance_score=0.98, content=rrf_results[0].content, metadata={}),
            RerankResult(index=1, relevance_score=0.85, content=rrf_results[1].content, metadata={}),
        ]

        assert len(rerank_results) == 2
        assert rerank_results[0].relevance_score > rerank_results[1].relevance_score
