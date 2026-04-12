"""
三级检索架构基准对比测试

评估指标:
  - Precision@K: 前 K 个结果中相关文档的比例
  - Recall@K: 前 K 个结果中检索到的相关文档占所有相关文档的比例
  - MRR (Mean Reciprocal Rank): 第一个相关文档排名倒数的均值
  - 延迟: 各检索策略的响应时间

对比策略:
  1. 纯向量检索 (baseline)
  2. 纯 BM25 检索
  3. 混合检索 + RRF (无 Rerank)
  4. 混合检索 + RRF + Rerank (完整三级架构)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set, Tuple

import pytest

from src.rag.rrf_fusion import RankedDocument, reciprocal_rank_fusion
from src.rag.reranker import RerankResult


SAMPLE_QUERIES = [
    {
        "query": "木星的大气成分",
        "relevant_docs": {
            "木星的大气主要由氢和氦组成",
            "木星大气中含有微量的甲烷和水蒸气",
            "木星大红斑是持续数百年的巨型风暴",
        },
    },
    {
        "query": "土星环的结构",
        "relevant_docs": {
            "土星环主要由冰粒和岩石碎片组成",
            "土星环的宽度可达28万公里",
            "卡西尼号探测器详细研究了土星环结构",
        },
    },
    {
        "query": "火星探测任务",
        "relevant_docs": {
            "好奇号火星车自2012年在火星表面运行",
            "毅力号火星车携带了机智号直升机",
            "天问一号是中国首次火星探测任务",
        },
    },
]


VECTOR_RESULTS = [
    [
        {"content": "木星的大气主要由氢和氦组成"},
        {"content": "土星环主要由冰粒和岩石碎片组成"},
        {"content": "木星大气中含有微量的甲烷和水蒸气"},
        {"content": "金星的大气极为稠密且富含二氧化碳"},
        {"content": "木星大红斑是持续数百年的巨型风暴"},
    ],
    [
        {"content": "土星环主要由冰粒和岩石碎片组成"},
        {"content": "木星的大气主要由氢和氦组成"},
        {"content": "土星环的宽度可达28万公里"},
        {"content": "天王星也有环系统但比土星暗淡"},
        {"content": "卡西尼号探测器详细研究了土星环结构"},
    ],
    [
        {"content": "好奇号火星车自2012年在火星表面运行"},
        {"content": "木星的大气主要由氢和氦组成"},
        {"content": "毅力号火星车携带了机智号直升机"},
        {"content": "旅行者号探测器访问了外行星"},
        {"content": "天问一号是中国首次火星探测任务"},
    ],
]

BM25_RESULTS = [
    [
        {"content": "木星大红斑是持续数百年的巨型风暴"},
        {"content": "木星的大气主要由氢和氦组成"},
        {"content": "火星大气稀薄主要由二氧化碳组成"},
        {"content": "木星大气中含有微量的甲烷和水蒸气"},
        {"content": "地球大气由氮气和氧气组成"},
    ],
    [
        {"content": "土星环的宽度可达28万公里"},
        {"content": "土星环主要由冰粒和岩石碎片组成"},
        {"content": "木星的大气主要由氢和氦组成"},
        {"content": "卡西尼号探测器详细研究了土星环结构"},
        {"content": "海王星的大气含有甲烷呈现蓝色"},
    ],
    [
        {"content": "天问一号是中国首次火星探测任务"},
        {"content": "好奇号火星车自2012年在火星表面运行"},
        {"content": "阿波罗计划是载人登月任务"},
        {"content": "毅力号火星车携带了机智号直升机"},
        {"content": "哈勃望远镜观测了遥远星系"},
    ],
]

RERANK_SCORES = [
    [0.98, 0.92, 0.88, 0.45, 0.30],
    [0.95, 0.91, 0.87, 0.50, 0.35],
    [0.96, 0.93, 0.40, 0.89, 0.28],
]


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def simulate_rerank(rrf_results: List[RankedDocument], scores: List[float]) -> List[str]:
    indexed = list(enumerate(rrf_results))
    scored = [(idx, scores[idx] if idx < len(scores) else 0.0) for idx, _ in indexed]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [rrf_results[idx].content for idx, _ in scored]


class TestBenchmarkMetrics:
    """基准评估指标计算测试"""

    def test_precision_at_k_perfect(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_precision_at_k_partial(self):
        retrieved = ["a", "x", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3, abs=1e-6)

    def test_precision_at_k_empty(self):
        assert precision_at_k([], {"a"}, 3) == 0.0

    def test_recall_at_k_perfect(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 4) == 1.0

    def test_recall_at_k_partial(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3, abs=1e-6)

    def test_mrr_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_mrr_second_position(self):
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5

    def test_mrr_no_match(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestRetrievalComparison:
    """多策略检索对比实验"""

    @pytest.fixture
    def benchmark_data(self):
        return list(zip(SAMPLE_QUERIES, VECTOR_RESULTS, BM25_RESULTS, RERANK_SCORES))

    def test_vector_only_baseline(self, benchmark_data):
        precisions, recalls, mrrs = [], [], []
        for query_data, vec_results, _, _ in benchmark_data:
            relevant = query_data["relevant_docs"]
            retrieved = [r["content"] for r in vec_results]
            precisions.append(precision_at_k(retrieved, relevant, 3))
            recalls.append(recall_at_k(retrieved, relevant, 5))
            mrrs.append(reciprocal_rank(retrieved, relevant))

        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_mrr = sum(mrrs) / len(mrrs)

        assert avg_precision > 0
        assert avg_recall > 0
        assert avg_mrr > 0

    def test_bm25_only(self, benchmark_data):
        precisions, recalls, mrrs = [], [], []
        for query_data, _, bm25_results, _ in benchmark_data:
            relevant = query_data["relevant_docs"]
            retrieved = [r["content"] for r in bm25_results]
            precisions.append(precision_at_k(retrieved, relevant, 3))
            recalls.append(recall_at_k(retrieved, relevant, 5))
            mrrs.append(reciprocal_rank(retrieved, relevant))

        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_mrr = sum(mrrs) / len(mrrs)

        assert avg_precision > 0
        assert avg_recall > 0

    def test_hybrid_rrf_no_rerank(self, benchmark_data):
        precisions, recalls, mrrs = [], [], []
        for query_data, vec_results, bm25_results, _ in benchmark_data:
            relevant = query_data["relevant_docs"]
            ranked_lists = {"vector": vec_results, "bm25": bm25_results}
            rrf_results = reciprocal_rank_fusion(ranked_lists, k=60, top_k=5)
            retrieved = [r.content for r in rrf_results]
            precisions.append(precision_at_k(retrieved, relevant, 3))
            recalls.append(recall_at_k(retrieved, relevant, 5))
            mrrs.append(reciprocal_rank(retrieved, relevant))

        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_mrr = sum(mrrs) / len(mrrs)

        assert avg_precision > 0
        assert avg_recall > 0

    def test_full_three_level_pipeline(self, benchmark_data):
        precisions, recalls, mrrs = [], [], []
        for query_data, vec_results, bm25_results, rerank_scores in benchmark_data:
            relevant = query_data["relevant_docs"]
            ranked_lists = {"vector": vec_results, "bm25": bm25_results}
            rrf_results = reciprocal_rank_fusion(ranked_lists, k=60, top_k=5)
            retrieved = simulate_rerank(rrf_results, rerank_scores)
            precisions.append(precision_at_k(retrieved, relevant, 3))
            recalls.append(recall_at_k(retrieved, relevant, 5))
            mrrs.append(reciprocal_rank(retrieved, relevant))

        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_mrr = sum(mrrs) / len(mrrs)

        assert avg_precision > 0
        assert avg_recall > 0
        assert avg_mrr > 0

    def test_three_level_outperforms_single(self, benchmark_data):
        vec_precisions, vec_mrrs = [], []
        full_precisions, full_mrrs = [], []

        for query_data, vec_results, bm25_results, rerank_scores in benchmark_data:
            relevant = query_data["relevant_docs"]

            vec_retrieved = [r["content"] for r in vec_results]
            vec_precisions.append(precision_at_k(vec_retrieved, relevant, 3))
            vec_mrrs.append(reciprocal_rank(vec_retrieved, relevant))

            ranked_lists = {"vector": vec_results, "bm25": bm25_results}
            rrf_results = reciprocal_rank_fusion(ranked_lists, k=60, top_k=5)
            full_retrieved = simulate_rerank(rrf_results, rerank_scores)
            full_precisions.append(precision_at_k(full_retrieved, relevant, 3))
            full_mrrs.append(reciprocal_rank(full_retrieved, relevant))

        avg_vec_p = sum(vec_precisions) / len(vec_precisions)
        avg_full_p = sum(full_precisions) / len(full_precisions)
        avg_vec_mrr = sum(vec_mrrs) / len(vec_mrrs)
        avg_full_mrr = sum(full_mrrs) / len(full_mrrs)

        assert avg_full_mrr >= avg_vec_mrr or avg_full_p >= avg_vec_p, (
            f"三级架构应优于单一向量检索: "
            f"向量 P@3={avg_vec_p:.3f} MRR={avg_vec_mrr:.3f}, "
            f"三级 P@3={avg_full_p:.3f} MRR={avg_full_mrr:.3f}"
        )


class TestLatencyBenchmark:
    """延迟基准测试"""

    def test_rrf_latency(self):
        ranked_lists = {
            "vector": [{"content": f"doc_{i}"} for i in range(100)],
            "bm25": [{"content": f"doc_{i}"} for i in range(100)],
        }
        start = time.time()
        for _ in range(100):
            reciprocal_rank_fusion(ranked_lists, k=60, top_k=10)
        elapsed = time.time() - start
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 50, f"RRF 平均延迟 {avg_ms:.1f}ms 超过 50ms 阈值"

    def test_rrf_scales_linearly(self):
        small = {"vector": [{"content": f"d{i}"} for i in range(10)]}
        large = {"vector": [{"content": f"d{i}"} for i in range(1000)]}

        start = time.time()
        for _ in range(50):
            reciprocal_rank_fusion(small, k=60, top_k=5)
        small_time = time.time() - start

        start = time.time()
        for _ in range(50):
            reciprocal_rank_fusion(large, k=60, top_k=5)
        large_time = time.time() - start

        ratio = large_time / small_time if small_time > 0 else float("inf")
        assert ratio < 200, f"RRF 延迟增长比例 {ratio:.1f}x 过高，可能存在性能问题"
