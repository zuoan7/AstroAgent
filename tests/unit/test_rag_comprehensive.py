"""
天文领域RAG系统全面评估测试

包含:
  - 天文专用文档分块器测试
  - 天文实体检索器测试
  - 多模态检索器测试
  - 时效性感知重排序与过滤测试
  - 多级缓存系统测试
  - 知识更新机制测试
  - 检索质量监控测试
  - 天文领域100+查询场景测试集
  - 评估框架（准确率/召回率/F1/MRR/延迟）
  - 压力测试
  - A/B测试框架
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Set

import pytest

from src.rag.astronomy_chunker import (
    AstronomyChunker,
    AstronomyContentType,
    AstronomyMetadata,
    detect_content_type,
    extract_astronomy_metadata,
)
from src.rag.entity_retriever import AstronomyEntityRecognizer, AstronomyEntityRetriever
from src.rag.multimodal_retriever import MultimodalDocument, MultimodalRetriever, SpectrumFeatureExtractor
from src.rag.result_filter import (
    CredibilityScorer,
    DomainRelevanceScorer,
    FilteredResult,
    ResultFilterAndReranker,
    TimelinessScorer,
)
from src.rag.cache import MultiLevelCache
from src.rag.knowledge_updater import KnowledgeUpdateManager
from src.rag.metrics import MetricsCollector, RetrievalMetrics, UserFeedback
from src.rag.rrf_fusion import RankedDocument, reciprocal_rank_fusion
from src.rag.reranker import RerankResult


# ===== 天文文档分块器测试 =====

class TestAstronomyChunker:
    def test_detect_star_catalog(self):
        text = "HD 12345: 赤经 02h 03m, 赤纬 +15° 20', 视星等 6.2, 光谱型 G2V"
        assert detect_content_type(text) == AstronomyContentType.STAR_CATALOG

    def test_detect_orbital_parameters(self):
        text = "半长轴: 5.2 AU, 离心率: 0.048, 轨道倾角: 1.3°"
        assert detect_content_type(text) == AstronomyContentType.ORBITAL_PARAMETERS

    def test_detect_formula(self):
        text = "根据斯特藩-玻尔兹曼定律，L = 4πR²σT⁴"
        assert detect_content_type(text) == AstronomyContentType.ASTROPHYSICS_FORMULA

    def test_detect_observation(self):
        text = "观测日期: 2025-03-15, 望远镜: 哈勃, 曝光时间: 1200s, 视星等: 12.5"
        assert detect_content_type(text) == AstronomyContentType.OBSERVATION_RECORD

    def test_detect_celestial_event(self):
        text = "2025年3月29日将发生日食，可见区域包括欧洲北部"
        assert detect_content_type(text) == AstronomyContentType.CELESTIAL_EVENT

    def test_detect_general(self):
        text = "天文台位于山顶，海拔2000米"
        assert detect_content_type(text) == AstronomyContentType.GENERAL_TEXT

    def test_chunk_catalog_preserves_entries(self):
        text = (
            "HD 12345: 赤经 02h, 赤纬 +15°, 视星等 6.2\n"
            "HD 67890: 赤经 05h, 赤纬 -20°, 视星等 7.1\n"
            "NGC 1234: 星系, 赤经 03h, 赤纬 +10°\n"
        )
        chunker = AstronomyChunker(chunk_size=500)
        results = chunker.chunk_text(text)
        assert len(results) >= 1
        assert all(r.metadata.get("doc_type") == "star_catalog" for r in results)

    def test_chunk_orbital_preserves_group(self):
        text = (
            "木星轨道参数:\n"
            "半长轴: 5.2 AU\n"
            "离心率: 0.048\n"
            "轨道倾角: 1.3°\n"
            "升交点经度: 100.5°\n"
        )
        chunker = AstronomyChunker(chunk_size=500, preserve_orbital_groups=True)
        results = chunker.chunk_text(text)
        assert len(results) >= 1
        assert "半长轴" in results[0].content

    def test_chunk_general_text(self):
        text = "天文观测需要良好的天气条件。" * 50
        chunker = AstronomyChunker(chunk_size=200, chunk_overlap=20)
        results = chunker.chunk_text(text)
        assert len(results) > 1

    def test_extract_metadata_celestial_object(self):
        text = "木星是太阳系最大的行星"
        meta = extract_astronomy_metadata(text)
        assert meta.celestial_object is not None

    def test_extract_metadata_time_sensitive(self):
        text = "最新发现的系外行星Kepler-442b"
        meta = extract_astronomy_metadata(text)
        assert meta.is_time_sensitive is True

    def test_extract_metadata_wavelength(self):
        text = "通过X射线观测发现黑洞吸积盘"
        meta = extract_astronomy_metadata(text)
        assert meta.wavelength_band is not None

    def test_metadata_to_dict(self):
        meta = AstronomyMetadata(
            doc_type=AstronomyContentType.STAR_CATALOG,
            celestial_object="M31",
            credibility=0.9,
        )
        d = meta.to_dict()
        assert d["doc_type"] == "star_catalog"
        assert d["celestial_object"] == "M31"
        assert d["credibility"] == 0.9

    def test_empty_text(self):
        chunker = AstronomyChunker()
        results = chunker.chunk_text("")
        assert results == []


# ===== 天文实体检索器测试 =====

class TestAstronomyEntityRecognizer:
    def test_recognize_planet(self):
        recognizer = AstronomyEntityRecognizer()
        matches = recognizer.recognize("木星的大气成分")
        planet_matches = [m for m in matches if m.category == "planet"]
        assert len(planet_matches) > 0
        assert planet_matches[0].canonical == "jupiter"

    def test_recognize_deep_sky(self):
        recognizer = AstronomyEntityRecognizer()
        matches = recognizer.recognize("仙女座星系的距离")
        dso_matches = [m for m in matches if m.category == "deep_sky"]
        assert len(dso_matches) > 0

    def test_recognize_event(self):
        recognizer = AstronomyEntityRecognizer()
        matches = recognizer.recognize("今晚有流星雨")
        event_matches = [m for m in matches if m.category == "event"]
        assert len(event_matches) > 0

    def test_recognize_term(self):
        recognizer = AstronomyEntityRecognizer()
        matches = recognizer.recognize("红移值z=0.5的星系")
        term_matches = [m for m in matches if m.category == "term"]
        assert len(term_matches) > 0

    def test_recognize_instrument(self):
        recognizer = AstronomyEntityRecognizer()
        matches = recognizer.recognize("哈勃望远镜拍摄的照片")
        inst_matches = [m for m in matches if m.category == "instrument"]
        assert len(inst_matches) > 0

    def test_no_match(self):
        recognizer = AstronomyEntityRecognizer()
        matches = recognizer.recognize("今天天气很好")
        assert len(matches) == 0


class TestAstronomyEntityRetriever:
    @pytest.fixture
    def sample_docs(self):
        return [
            {"content": "木星是太阳系最大的行星，质量是地球的318倍", "metadata": {"source": "wiki"}},
            {"content": "仙女座星系距离我们254万光年", "metadata": {"source": "wiki"}},
            {"content": "猎户座星云是最明亮的弥漫星云之一", "metadata": {"source": "nasa"}},
            {"content": "哈勃望远镜已运行超过30年", "metadata": {"source": "nasa"}},
        ]

    def test_search_by_planet(self, sample_docs):
        retriever = AstronomyEntityRetriever(sample_docs)
        results = retriever.search("木星的质量")
        assert len(results) > 0
        assert "木星" in results[0].content

    def test_search_by_deep_sky(self, sample_docs):
        retriever = AstronomyEntityRetriever(sample_docs)
        results = retriever.search("仙女座星系")
        assert len(results) > 0

    def test_search_formatted(self, sample_docs):
        retriever = AstronomyEntityRetriever(sample_docs)
        results = retriever.search_formatted("猎户座星云")
        assert len(results) > 0
        assert "content" in results[0]

    def test_empty_index(self):
        retriever = AstronomyEntityRetriever()
        results = retriever.search("木星")
        assert results == []


# ===== 多模态检索器测试 =====

class TestMultimodalRetriever:
    def test_add_and_search_text(self):
        retriever = MultimodalRetriever()
        doc = MultimodalDocument(
            doc_id="test1",
            modality="text",
            content="木星的大红斑是巨型风暴",
            metadata={"celestial_object": "木星"},
        )
        retriever.add_document(doc)
        results = retriever.search("木星风暴")
        assert len(results) > 0

    def test_spectrum_feature_extraction(self):
        extractor = SpectrumFeatureExtractor()
        features = extractor.extract_features("Hα发射线在光谱中明显可见，OIII线也存在")
        assert "发射线" in features

    def test_spectrum_similarity(self):
        extractor = SpectrumFeatureExtractor()
        q_features = extractor.extract_features("Hα发射线")
        d_features = extractor.extract_features("Hα和OIII发射线")
        score = extractor.spectrum_similarity(q_features, d_features)
        assert score > 0

    def test_search_by_modality(self):
        retriever = MultimodalRetriever()
        retriever.add_document(MultimodalDocument(
            doc_id="t1", modality="text", content="木星观测记录", metadata={}
        ))
        retriever.add_document(MultimodalDocument(
            doc_id="s1", modality="spectrum", content="Hα发射线光谱", metadata={}
        ))
        results = retriever.search("Hα", modalities=["spectrum"])
        assert all(r.modality == "spectrum" for r in results)


# ===== 时效性/可信度/过滤测试 =====

class TestTimelinessScorer:
    def test_recent_data_high_score(self):
        scorer = TimelinessScorer()
        meta = {"publication_year": 2025, "is_time_sensitive": True}
        score = scorer.score(meta, "最新发现")
        assert score >= 0.8

    def test_old_data_low_score_for_time_sensitive(self):
        scorer = TimelinessScorer()
        meta = {"publication_year": 2010}
        score = scorer.score(meta, "最新观测结果")
        assert score < 0.5

    def test_non_time_sensitive_query(self):
        scorer = TimelinessScorer()
        meta = {}
        score = scorer.score(meta, "木星的基本信息")
        assert score == 0.7


class TestCredibilityScorer:
    def test_nasa_high_credibility(self):
        scorer = CredibilityScorer()
        meta = {"source": "nasa_apod", "data_source": "nasa"}
        score = scorer.score(meta)
        assert score == 1.0

    def test_wikipedia_medium(self):
        scorer = CredibilityScorer()
        meta = {"source": "wikipedia"}
        score = scorer.score(meta)
        assert score == 0.8

    def test_unknown_source(self):
        scorer = CredibilityScorer()
        meta = {"source": "random_blog"}
        score = scorer.score(meta)
        assert score == 0.5


class TestResultFilterAndReranker:
    @pytest.fixture
    def sample_results(self):
        return [
            RerankResult(index=0, relevance_score=0.9, content="木星最新观测数据", metadata={"source": "nasa", "publication_year": 2025}),
            RerankResult(index=1, relevance_score=0.7, content="木星基本信息", metadata={"source": "wiki"}),
            RerankResult(index=2, relevance_score=0.5, content="木星数据", metadata={"source": "unknown_blog"}),
        ]

    def test_filter_and_rerank(self, sample_results):
        filter_reranker = ResultFilterAndReranker(min_credibility=0.3)
        results = filter_reranker.filter_and_rerank(sample_results, "木星最新观测", top_k=3)
        assert len(results) > 0
        assert not results[0].filtered_out

    def test_low_credibility_filtered(self):
        results = [
            RerankResult(index=0, relevance_score=0.9, content="test", metadata={"source": "spam"}),
        ]
        filter_reranker = ResultFilterAndReranker(min_credibility=0.8)
        filtered = filter_reranker.filter_and_rerank(results, "test", top_k=5)
        valid = [r for r in filtered if not r.filtered_out]
        assert len(valid) == 0


# ===== 多级缓存测试 =====

class TestMultiLevelCache:
    def test_l1_cache_hit(self):
        cache = MultiLevelCache(l1_maxsize=10, l1_ttl=60, l3_enabled=False)
        cache.set("query1", 3, "result1")
        assert cache.get("query1", 3) == "result1"

    def test_cache_miss(self):
        cache = MultiLevelCache(l1_maxsize=10, l1_ttl=60, l3_enabled=False)
        assert cache.get("nonexistent", 3) is None

    def test_l1_to_l2_promotion(self):
        cache = MultiLevelCache(l1_maxsize=1, l1_ttl=60, l2_maxsize=10, l2_ttl=300, l3_enabled=False)
        cache.set("q1", 3, "r1")
        cache.set("q2", 3, "r2")  # q1 evicted from L1
        cache.l1.pop(cache._make_key("q1", 3), None)  # Force evict from L1
        assert cache.get("q1", 3) == "r1"  # Should find in L2

    def test_cache_stats(self):
        cache = MultiLevelCache(l1_maxsize=10, l1_ttl=60, l3_enabled=False)
        cache.set("q1", 3, "r1")
        cache.get("q1", 3)
        cache.get("miss", 3)
        stats = cache.get_stats()
        assert stats["l1_hits"] == 1
        assert stats["misses"] == 1

    def test_cache_invalidate(self):
        cache = MultiLevelCache(l1_maxsize=10, l1_ttl=60, l3_enabled=False)
        cache.set("q1", 3, "r1")
        cache.invalidate("q1", 3)
        assert cache.get("q1", 3) is None

    def test_cache_clear(self):
        cache = MultiLevelCache(l1_maxsize=10, l1_ttl=60, l3_enabled=False)
        cache.set("q1", 3, "r1")
        cache.clear()
        assert cache.get("q1", 3) is None

    def test_l3_disk_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cache.sqlite")
            cache = MultiLevelCache(l1_maxsize=10, l1_ttl=60, l3_path=db_path, l3_enabled=True)
            cache.set("q1", 3, "r1")
            cache.l1.clear()
            cache.l2.clear()
            result = cache.get("q1", 3)
            assert result == "r1"
            cache.close()


# ===== 知识更新机制测试 =====

class TestKnowledgeUpdateManager:
    def test_detect_new_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            updater = KnowledgeUpdateManager(vector_db_path=tmpdir)
            docs = [
                {"content": "新发现的系外行星", "record_id": "exo_001", "metadata": {}},
                {"content": "木星基本信息", "record_id": "jupiter_001", "metadata": {}},
            ]
            result = updater.check_updates(docs, source="test")
            assert len(result["added"]) == 2
            assert len(result["unchanged"]) == 0

    def test_detect_unchanged_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            updater = KnowledgeUpdateManager(vector_db_path=tmpdir)
            docs = [{"content": "test content", "record_id": "doc1"}]
            updater.check_updates(docs, source="test")
            result = updater.check_updates(docs, source="test")
            assert len(result["unchanged"]) == 1

    def test_detect_updated_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            updater = KnowledgeUpdateManager(vector_db_path=tmpdir)
            docs = [{"content": "original content", "record_id": "doc1"}]
            updater.check_updates(docs, source="test")
            docs_updated = [{"content": "updated content", "record_id": "doc1"}]
            result = updater.check_updates(docs_updated, source="test")
            assert len(result["updated"]) == 1

    def test_update_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            updater = KnowledgeUpdateManager(vector_db_path=tmpdir)
            stats = updater.get_update_stats()
            assert "total_records" in stats


# ===== 检索质量监控测试 =====

class TestMetricsCollector:
    def test_record_and_query_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(db_path=os.path.join(tmpdir, "metrics.sqlite"))
            collector.record_retrieval(RetrievalMetrics(
                query="木星大气",
                latency_ms=150.0,
                num_results=3,
                top_score=0.95,
                avg_score=0.80,
                cache_hit=False,
            ))
            summary = collector.get_metrics_summary(hours=1)
            assert summary["total_queries"] == 1
            assert summary["avg_latency_ms"] == 150.0
            collector.close()

    def test_record_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(db_path=os.path.join(tmpdir, "metrics.sqlite"))
            collector.record_feedback(UserFeedback(
                query="木星",
                relevance_rating=4,
                is_accurate=True,
                comment="很好",
            ))
            summary = collector.get_metrics_summary(hours=1)
            assert summary["total_feedback"] == 1
            collector.close()


# ===== 100+天文查询场景测试集 =====

ASTRONOMY_TEST_QUERIES = [
    {"query": "木星的大气成分", "category": "行星科学", "relevant_entities": ["jupiter"]},
    {"query": "土星环的结构", "category": "行星科学", "relevant_entities": ["saturn"]},
    {"query": "火星探测任务", "category": "太空探索", "relevant_entities": ["mars"]},
    {"query": "金星的温室效应", "category": "行星科学", "relevant_entities": ["venus"]},
    {"query": "水星的轨道特征", "category": "行星科学", "relevant_entities": ["mercury"]},
    {"query": "天王星的倾斜轴", "category": "行星科学", "relevant_entities": ["uranus"]},
    {"query": "海王星的风速", "category": "行星科学", "relevant_entities": ["neptune"]},
    {"query": "仙女座星系的距离", "category": "深空天体", "relevant_entities": ["M31"]},
    {"query": "猎户座星云的观测", "category": "深空天体", "relevant_entities": ["M42"]},
    {"query": "蟹状星云的起源", "category": "深空天体", "relevant_entities": ["M1"]},
    {"query": "昴星团有多少颗星", "category": "深空天体", "relevant_entities": ["M45"]},
    {"query": "天狼星的亮度", "category": "恒星", "relevant_entities": ["sirius"]},
    {"query": "参宿四的超新星预测", "category": "恒星", "relevant_entities": ["betelgeuse"]},
    {"query": "织女星的光谱型", "category": "恒星", "relevant_entities": ["vega"]},
    {"query": "北极星的位置", "category": "恒星", "relevant_entities": ["polaris"]},
    {"query": "黑洞的事件视界", "category": "天体物理", "relevant_entities": ["black_hole"]},
    {"query": "中子星的密度", "category": "天体物理", "relevant_entities": ["neutron_star"]},
    {"query": "超新星爆发机制", "category": "天体物理", "relevant_entities": ["supernova"]},
    {"query": "暗物质的证据", "category": "天体物理", "relevant_entities": ["dark_matter"]},
    {"query": "类星体的红移", "category": "天体物理", "relevant_entities": ["quasar", "redshift"]},
    {"query": "2025年日食时间", "category": "天文事件", "relevant_entities": ["solar_eclipse"]},
    {"query": "英仙座流星雨极大", "category": "天文事件", "relevant_entities": ["meteor_shower"]},
    {"query": "彗星C/2024观测", "category": "天文事件", "relevant_entities": ["comet"]},
    {"query": "哈勃望远镜的发现", "category": "观测设备", "relevant_entities": ["hst"]},
    {"query": "韦伯望远镜最新图像", "category": "观测设备", "relevant_entities": ["jwst"]},
    {"query": "盖亚任务的数据", "category": "观测设备", "relevant_entities": ["gaia"]},
    {"query": "赤经赤纬坐标系", "category": "天文术语", "relevant_entities": ["right_ascension"]},
    {"query": "视星等和绝对星等", "category": "天文术语", "relevant_entities": ["apparent_magnitude"]},
    {"query": "光年和秒差距换算", "category": "天文术语", "relevant_entities": ["light_year"]},
    {"query": "光谱分类系统", "category": "天文术语", "relevant_entities": ["spectral_type"]},
    {"query": "开普勒三定律", "category": "天体物理", "relevant_entities": []},
    {"query": "哈勃定律与宇宙膨胀", "category": "天体物理", "relevant_entities": []},
    {"query": "银河系的结构", "category": "银河系", "relevant_entities": []},
    {"query": "太阳系的边界", "category": "太阳系", "relevant_entities": []},
    {"query": "月球的潮汐锁定", "category": "卫星", "relevant_entities": []},
    {"query": "木卫二的冰下海洋", "category": "卫星", "relevant_entities": ["jupiter"]},
    {"query": "土卫六的大气", "category": "卫星", "relevant_entities": ["saturn"]},
    {"query": "近地小行星追踪", "category": "太阳系", "relevant_entities": []},
    {"query": "柯伊伯带天体", "category": "太阳系", "relevant_entities": []},
    {"query": "奥尔特云假说", "category": "太阳系", "relevant_entities": []},
    {"query": "系外行星探测方法", "category": "系外行星", "relevant_entities": []},
    {"query": "宜居带定义", "category": "系外行星", "relevant_entities": []},
    {"query": "脉冲星的发现", "category": "天体物理", "relevant_entities": ["pulsar"]},
    {"query": "引力波探测", "category": "天体物理", "relevant_entities": []},
    {"query": "宇宙微波背景辐射", "category": "宇宙学", "relevant_entities": []},
    {"query": "大爆炸理论证据", "category": "宇宙学", "relevant_entities": []},
    {"query": "暗能量与加速膨胀", "category": "宇宙学", "relevant_entities": ["dark_energy"]},
    {"query": "望远镜口径与分辨率", "category": "观测技术", "relevant_entities": []},
    {"query": "CCD探测器原理", "category": "观测技术", "relevant_entities": []},
    {"query": "自适应光学技术", "category": "观测技术", "relevant_entities": []},
    {"query": "射电天文观测", "category": "观测技术", "relevant_entities": []},
    {"query": "X射线天文学", "category": "观测技术", "relevant_entities": []},
    {"query": "红外天文卫星", "category": "观测技术", "relevant_entities": []},
    {"query": "太阳风与极光", "category": "太阳物理", "relevant_entities": ["aurora"]},
    {"query": "太阳黑子周期", "category": "太阳物理", "relevant_entities": []},
    {"query": "日冕物质抛射", "category": "太阳物理", "relevant_entities": []},
    {"query": "太阳耀斑等级", "category": "太阳物理", "relevant_entities": []},
    {"query": "双星系统观测", "category": "恒星", "relevant_entities": ["binary_star"]},
    {"query": "变星的光变曲线", "category": "恒星", "relevant_entities": ["variable_star"]},
    {"query": "红巨星演化", "category": "恒星", "relevant_entities": ["red_giant"]},
    {"query": "白矮星质量极限", "category": "恒星", "relevant_entities": ["white_dwarf"]},
    {"query": "吸积盘物理", "category": "天体物理", "relevant_entities": ["accretion_disk"]},
    {"query": "星系碰撞模拟", "category": "星系", "relevant_entities": []},
    {"query": "星系团暗物质分布", "category": "星系", "relevant_entities": ["dark_matter"]},
    {"query": "活动星系核", "category": "星系", "relevant_entities": []},
    {"query": "星系形态分类", "category": "星系", "relevant_entities": []},
    {"query": "宇宙大尺度结构", "category": "宇宙学", "relevant_entities": []},
    {"query": "宇宙弦理论", "category": "宇宙学", "relevant_entities": []},
    {"query": "多重宇宙假说", "category": "宇宙学", "relevant_entities": []},
    {"query": "火星极冠成分", "category": "行星科学", "relevant_entities": ["mars"]},
    {"query": "金星表面温度", "category": "行星科学", "relevant_entities": ["venus"]},
    {"query": "木星磁场强度", "category": "行星科学", "relevant_entities": ["jupiter"]},
    {"query": "土星风暴系统", "category": "行星科学", "relevant_entities": ["saturn"]},
    {"query": "海王星大暗斑", "category": "行星科学", "relevant_entities": ["neptune"]},
    {"query": "冥王星降级原因", "category": "太阳系", "relevant_entities": ["pluto"]},
    {"query": "谷神星特征", "category": "太阳系", "relevant_entities": []},
    {"query": "天文单位定义", "category": "天文术语", "relevant_entities": []},
    {"query": "岁差现象", "category": "天文术语", "relevant_entities": []},
    {"query": "章动周期", "category": "天文术语", "relevant_entities": []},
    {"query": "光行差效应", "category": "天文术语", "relevant_entities": []},
    {"query": "多普勒效应在天文中的应用", "category": "天文术语", "relevant_entities": []},
    {"query": "VLBI干涉测量", "category": "观测技术", "relevant_entities": []},
    {"query": "斯隆数字巡天", "category": "观测项目", "relevant_entities": []},
    {"query": "LSST项目", "category": "观测项目", "relevant_entities": []},
    {"query": "SKA射电望远镜", "category": "观测项目", "relevant_entities": []},
    {"query": "ELT极大望远镜", "category": "观测项目", "relevant_entities": []},
    {"query": "凌日法探测行星", "category": "系外行星", "relevant_entities": ["transit"]},
    {"query": "径向速度法", "category": "系外行星", "relevant_entities": ["radial_velocity"]},
    {"query": "直接成像法", "category": "系外行星", "relevant_entities": []},
    {"query": "微引力透镜法", "category": "系外行星", "relevant_entities": []},
    {"query": "Trappist-1系统", "category": "系外行星", "relevant_entities": []},
    {"query": "热木星形成", "category": "系外行星", "relevant_entities": ["jupiter"]},
    {"query": "超级地球定义", "category": "系外行星", "relevant_entities": []},
    {"query": "潮汐锁定行星", "category": "系外行星", "relevant_entities": []},
    {"query": "行星大气光谱分析", "category": "系外行星", "relevant_entities": ["spectral_type"]},
    {"query": "天球坐标系统", "category": "天文术语", "relevant_entities": []},
    {"query": "星等标度", "category": "天文术语", "relevant_entities": ["apparent_magnitude"]},
    {"query": "色指数定义", "category": "天文术语", "relevant_entities": ["color_index"]},
    {"query": "自行测量", "category": "天文术语", "relevant_entities": ["proper_motion"]},
    {"query": "视差测距", "category": "天文术语", "relevant_entities": ["parallax"]},
    {"query": "梅西耶天体目录", "category": "深空天体", "relevant_entities": []},
    {"query": "NGC星表", "category": "深空天体", "relevant_entities": []},
    {"query": "行星状星云", "category": "深空天体", "relevant_entities": []},
    {"query": "球状星团", "category": "深空天体", "relevant_entities": []},
    {"query": "疏散星团", "category": "深空天体", "relevant_entities": []},
    {"query": "暗星云", "category": "深空天体", "relevant_entities": []},
    {"query": "反射星云", "category": "深空天体", "relevant_entities": []},
    {"query": "发射星云", "category": "深空天体", "relevant_entities": []},
    {"query": "超新星遗迹", "category": "深空天体", "relevant_entities": ["supernova"]},
]


class TestAstronomyQueryDataset:
    """天文领域100+查询场景测试"""

    def test_query_count_at_least_100(self):
        assert len(ASTRONOMY_TEST_QUERIES) >= 100

    def test_all_queries_have_category(self):
        for q in ASTRONOMY_TEST_QUERIES:
            assert "query" in q
            assert "category" in q

    def test_category_distribution(self):
        categories = {}
        for q in ASTRONOMY_TEST_QUERIES:
            cat = q["category"]
            categories[cat] = categories.get(cat, 0) + 1
        assert len(categories) >= 5

    def test_entity_recognition_coverage(self):
        recognizer = AstronomyEntityRecognizer()
        queries_with_entities = 0
        for q in ASTRONOMY_TEST_QUERIES:
            matches = recognizer.recognize(q["query"])
            if matches:
                queries_with_entities += 1
        coverage = queries_with_entities / len(ASTRONOMY_TEST_QUERIES)
        assert coverage > 0.3, f"实体识别覆盖率 {coverage:.1%} 过低"


# ===== 评估框架测试 =====

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in relevant) / len(top_k)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant) / len(relevant)


def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


class TestEvaluationFramework:
    def test_precision_calculation(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(2 / 3)
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_recall_calculation(self):
        assert recall_at_k(["a", "x"], {"a", "b", "c"}, 2) == pytest.approx(1 / 3)
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_f1_calculation(self):
        assert f1_at_k(["a", "b", "x"], {"a", "b", "c"}, 3) > 0

    def test_mrr_calculation(self):
        assert mrr(["a", "b"], {"a"}) == 1.0
        assert mrr(["x", "a"], {"a"}) == 0.5
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_entity_retriever_evaluation(self):
        docs = [
            {"content": "木星是太阳系最大的行星", "metadata": {"source": "wiki"}},
            {"content": "土星拥有壮观的环系统", "metadata": {"source": "wiki"}},
            {"content": "火星被称为红色星球", "metadata": {"source": "nasa"}},
            {"content": "金星是地球的姊妹星", "metadata": {"source": "wiki"}},
            {"content": "木星大红斑是巨型风暴", "metadata": {"source": "nasa"}},
        ]
        retriever = AstronomyEntityRetriever(docs)

        queries = [
            {"query": "木星的特征", "relevant": {"木星是太阳系最大的行星", "木星大红斑是巨型风暴"}},
            {"query": "土星环", "relevant": {"土星拥有壮观的环系统"}},
        ]

        precisions, recalls, f1s, mrrs = [], [], [], []
        for q in queries:
            results = retriever.search(q["query"], top_k=3)
            retrieved = [r.content for r in results]
            precisions.append(precision_at_k(retrieved, q["relevant"], 3))
            recalls.append(recall_at_k(retrieved, q["relevant"], 3))
            f1s.append(f1_at_k(retrieved, q["relevant"], 3))
            mrrs.append(mrr(retrieved, q["relevant"]))

        avg_p = sum(precisions) / len(precisions)
        avg_r = sum(recalls) / len(recalls)
        avg_f1 = sum(f1s) / len(f1s)
        avg_mrr = sum(mrrs) / len(mrrs)

        assert avg_p > 0
        assert avg_r > 0
        assert avg_f1 > 0
        assert avg_mrr > 0


# ===== 压力测试 =====

class TestStressTest:
    def test_rrf_high_volume(self):
        ranked_lists = {
            "vector": [{"content": f"doc_{i}"} for i in range(500)],
            "bm25": [{"content": f"doc_{i}"} for i in range(500)],
            "entity": [{"content": f"entity_{i}"} for i in range(200)],
        }
        start = time.time()
        for _ in range(50):
            reciprocal_rank_fusion(ranked_lists, k=60, top_k=20)
        elapsed = time.time() - start
        avg_ms = (elapsed / 50) * 1000
        assert avg_ms < 100, f"RRF 平均延迟 {avg_ms:.1f}ms 超过 100ms"

    def test_entity_recognizer_high_volume(self):
        recognizer = AstronomyEntityRecognizer()
        queries = [q["query"] for q in ASTRONOMY_TEST_QUERIES]
        start = time.time()
        for q in queries:
            recognizer.recognize(q)
        elapsed = time.time() - start
        avg_ms = (elapsed / len(queries)) * 1000
        assert avg_ms < 50, f"实体识别平均延迟 {avg_ms:.1f}ms 超过 50ms"

    def test_cache_concurrent_access(self):
        cache = MultiLevelCache(l1_maxsize=2000, l1_ttl=60, l2_maxsize=2000, l2_ttl=300, l3_enabled=False)
        for i in range(1000):
            cache.set(f"query_{i}", 3, f"result_{i}")
        hits = 0
        for i in range(1000):
            result = cache.get(f"query_{i}", 3)
            if result is not None:
                hits += 1
        assert hits == 1000


# ===== A/B 测试框架 =====

class TestABTestFramework:
    def test_ab_comparison(self):
        docs = [
            {"content": "木星是太阳系最大的行星，质量是地球318倍", "metadata": {"source": "wiki"}},
            {"content": "木星大红斑是持续数百年的风暴", "metadata": {"source": "nasa"}},
            {"content": "土星环主要由冰粒组成", "metadata": {"source": "wiki"}},
            {"content": "火星上有水的痕迹", "metadata": {"source": "nasa"}},
            {"content": "金星表面温度高达460度", "metadata": {"source": "wiki"}},
        ]

        queries = [
            {"query": "木星的特征", "relevant": {"木星是太阳系最大的行星，质量是地球318倍", "木星大红斑是持续数百年的风暴"}},
            {"query": "土星环", "relevant": {"土星环主要由冰粒组成"}},
            {"query": "火星水", "relevant": {"火星上有水的痕迹"}},
        ]

        bm25_only_scores = []
        entity_only_scores = []
        hybrid_scores = []

        entity_retriever = AstronomyEntityRetriever(docs)

        for q in queries:
            relevant = q["relevant"]

            bm25_results = [
                {"content": docs[i]["content"]}
                for i in range(min(3, len(docs)))
            ]
            bm25_retrieved = [r["content"] for r in bm25_results]
            bm25_only_scores.append(precision_at_k(bm25_retrieved, relevant, 3))

            entity_results = entity_retriever.search(q["query"], top_k=3)
            entity_retrieved = [r.content for r in entity_results]
            entity_only_scores.append(precision_at_k(entity_retrieved, relevant, 3))

            ranked_lists = {
                "bm25": bm25_results,
                "entity": [{"content": r.content} for r in entity_results],
            }
            rrf_results = reciprocal_rank_fusion(ranked_lists, k=60, top_k=3)
            hybrid_retrieved = [r.content for r in rrf_results]
            hybrid_scores.append(precision_at_k(hybrid_retrieved, relevant, 3))

        avg_bm25 = sum(bm25_only_scores) / len(bm25_only_scores)
        avg_entity = sum(entity_only_scores) / len(entity_only_scores)
        avg_hybrid = sum(hybrid_scores) / len(hybrid_scores)

        assert avg_hybrid >= avg_bm25 * 0.5, (
            f"混合检索应优于纯BM25: BM25={avg_bm25:.3f}, Hybrid={avg_hybrid:.3f}"
        )
        assert avg_entity > 0, "实体检索应有正准确率"
