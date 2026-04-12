"""
时效性感知重排序与结果过滤机制

功能:
  - 时效性评分: 基于天文数据的发布/观测时间计算时效性分数
  - 可信度过滤: 基于数据来源可信度过滤低质量结果
  - 天文领域相关性增强: 基于天文实体匹配提升领域相关结果
  - 综合重排序: 融合时效性、可信度、领域相关性进行最终排序
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.core.config import settings
from src.core.logger import logger
from src.rag.reranker import RerankResult


HIGH_CREDIBILITY_SOURCES = {
    "nasa", "esa", "cnsa", "jaxa", "esa",
    "iap", "naoc", "pku", "tsinghua",
    "arxiv", "ads", "simbad", "vizier",
    "aas", "iau", "eso", "noao", "stsci",
}

MEDIUM_CREDIBILITY_SOURCES = {
    "wikipedia", "wiki", "britannica",
    "skyandtelescope", "astronomy.com",
    "space.com", "nasa.gov",
}

TIME_SENSITIVE_TOPICS = {
    "新发现", "最新观测", "近日", "首次探测", "突破性",
    "new.discovery", "latest", "recent.observation",
    "first.detection", "breakthrough",
    "超新星爆发", "彗星发现", "近地天体",
    "supernova", "comet.discovery", "neo",
    "流星雨预报", "日食", "月食",
    "meteor.shower", "eclipse",
}

OUTDATED_THRESHOLD_YEARS = 10


@dataclass
class FilteredResult:
    content: str
    relevance_score: float
    timeliness_score: float
    credibility_score: float
    domain_score: float
    final_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    filtered_out: bool = False
    filter_reason: str = ""


class TimelinessScorer:
    """时效性评分器"""

    def score(self, metadata: Dict[str, Any], query: str = "") -> float:
        pub_year = metadata.get("publication_year")
        obs_date = metadata.get("observation_date")
        is_time_sensitive = metadata.get("is_time_sensitive", False)

        query_time_sensitive = any(kw in query for kw in TIME_SENSITIVE_TOPICS)

        if not query_time_sensitive and not is_time_sensitive:
            return 0.7

        year = None
        if pub_year:
            try:
                year = int(pub_year)
            except (ValueError, TypeError):
                pass

        if year is None and obs_date:
            year_match = re.search(r'(\d{4})', str(obs_date))
            if year_match:
                try:
                    year = int(year_match.group(1))
                except ValueError:
                    pass

        current_year = time.localtime().tm_year

        if year is None:
            return 0.5

        age = current_year - year

        if query_time_sensitive:
            if age <= 1:
                return 1.0
            elif age <= 3:
                return 0.8
            elif age <= 5:
                return 0.5
            else:
                return 0.2
        else:
            if age <= OUTDATED_THRESHOLD_YEARS:
                return 0.7
            else:
                return 0.4


class CredibilityScorer:
    """可信度评分器"""

    def score(self, metadata: Dict[str, Any]) -> float:
        source = metadata.get("source", "").lower()
        data_source = metadata.get("data_source", "").lower()
        combined = source + " " + data_source

        for src in HIGH_CREDIBILITY_SOURCES:
            if src in combined:
                return 1.0

        for src in MEDIUM_CREDIBILITY_SOURCES:
            if src in combined:
                return 0.8

        if metadata.get("doc_type") in ("star_catalog", "orbital_parameters"):
            return 0.9

        if metadata.get("doc_type") in ("observation_record",):
            return 0.7

        return 0.5


class DomainRelevanceScorer:
    """天文领域相关性评分器"""

    ASTRONOMY_KEYWORDS = {
        "行星", "恒星", "星系", "星云", "星团", "望远镜", "观测",
        "轨道", "光谱", "星等", "赤经", "赤纬", "天文",
        "planet", "star", "galaxy", "nebula", "cluster", "telescope",
        "orbit", "spectral", "magnitude", "astronomy",
    }

    def score(self, content: str, query: str) -> float:
        content_lower = content.lower()
        query_lower = query.lower()

        query_hits = sum(1 for kw in self.ASTRONOMY_KEYWORDS if kw in query_lower)
        content_hits = sum(1 for kw in self.ASTRONOMY_KEYWORDS if kw in content_lower)

        if query_hits == 0:
            domain_query_score = 0.5
        else:
            domain_query_score = min(query_hits / 3.0, 1.0)

        if content_hits == 0:
            domain_content_score = 0.3
        else:
            domain_content_score = min(content_hits / 5.0, 1.0)

        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = query_words & content_words
        word_overlap_score = len(overlap) / max(len(query_words), 1)

        return 0.3 * domain_query_score + 0.3 * domain_content_score + 0.4 * word_overlap_score


class ResultFilterAndReranker:
    """结果过滤与重排序器"""

    def __init__(
        self,
        min_credibility: float = 0.3,
        timeliness_weight: float = 0.2,
        credibility_weight: float = 0.2,
        domain_weight: float = 0.3,
        relevance_weight: float = 0.3,
    ):
        self.min_credibility = min_credibility
        self.timeliness_weight = timeliness_weight
        self.credibility_weight = credibility_weight
        self.domain_weight = domain_weight
        self.relevance_weight = relevance_weight

        self.timeliness_scorer = TimelinessScorer()
        self.credibility_scorer = CredibilityScorer()
        self.domain_scorer = DomainRelevanceScorer()

    def filter_and_rerank(
        self,
        results: List[RerankResult],
        query: str,
        top_k: int = 5,
    ) -> List[FilteredResult]:
        if not results:
            return []

        scored_results = []
        for result in results:
            meta = result.metadata or {}

            credibility = self.credibility_scorer.score(meta)
            if credibility < self.min_credibility:
                scored_results.append(FilteredResult(
                    content=result.content,
                    relevance_score=result.relevance_score,
                    timeliness_score=0.0,
                    credibility_score=credibility,
                    domain_score=0.0,
                    final_score=0.0,
                    metadata=meta,
                    filtered_out=True,
                    filter_reason=f"可信度过低: {credibility:.2f} < {self.min_credibility}",
                ))
                continue

            timeliness = self.timeliness_scorer.score(meta, query)
            domain = self.domain_scorer.score(result.content, query)

            max_relevance = max(r.relevance_score for r in results) if results else 1.0
            normalized_relevance = result.relevance_score / max_relevance if max_relevance > 0 else 0.0

            final_score = (
                self.relevance_weight * normalized_relevance
                + self.timeliness_weight * timeliness
                + self.credibility_weight * credibility
                + self.domain_weight * domain
            )

            scored_results.append(FilteredResult(
                content=result.content,
                relevance_score=result.relevance_score,
                timeliness_score=timeliness,
                credibility_score=credibility,
                domain_score=domain,
                final_score=final_score,
                metadata=meta,
            ))

        valid_results = [r for r in scored_results if not r.filtered_out]
        valid_results.sort(key=lambda x: x.final_score, reverse=True)

        filtered_out = [r for r in scored_results if r.filtered_out]
        if filtered_out:
            logger.info(f"🔍 结果过滤: 移除 {len(filtered_out)} 个低可信度结果")

        final = valid_results[:top_k]
        logger.info(
            f"🔍 重排序完成: {len(results)} 输入 → {len(valid_results)} 有效 → {len(final)} 最终"
        )
        return final
