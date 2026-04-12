"""
RRF (Reciprocal Rank Fusion) 重排序算法

将多个检索系统的排名结果融合为统一排序。
公式: RRF_score(d) = Σ 1/(k + rank_i(d))
其中 k 为常数（默认60），rank_i(d) 为文档 d 在第 i 个检索系统中的排名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger


@dataclass
class RankedDocument:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    rrf_score: float = 0.0
    source: str = ""
    rank_info: Dict[str, int] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.content[:200]


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[Dict[str, Any]]],
    k: int = None,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = None,
) -> List[RankedDocument]:
    """
    RRF 融合算法

    Args:
        ranked_lists: 检索系统名称到排序结果列表的映射
            每个结果为 dict，需包含 "content" 键和可选的 "metadata" 键
        k: RRF 常数，控制排名衰减速度（默认从配置读取）
        weights: 各检索系统的权重，默认等权
        top_k: 返回前 top_k 个结果

    Returns:
        按 RRF 分数降序排列的 RankedDocument 列表
    """
    if k is None:
        k = settings.RRF_K
    if top_k is None:
        top_k = settings.RAG_RETRIEVAL_CANDIDATES

    if not ranked_lists:
        return []

    if weights is None:
        weights = {name: 1.0 for name in ranked_lists}

    doc_scores: Dict[str, RankedDocument] = {}

    for retriever_name, results in ranked_lists.items():
        weight = weights.get(retriever_name, 1.0)

        for rank_idx, result in enumerate(results):
            content = result.get("content", "") or result.get("document", "")
            if not content:
                continue

            metadata = result.get("metadata", {})
            doc_id = content[:200]
            rank = rank_idx + 1

            rrf_contribution = weight / (k + rank)

            if doc_id not in doc_scores:
                doc_scores[doc_id] = RankedDocument(
                    content=content,
                    metadata=metadata,
                    rrf_score=0.0,
                    source=retriever_name,
                    rank_info={},
                )

            doc = doc_scores[doc_id]
            doc.rrf_score += rrf_contribution
            doc.rank_info[retriever_name] = rank

            if doc.source != retriever_name and "hybrid" not in doc.source:
                doc.source = "hybrid"

    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x.rrf_score,
        reverse=True,
    )

    result = sorted_docs[:top_k]

    logger.info(
        f"📊 RRF 融合完成: {len(ranked_lists)} 个检索系统, "
        f"共 {len(doc_scores)} 个唯一文档, 返回 top {len(result)}"
    )

    return result
