"""
Rerank 模型重排序模块

接入 DashScope qwen3-rerank 模型，对 RRF 融合后的候选文档进行二次精细化排序。
支持两种调用方式：
1. DashScope SDK（优先）
2. OpenAI 兼容 HTTP API（降级）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger


@dataclass
class RerankResult:
    index: int
    relevance_score: float
    content: str
    metadata: Dict[str, Any]


class DashScopeReranker:
    """DashScope Rerank 模型客户端"""

    RERANK_API_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    RERANK_NATIVE_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        top_n: Optional[int] = None,
        request_timeout: float = 30,
        enabled: Optional[bool] = None,
    ):
        """初始化 DashScope rerank 客户端、超时和显式启停配置。"""

        self.model_name = model_name or settings.RERANK_MODEL_NAME
        self.api_key = api_key or settings.DASHSCOPE_API_KEY
        self.top_n = top_n or settings.RERANK_TOP_N
        self.request_timeout = request_timeout
        rerank_enabled = settings.RERANK_ENABLED if enabled is None else enabled
        self.enabled = bool(rerank_enabled) and bool(self.api_key)

        if not self.enabled:
            if not self.api_key:
                logger.warning("⚠️  DASHSCOPE_API_KEY 未配置，Rerank 重排序已禁用")
            else:
                logger.warning("⚠️  RERANK_ENABLED=False，Rerank 重排序已禁用")
            return

        self._use_sdk = self._check_sdk_available()
        if self._use_sdk:
            logger.info(f"✅ Rerank 初始化完成（SDK 模式）: model={self.model_name}")
        else:
            logger.info(f"✅ Rerank 初始化完成（HTTP 模式）: model={self.model_name}")

    @staticmethod
    def _check_sdk_available() -> bool:
        """检测本地 dashscope SDK 是否提供 TextReRank 能力。"""

        try:
            import dashscope
            return hasattr(dashscope, "TextReRank")
        except ImportError:
            return False

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        对文档列表进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档内容列表
            top_n: 返回前 N 个结果

        Returns:
            按 relevance_score 降序排列的 RerankResult 列表
        """
        if not self.enabled:
            return self._fallback_passthrough(documents)

        if not documents:
            return []

        n = top_n or self.top_n

        if self._use_sdk:
            return self._rerank_via_sdk(query, documents, n)
        return self._rerank_via_http(query, documents, n)

    def _rerank_via_sdk(
        self,
        query: str,
        documents: List[str],
        top_n: int,
    ) -> List[RerankResult]:
        """通过 DashScope SDK 调用 Rerank"""
        try:
            import dashscope
            dashscope.api_key = self.api_key

            resp = dashscope.TextReRank.call(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=True,
            )

            if resp.status_code != 200:
                logger.error(f"❌ Rerank SDK 调用失败: status={resp.status_code}, message={resp.message}")
                return self._fallback_passthrough(documents[:top_n])

            results = []
            output = resp.output if hasattr(resp, "output") else {}
            rerank_results = output.get("results", []) if isinstance(output, dict) else []

            for item in rerank_results:
                idx = item.get("index", 0)
                score = item.get("relevance_score", 0.0)
                doc_text = item.get("document", {}).get("text", "") if isinstance(item.get("document"), dict) else documents[idx] if idx < len(documents) else ""
                results.append(RerankResult(
                    index=idx,
                    relevance_score=score,
                    content=doc_text or documents[idx],
                    metadata={},
                ))

            results.sort(key=lambda x: x.relevance_score, reverse=True)
            logger.info(f"✅ Rerank SDK 完成: {len(documents)} -> {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"❌ Rerank SDK 调用异常: {e}")
            return self._fallback_passthrough(documents[:top_n])

    def _rerank_via_http(
        self,
        query: str,
        documents: List[str],
        top_n: int,
    ) -> List[RerankResult]:
        """通过 OpenAI 兼容 HTTP API 调用 Rerank"""
        try:
            import requests

            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": True,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            resp = requests.post(
                self.RERANK_API_URL,
                json=payload,
                headers=headers,
                timeout=getattr(self, "request_timeout", 30),
            )
            resp.raise_for_status()

            data = resp.json()
            raw_results = data.get("results", [])

            results = []
            for item in raw_results:
                idx = item.get("index", 0)
                score = item.get("relevance_score", 0.0)
                doc_text = item.get("document", {}).get("text", "") if isinstance(item.get("document"), dict) else ""
                results.append(RerankResult(
                    index=idx,
                    relevance_score=score,
                    content=doc_text or (documents[idx] if idx < len(documents) else ""),
                    metadata={},
                ))

            results.sort(key=lambda x: x.relevance_score, reverse=True)
            logger.info(f"✅ Rerank HTTP 完成: {len(documents)} -> {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"❌ Rerank HTTP 调用异常: {e}")
            return self._fallback_passthrough(documents[:top_n])

    @staticmethod
    def _fallback_passthrough(documents: List[str]) -> List[RerankResult]:
        """降级策略：不排序，直接按原序返回"""
        return [
            RerankResult(index=i, relevance_score=0.0, content=doc, metadata={})
            for i, doc in enumerate(documents)
        ]
