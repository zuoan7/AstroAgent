import os

import requests
from cachetools import TTLCache
from pybreaker import CircuitBreaker

from src.core.config import settings
from src.core.errors import AgentError, ErrorCode
from src.core.logger import logger
from src.astronomy.base_api_service import BaseAPIService

_SEARCH_CACHE = TTLCache(maxsize=128, ttl=1800)

_SEARCH_BREAKER = CircuitBreaker(fail_max=5, reset_timeout=60)


class SearchService(BaseAPIService):

    _api_key_attr = "TAVILY_API_KEY"
    _cache = _SEARCH_CACHE
    _breaker = _SEARCH_BREAKER

    def __init__(self):
        super().__init__()
        env_key = os.getenv("TAVILY_API_KEY")
        if env_key:
            self.api_key = env_key

    def search(self, query: str, max_results: int = 5) -> dict:
        try:
            if not self.api_key:
                return AgentError(
                    code=ErrorCode.API_ERROR,
                    message="TAVILY_API_KEY 未配置",
                    details={"suggestion": "请在.env文件中配置 TAVILY_API_KEY"}
                ).to_dict()

            cache_key = f"search:{query}:{max_results}"
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False
            }

            response = self._request_post(url, json=payload)
            response.raise_for_status()

            data = response.json()

            results = []
            if "results" in data:
                for item in data["results"]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", "")[:500]
                    })

            answer = data.get("answer", "")

            result = {
                "query": query,
                "answer": answer,
                "results": results,
                "total": len(results)
            }

            self._set_cached(cache_key, result)
            return result

        except requests.exceptions.Timeout:
            logger.error(f"搜索超时: {query}")
            return AgentError(
                code=ErrorCode.API_ERROR,
                message="搜索请求超时，请稍后重试",
                details={"query": query}
            ).to_dict()

        except requests.exceptions.RequestException as e:
            logger.error(f"搜索请求失败: {e}")
            return AgentError(
                code=ErrorCode.API_ERROR,
                message=f"搜索请求失败: {str(e)}",
                details={"query": query}
            ).to_dict()

        except Exception as e:
            logger.error(f"搜索异常: {e}")
            return AgentError(
                code=ErrorCode.UNKNOWN_ERROR,
                message=f"搜索异常: {str(e)}",
                details={"query": query}
            ).to_dict()


__all__ = ['SearchService']
