# -*- coding: utf-8 -*-
import os
import logging

import requests
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pybreaker import CircuitBreaker

from src.core.config import settings

logger = logging.getLogger(__name__)

SEARCH_CACHE = TTLCache(maxsize=128, ttl=1800)

search_api_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
)


class SearchService:

    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY") or getattr(settings, "TAVILY_API_KEY", None)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        reraise=True,
    )
    @search_api_breaker
    def _request_post(self, url: str, json: dict, timeout: int = 30) -> requests.Response:
        return requests.post(url, json=json, timeout=timeout)

    def search(self, query: str, max_results: int = 5) -> dict:
        try:
            if not self.api_key:
                return {
                    "error": "TAVILY_API_KEY 未配置",
                    "suggestion": "请在.env文件中配置 TAVILY_API_KEY"
                }

            cache_key = f"search:{query}:{max_results}"
            cached = SEARCH_CACHE.get(cache_key)
            if cached is not None:
                logger.debug(f"搜索缓存命中: {cache_key}")
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

            SEARCH_CACHE[cache_key] = result
            return result

        except requests.exceptions.Timeout:
            logger.error(f"搜索超时: {query}")
            return {"error": "搜索请求超时，请稍后重试"}

        except requests.exceptions.RequestException as e:
            logger.error(f"搜索请求失败: {e}")
            return {"error": f"搜索请求失败: {str(e)}"}

        except Exception as e:
            logger.error(f"搜索异常: {e}")
            return {"error": f"搜索异常: {str(e)}"}


__all__ = ['SearchService']
