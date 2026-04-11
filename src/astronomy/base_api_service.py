from __future__ import annotations

from typing import Any, Dict, Optional

import requests
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pybreaker import CircuitBreaker

from src.core.config import settings
from src.core.errors import AgentError, ErrorCode, ErrorHandler
from src.core.logger import logger


class BaseAPIService:
    """
    Base class for external API services providing shared resilience patterns.

    Encapsulates:
    - TTLCache for response caching
    - CircuitBreaker for fault tolerance
    - Tenacity retry with exponential backoff
    - Common _request_get / _request_post methods
    - Unified error handling via AgentError

    Subclasses must set:
    - _api_key_attr: settings attribute name for the API key
    - _cache: TTLCache instance
    - _breaker: CircuitBreaker instance
    - _retry_config: dict with stop, wait, retry, reraise keys for tenacity
    """

    _api_key_attr: str = ""
    _cache: Optional[TTLCache] = None
    _breaker: Optional[CircuitBreaker] = None
    _retry_stop = stop_after_attempt(3)
    _retry_wait = wait_exponential(multiplier=1, min=2, max=10)
    _retry_on = (requests.Timeout, requests.ConnectionError)

    def __init__(self) -> None:
        self.api_key: Optional[str] = getattr(settings, self._api_key_attr, None) if self._api_key_attr else None

    def _check_api_key(self, api_name: str) -> Optional[AgentError]:
        if not self.api_key:
            return AgentError(
                code=ErrorCode.API_ERROR,
                message=f"{self._api_key_attr} 未配置，无法查询{api_name}数据",
                details={"api": api_name}
            )
        return None

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        if self._cache is None:
            return None
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"{self.__class__.__name__}缓存命中: {cache_key}")
            return cached
        return None

    def _set_cached(self, cache_key: str, value: Any) -> None:
        if self._cache is not None:
            self._cache[cache_key] = value

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        reraise=True,
    )
    def _request_get(self, url: str, params: dict, timeout: int = 30) -> requests.Response:
        if self._breaker:
            return self._breaker(requests.get)(url, params=params, timeout=timeout)
        return requests.get(url, params=params, timeout=timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        reraise=True,
    )
    def _request_post(self, url: str, json: dict, timeout: int = 30) -> requests.Response:
        if self._breaker:
            return self._breaker(requests.post)(url, json=json, timeout=timeout)
        return requests.post(url, json=json, timeout=timeout)

    def _handle_error(self, error: Exception, api_name: str, details: Optional[Dict] = None) -> AgentError:
        logger.error(f"{self.__class__.__name__}调用失败: {error}")
        return ErrorHandler.handle(error, {"api": api_name, **(details or {})})
