# -*- coding: utf-8 -*-
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pybreaker import CircuitBreaker

from src.core.config import settings
from src.core.errors import AgentError, ErrorCode, ErrorHandler

logger = logging.getLogger(__name__)

NASA_APOD_CACHE = TTLCache(maxsize=128, ttl=86400)
NASA_NEO_CACHE = TTLCache(maxsize=64, ttl=3600)

nasa_api_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
)


class NASAAPIService:

    def __init__(self):
        self.api_key = settings.NASA_API_KEY

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        reraise=True,
    )
    @nasa_api_breaker
    def _request_get(self, url: str, params: dict, timeout: int = 30) -> requests.Response:
        return requests.get(url, params=params, timeout=timeout)

    def get_apod(self, date=None, hd=False) -> dict:
        if not self.api_key:
            raise AgentError(
                code=ErrorCode.NASA_API_ERROR,
                message="NASA_API_KEY 未配置，无法查询 NASA 数据",
                details={"api": "APOD"}
            )

        cache_key = f"apod:{date or 'today'}:{hd}"
        cached = NASA_APOD_CACHE.get(cache_key)
        if cached is not None:
            logger.debug(f"NASA APOD缓存命中: {cache_key}")
            return cached

        try:
            params = {
                "api_key": self.api_key,
                "hd": str(hd).lower()
            }

            if date:
                params["date"] = date

            response = self._request_get(settings.NASA_APOD_URL, params=params)
            response.raise_for_status()

            result = response.json()
            NASA_APOD_CACHE[cache_key] = result
            return result

        except AgentError:
            raise
        except Exception as e:
            logger.error(f"获取NASA APOD失败: {e}")
            raise AgentError(
                code=ErrorCode.NASA_API_ERROR,
                message=f"获取NASA每日天文图时出错: {e}",
                details={"api": "APOD", "date": date},
                original_error=e
            )

    def get_neo_data(self, start_date=None, end_date=None, limit=20) -> dict:
        if not self.api_key:
            raise AgentError(
                code=ErrorCode.NASA_API_ERROR,
                message="NASA_API_KEY 未配置，无法查询近地天体数据",
                details={"api": "NEO"}
            )

        cache_key = f"neo:{start_date}:{end_date}:{limit}"
        cached = NASA_NEO_CACHE.get(cache_key)
        if cached is not None:
            logger.debug(f"NASA NEO缓存命中: {cache_key}")
            return cached

        try:
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                start_dt = datetime.now()
                start_date = start_dt.strftime("%Y-%m-%d")

            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end_dt = start_dt + timedelta(days=7)
                end_date = end_dt.strftime("%Y-%m-%d")

            delta_days = (end_dt - start_dt).days
            if delta_days > settings.NASA_NEO_MAX_DAYS:
                logger.warning(f"请求的日期范围({delta_days}天)超过限制({settings.NASA_NEO_MAX_DAYS}天)，将截断")
                end_dt = start_dt + timedelta(days=settings.NASA_NEO_MAX_DAYS)
                end_date = end_dt.strftime("%Y-%m-%d")

            params = {
                "api_key": self.api_key,
                "limit": limit,
                "start_date": start_date,
                "end_date": end_date
            }

            response = self._request_get(settings.NASA_NEO_URL, params=params)
            response.raise_for_status()

            result = response.json()
            NASA_NEO_CACHE[cache_key] = result
            return result

        except AgentError:
            raise
        except Exception as e:
            logger.error(f"获取NEO数据失败: {e}")
            raise AgentError(
                code=ErrorCode.NASA_API_ERROR,
                message=f"获取近地天体数据时出错: {e}",
                details={"api": "NEO", "start_date": start_date, "end_date": end_date},
                original_error=e
            )


__all__ = ['NASAAPIService']
