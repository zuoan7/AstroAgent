from datetime import datetime, timedelta
from typing import Optional

from cachetools import TTLCache
from pybreaker import CircuitBreaker

from src.core.config import settings
from src.core.errors import AgentError, ErrorCode
from src.core.logger import logger
from src.astronomy.base_api_service import BaseAPIService

_APOD_CACHE = TTLCache(maxsize=128, ttl=86400)
_NEO_CACHE = TTLCache(maxsize=64, ttl=3600)

_NASA_BREAKER = CircuitBreaker(fail_max=5, reset_timeout=60)


class NASAAPIService(BaseAPIService):

    _api_key_attr = "NASA_API_KEY"
    _cache = _APOD_CACHE
    _breaker = _NASA_BREAKER

    def __init__(self):
        super().__init__()
        self._neo_cache = _NEO_CACHE

    def get_apod(self, date=None, hd=False) -> dict:
        key_error = self._check_api_key("APOD")
        if key_error:
            raise key_error

        cache_key = f"apod:{date or 'today'}:{hd}"
        cached = self._get_cached(cache_key)
        if cached is not None:
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
            self._set_cached(cache_key, result)
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
        key_error = self._check_api_key("NEO")
        if key_error:
            raise key_error

        cache_key = f"neo:{start_date}:{end_date}:{limit}"
        cached = self._neo_cache.get(cache_key)
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
            self._neo_cache[cache_key] = result
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
