# -*- coding: utf-8 -*-
import logging
from datetime import datetime, timedelta

import requests

from config import settings
from src.core.errors import AgentError, ErrorCode, ErrorHandler

logger = logging.getLogger(__name__)


class NASAAPIService:

    def __init__(self):
        self.api_key = settings.NASA_API_KEY

    def get_apod(self, date=None, hd=False) -> dict:
        try:
            params = {
                "api_key": self.api_key,
                "hd": str(hd).lower()
            }

            if date:
                params["date"] = date

            response = requests.get(settings.NASA_APOD_URL, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

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

            response = requests.get(settings.NASA_NEO_URL, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

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
