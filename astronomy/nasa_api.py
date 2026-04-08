# -*- coding: utf-8 -*-
"""
NASA API服务模块 - APOD和NEO数据查询
"""

import logging
from datetime import datetime, timedelta

import requests

from config import settings

logger = logging.getLogger(__name__)


class NASAAPIService:
    """
    NASA API 服务
    
    提供对NASA各种API的访问，包括APOD（每日天文图）和NEO（近地天体）数据。
    """
    
    def __init__(self):
        self.api_key = settings.NASA_API_KEY
    
    def get_apod(self, date=None, hd=False) -> dict:
        """
        获取NASA每日天文图（Astronomy Picture of the Day）
        
        Args:
            date: 日期（YYYY-MM-DD格式，可选）
            hd: 是否获取高清图像
            
        Returns:
            APOD信息字典
        """
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
            
        except Exception as e:
            logger.error(f"获取NASA APOD失败: {e}")
            return {'error': f'获取NASA每日天文图时出错: {e}'}
    
    def get_neo_data(self, start_date=None, end_date=None, limit=20) -> dict:
        """
        获取近地天体（Near-Earth Objects）数据
        
        Args:
            start_date: 开始日期（YYYY-MM-DD格式，可选）
            end_date: 结束日期（YYYY-MM-DD格式，可选）
            limit: 返回结果数量限制
            
        Returns:
            NEO数据字典
        """
        try:
            # 处理日期参数
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
            
            # 确保不超过API限制
            delta_days = (end_dt - start_dt).days
            if delta_days > settings.NASA_NEO_MAX_DAYS:
                logger.warning(f"请求的日期范围({delta_days}天)超过限制({settings.NASA_NEO_MAX_DAYS}天)，将截断")
                end_dt = start_dt + timedelta(days=settings.NASA_NEO_MAX_DAYS)
                end_date = end_dt.strftime("%Y-%m-%d")
            
            # 构建请求
            params = {
                "api_key": self.api_key,
                "limit": limit,
                "start_date": start_date,
                "end_date": end_date
            }
            
            response = requests.get(settings.NASA_NEO_URL, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"获取NEO数据失败: {e}")
            return {'error': f'获取近地天体数据时出错: {e}'}


__all__ = ['NASAAPIService']
