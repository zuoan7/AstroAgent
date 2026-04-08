# -*- coding: utf-8 -*-
"""
天气服务模块 - 高德天气API接口
"""

import logging

import requests

from config import settings
from src.core.errors import ErrorHandler, ErrorCode
from src.utils.helpers import (
    parse_mixed_input,
    is_coordinates,
)

logger = logging.getLogger(__name__)


class WeatherService:
    """
    天气查询服务
    
    使用高德地图天气API提供实时和预报天气查询，
    并生成适合天文观测的建议。
    """
    
    def __init__(self):
        self.api_key = settings.AMAP_API_KEY
    
    def reverse_geocode(self, longitude: float, latitude: float) -> str:
        """
        使用高德逆地理编码API将经纬度转换为城市名
        
        Args:
            longitude: 经度
            latitude: 纬度
            
        Returns:
            城市名称，失败返回None
        """
        try:
            if not self.api_key:
                return None
            
            params = {
                "key": self.api_key,
                "location": f"{longitude},{latitude}",
                "output": "JSON",
            }
            
            resp = requests.get(settings.AMAP_GEOCODE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if str(data.get("status")) == "1" and data.get("regeocode"):
                address = data["regeocode"].get("addressComponent", {})
                city = address.get("city") or address.get("province")
                return city if city else None
                
            return None
            
        except Exception as e:
            logger.warning(f"逆地理编码失败: {e}")
            return None
    
    def get_weather(self, city=None, extensions="base") -> dict:
        """
        查询天气并生成观测建议
        
        Args:
            city: 城市名称或adcode（可选）
            extensions: "base"(实时) 或 "all"(预报)
            
        Returns:
            包含天气信息和观测建议的字典
        """
        try:
            # 统一参数解析
            params = parse_mixed_input(city, {"city": None, "extensions": extensions})
            
            if params.get("city"):
                city = params["city"]
                extensions = params.get("extensions", extensions)
            
            # 处理坐标输入
            if city and is_coordinates(city):
                parts = city.split(",")
                lon = float(parts[1].strip())
                lat = float(parts[0].strip())
                city_name = self.reverse_geocode(lon, lat)
                if city_name:
                    city = city_name
            
            # 验证API密钥
            if not self.api_key:
                error = ErrorHandler.create_tool_error(
                    "get_weather",
                    "AMAP_API_KEY 未配置，无法查询天气"
                )
                return error.to_dict()
            
            # 设置默认城市
            if not city:
                city = settings.AMAP_DEFAULT_CITY
            
            # 调用天气API
            params = {
                "key": self.api_key,
                "city": city,
                "extensions": extensions or "base",
                "output": "JSON",
            }
            
            resp = requests.get(settings.AMAP_WEATHER_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            # 检查响应状态
            if str(data.get("status")) != "1":
                error = ErrorHandler.create_tool_error(
                    "get_weather",
                    data.get("info") or "高德天气查询失败",
                    {"raw": data}
                )
                return error.to_dict()
            
            # 处理响应数据
            result = self._process_weather_response(data, city, extensions)
            return result
            
        except Exception as e:
            logger.error(f"天气查询失败: {e}")
            error = ErrorHandler.handle(e, {"tool": "get_weather", "city": city})
            return error.to_dict()
    
    def _process_weather_response(self, data: dict, city: str, extensions: str) -> dict:
        """处理天气API响应并生成观测建议"""
        
        lives = data.get("lives") or []
        forecasts = data.get("forecasts") or []
        
        result = {
            "query_city": city,
            "extensions": extensions or "base",
            "raw": data,
        }
        
        # 处理实时天气
        if lives:
            live = lives[0]
            weather = live.get("weather")
            humidity = live.get("humidity")
            windpower = live.get("windpower")
            
            result["live"] = {
                "city": live.get("city"),
                "weather": weather,
                "temperature": live.get("temperature"),
                "humidity": humidity,
                "winddirection": live.get("winddirection"),
                "windpower": windpower,
                "reporttime": live.get("reporttime"),
            }
            
            # 生成观测建议
            tips = self._generate_observing_tips(weather, humidity, windpower)
            result["observing_tips"] = tips
        
        # 处理天气预报
        if forecasts:
            result["forecast"] = forecasts[0]
        
        return result
    
    def _generate_observing_tips(self, weather: str, humidity: str, windpower: str) -> list:
        """根据天气条件生成观测建议"""
        
        tips = []
        
        # 天气现象判断
        if weather and any(k in weather for k in ["雨", "雪", "雷", "雾", "霾"]):
            tips.append(settings.OBSERVING_TIPS_TEMPLATES['bad_weather'])
        else:
            tips.append(settings.OBSERVING_TIPS_TEMPLATES['good_weather'])
        
        # 湿度判断
        if humidity is not None:
            try:
                h = float(humidity)
                if h >= 80:
                    tips.append(settings.OBSERVING_TIPS_TEMPLATES['high_humidity'])
            except (ValueError, TypeError):
                pass
        
        # 风力判断
        if windpower is not None:
            try:
                wp = float(str(windpower).replace("级", "").strip())
                if wp >= 4:
                    tips.append(settings.OBSERVING_TIPS_TEMPLATES['high_wind'])
            except (ValueError, TypeError):
                pass
        
        return tips


__all__ = ['WeatherService']
