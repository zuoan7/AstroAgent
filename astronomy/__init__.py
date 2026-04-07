# -*- coding: utf-8 -*-
"""
天文工具模块 - 提供统一的天文计算和查询接口

本模块将原来的 astronomy_tools.py 拆分为多个子模块：
- base: 基础类和星历数据管理
- planetary: 行星位置计算、坐标转换等
- celestial_databases: 天体数据库查询（SIMBAD, NED）
- nasa_api: NASA API调用（APOD, NEO）
- weather_service: 天气服务（高德API）
- search_service: 联网搜索服务（Tavily）
- events_predictor: 天象预测器
"""

from .base import EphemerisManager
from .planetary import PlanetaryCalculator
from .celestial_databases import CelestialDatabaseService
from .nasa_api import NASAAPIService
from .weather_service import WeatherService
from .search_service import SearchService
from .events_predictor import EventsPredictor


class AstronomyTools:
    """
    统一的天文工具入口（向后兼容）
    
    整合所有天文子模块，提供统一的接口。
    保持与原 AstronomyTools 类的接口兼容性。
    """
    
    def __init__(self):
        # 初始化星历管理器（共享星历数据，避免重复加载）
        self._ephemeris = EphemerisManager()
        
        # 初始化各子模块
        self.planetary = PlanetaryCalculator(self._ephemeris)
        self.celestial_db = CelestialDatabaseService()
        self.nasa_api = NASAAPIService()
        self.weather = WeatherService()
        self.search = SearchService()
        self.events = EventsPredictor(self._ephemeris)
        
        # 兼容属性
        self.data_loaded = self._ephemeris.is_loaded
        self.planets = self._ephemeris.planets
        self.earth = self._ephemeris.earth
    
    def get_planet_position(self, planet_name, observation_time=None, latitude=None, longitude=None):
        """获取行星位置（兼容接口）"""
        return self.planetary.get_planet_position(planet_name, observation_time, latitude, longitude)
    
    def coordinate_transformation(self, ra, dec, epoch='J2000', target_system='fk5'):
        """天体坐标转换（兼容接口）"""
        return self.planetary.coordinate_transformation(ra, dec, epoch, target_system)
    
    def get_rise_set_times(self, body_name, latitude, longitude, date=None):
        """获取天体升起落下时间（兼容接口）"""
        return self.planetary.get_rise_set_times(body_name, latitude, longitude, date)
    
    def get_current_sky_objects(self, latitude, longitude, date=None):
        """获取当前天空中的主要天体（兼容接口）"""
        return self.planetary.get_current_sky_objects(latitude, longitude, date)
    
    def get_astrophysical_object_info(self, object_name):
        """查询天体基本信息（兼容接口）"""
        return self.celestial_db.get_object_info(object_name)
    
    def get_galaxy_data(self, galaxy_name):
        """星系数据查询（兼容接口）"""
        return self.celestial_db.get_galaxy_data(galaxy_name)
    
    def get_nasa_apod(self, date=None, hd=False):
        """获取NASA每日天文图（兼容接口）"""
        return self.nasa_api.get_apod(date, hd)
    
    def get_neo_data(self, start_date=None, end_date=None, limit=20):
        """获取近地天体数据（兼容接口）"""
        return self.nasa_api.get_neo_data(start_date, end_date, limit)
    
    def _reverse_geocode(self, longitude, latitude):
        """逆地理编码（兼容接口）"""
        return self.weather.reverse_geocode(longitude, latitude)
    
    def _is_coordinates(self, text):
        """检测是否为坐标格式（兼容接口）"""
        from utils.helpers import is_coordinates
        return is_coordinates(text)
    
    def get_weather(self, city=None, extensions="base"):
        """查询天气（兼容接口）"""
        return self.weather.get_weather(city, extensions)
    
    def web_search(self, query: str, max_results: int = 5):
        """联网搜索（兼容接口）"""
        return self.search.search(query, max_results)


class AstronomyEventsPredictor:
    """
    天象预测工具（向后兼容）
    
    简单包装 EventsPredictor 类，保持原有接口不变。
    """
    
    def __init__(self, location=None):
        self._predictor = EventsPredictor(location=location)
        
        # 兼容属性
        self.planets = self._predictor.ephemeris.planets
        self.ts = self._predictor.ephemeris.timescale
        self.lat = self._predictor.lat
        self.lon = self._predictor.lon
        self.earth = self._predictor.earth
        self.observer_location = self._predictor.observer_location
        self.special_events_2026 = self._predictor.special_events
    
    def get_moon_phase(self, date):
        """计算月相（兼容接口）"""
        return self._predictor.get_moon_phase(date)
    
    def get_sunrise_sunset(self, date):
        """计算日出日落（兼容接口）"""
        return self._predictor.get_sunrise_sunset(date)
    
    def get_visible_planets(self, date):
        """获取可见行星（兼容接口）"""
        return self._predictor.get_visible_planets(date)
    
    def _get_direction(self, az_degrees):
        """获取方向（兼容接口）"""
        from utils.helpers import get_direction_from_azimuth
        return get_direction_from_azimuth(az_degrees)
    
    def get_weekly_events(self, start_date=None):
        """获取周预报（兼容接口）"""
        return self._predictor.get_weekly_events(start_date)
    
    def get_monthly_events(self, year=None, month=None):
        """获取月预报（兼容接口）"""
        return self._predictor.get_monthly_events(year, month)
    
    def get_tonight_best(self):
        """获取今晚最佳观测目标（兼容接口）"""
        return self._predictor.get_tonight_best()


__all__ = [
    'AstronomyTools',
    'AstronomyEventsPredictor',
    'EphemerisManager',
    'PlanetaryCalculator',
    'CelestialDatabaseService',
    'NASAAPIService',
    'WeatherService',
    'SearchService',
    'EventsPredictor',
]
