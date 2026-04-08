# -*- coding: utf-8 -*-
"""
常量配置模块 - 向后兼容层

注意：此模块已迁移至 config.py，保留此文件仅为向后兼容。
新代码应直接从 config.settings 导入配置项。
"""

from config import settings

# ========== 行星相关常量 ==========
PLANET_MAPPING: dict = settings.PLANET_MAPPING
VALID_PLANETS: set = settings.VALID_PLANETS
PLANET_NAMES_CN: dict = settings.PLANET_NAMES_CN
PLANET_MAX_MAGNITUDE: dict = settings.PLANET_MAX_MAGNITUDE

# ========== 天体名称映射 ==========
CELESTIAL_NAME_MAPPING: dict = settings.CELESTIAL_NAME_MAPPING
GALAXY_NAME_MAPPING: dict = settings.GALAXY_NAME_MAPPING

# ========== 支持的天体（用于升起落下计算） ==========
SUPPORTED_BODIES: list = settings.SUPPORTED_BODIES

# ========== 默认观测位置 ==========
DEFAULT_LOCATION: tuple = settings.DEFAULT_LOCATION

# ========== API 配置 ==========
NASA_APOD_URL: str = settings.NASA_APOD_URL
NASA_NEO_URL: str = settings.NASA_NEO_URL
AMAP_WEATHER_URL: str = settings.AMAP_WEATHER_URL
AMAP_GEOCODE_URL: str = settings.AMAP_GEOCODE_URL
TAVILY_SEARCH_URL: str = settings.TAVILY_SEARCH_URL

# ========== NASA API 限制 ==========
NASA_NEO_MAX_DAYS: int = settings.NASA_NEO_MAX_DAYS

# ========== 行星可见性建议（按月份） ==========
PLANET_VISIBILITY_BY_MONTH: dict = settings.PLANET_VISIBILITY_BY_MONTH

# ========== 月相阈值（度） ==========
MOON_PHASE_THRESHOLDS: list = settings.MOON_PHASE_THRESHOLDS

# ========== 观测建议模板 ==========
OBSERVING_TIPS_TEMPLATES: dict = settings.OBSERVING_TIPS_TEMPLATES


__all__ = [
    'PLANET_MAPPING',
    'VALID_PLANETS',
    'PLANET_NAMES_CN',
    'PLANET_MAX_MAGNITUDE',
    'CELESTIAL_NAME_MAPPING',
    'GALAXY_NAME_MAPPING',
    'SUPPORTED_BODIES',
    'DEFAULT_LOCATION',
    'NASA_APOD_URL',
    'NASA_NEO_URL',
    'AMAP_WEATHER_URL',
    'AMAP_GEOCODE_URL',
    'TAVILY_SEARCH_URL',
    'NASA_NEO_MAX_DAYS',
    'PLANET_VISIBILITY_BY_MONTH',
    'MOON_PHASE_THRESHOLDS',
    'OBSERVING_TIPS_TEMPLATES',
]
