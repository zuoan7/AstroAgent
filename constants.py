# -*- coding: utf-8 -*-
"""
常量配置模块 - 集中管理所有常量和映射关系
"""

from typing import Dict, List


# ========== 行星相关常量 ==========
PLANET_MAPPING: Dict[str, int] = {
    'mercury': 199,
    'venus': 299,
    'mars': 499,
    'jupiter': 5,
    'saturn': 6,
    'uranus': 7,
    'neptune': 8
}

VALID_PLANETS: set = set(PLANET_MAPPING.keys())

PLANET_NAMES_CN: Dict[str, str] = {
    'mercury': '水星',
    'venus': '金星',
    'mars': '火星',
    'jupiter': '木星',
    'saturn': '土星',
    'uranus': '天王星',
    'neptune': '海王星'
}

PLANET_MAX_MAGNITUDE: Dict[str, float] = {
    'mercury': -1.9,
    'venus': -4.6,
    'mars': -2.0,
    'jupiter': -2.7,
    'saturn': -0.3,
}


# ========== 天体名称映射 ==========
CELESTIAL_NAME_MAPPING: Dict[str, str] = {
    # 星系
    'Andromeda Galaxy': 'M31',
    '仙女座星系': 'M31',
    '仙女座大星系': 'M31',
    'Milky Way': 'Milky Way Galaxy',
    '银河系': 'Milky Way Galaxy',
    'Triangulum Galaxy': 'M33',
    '三角座星系': 'M33',
    
    # 恒星
    'Sirius': 'Sirius',
    '天狼星': 'Sirius',
    
    # 星云
    'Orion Nebula': 'M42',
    '猎户座星云': 'M42',
    '猎户座大星云': 'M42',
}

GALAXY_NAME_MAPPING: Dict[str, str] = {
    'Milky Way': 'Milky Way Galaxy',
    '银河系': 'Milky Way Galaxy',
    'Andromeda Galaxy': 'M31',
    '仙女座星系': 'M31',
    '仙女座大星系': 'M31',
    'Triangulum Galaxy': 'M33',
    '三角座星系': 'M33'
}


# ========== 支持的天体（用于升起落下计算） ==========
SUPPORTED_BODIES: List[str] = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn']


# ========== 默认观测位置 ==========
DEFAULT_LOCATION: tuple = (39.9, 116.4)  # 北京（纬度, 经度）


# ========== API配置 ==========
NASA_APOD_URL: str = "https://api.nasa.gov/planetary/apod"
NASA_NEO_URL: str = "https://api.nasa.gov/neo/rest/v1/feed"
AMAP_WEATHER_URL: str = "https://restapi.amap.com/v3/weather/weatherInfo"
AMAP_GEOCODE_URL: str = "https://restapi.amap.com/v3/geocode/regeo"
TAVILY_SEARCH_URL: str = "https://api.tavily.com/search"


# ========== NASA API限制 ==========
NASA_NEO_MAX_DAYS: int = 7  # NEO API最大查询天数范围


# ========== 行星可见性建议（按月份） ==========
PLANET_VISIBILITY_BY_MONTH: Dict[int, Dict[str, str]] = {
    # 春季 (3-5月)
    3: {'best': '土星', 'morning': '金星', 'evening': '木星'},
    4: {'best': '土星', 'morning': '金星', 'evening': '木星'},
    5: {'best': '土星', 'morning': '金星', 'evening': '木星'},
    
    # 夏季 (6-8月)
    6: {'best': '土星', 'evening': '火星', 'morning': '木星'},
    7: {'best': '土星', 'evening': '火星', 'morning': '木星'},
    8: {'best': '土星', 'evening': '火星', 'morning': '木星'},
    
    # 秋季 (9-11月)
    9: {'best': '木星', 'evening': '土星', 'evening2': '火星'},
    10: {'best': '木星', 'evening': '土星', 'evening2': '火星'},
    11: {'best': '木星', 'evening': '土星', 'evening2': '火星'},
    
    # 冬季 (12-2月)
    12: {'best': '木星', 'evening': '木星', 'late_night': '土星', 'morning': '金星'},
    1: {'best': '木星', 'evening': '木星', 'late_night': '土星', 'morning': '金星'},
    2: {'best': '木星', 'evening': '木星', 'late_night': '土星', 'morning': '金星'},
}


# ========== 月相阈值（度） ==========
MOON_PHASE_THRESHOLDS: List[tuple] = [
    (22.5, "🌑 新月", "月光微弱，适合观测深空天体"),
    (67.5, "🌒 娥眉月", "傍晚可见，适合观测"),
    (112.5, "🌓 上弦月", "下午至前半夜可见"),
    (157.5, "🌔 盈凸月", "傍晚至后半夜可见"),
    (202.5, "🌕 满月", "整夜可见，月光强，不适合深空观测"),
    (247.5, "🌖 亏凸月", "前半夜可见"),
    (292.5, "🌗 下弦月", "后半夜可见"),
    (337.5, "🌘 残月", "凌晨可见"),
]


# ========== 观测建议模板 ==========
OBSERVING_TIPS_TEMPLATES: Dict[str, str] = {
    'bad_weather': "天气现象不佳（雨雪雷雾霾等），不建议深空观测；可改观测月亮/行星或室内学习。",
    'good_weather': "若夜间少云、能见度好，可尝试行星/亮星团观测。",
    'high_humidity': "湿度偏高，易起雾/结露，建议准备除露带、镜头加热。",
    'high_wind': "风力偏大，三脚架需加重，长曝光拍摄成功率下降。",
    'new_moon': "**今晚特别适合深空观测！** 无月光干扰，可以挑战星系、星云。",
    'full_moon': "**满月光太强**，建议观测明亮的行星和双星。",
}
