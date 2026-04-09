from pydantic_settings import BaseSettings
from typing import Dict, List, Optional, Tuple
from pydantic import Field


class Settings(BaseSettings):
    """项目统一配置管理"""
    
    # ========== 千问模型配置 ==========
    DASHSCOPE_API_KEY: Optional[str] = Field(None, description="千问模型 API Key")
    MODEL_NAME: str = Field("qwen-max", description="主模型名称")
    EMBEDDING_MODEL_NAME: str = Field("text-embedding-v2", description="嵌入模型名称")
    VISION_MODEL_NAME: str = Field("qwen-vl-plus", description="视觉模型名称")
    SPEECH_MODEL_NAME: str = Field("paraformer-realtime-v2", description="语音模型名称")
    
    # ========== RAG 配置 ==========
    RAG_ENABLED: bool = Field(True, description="是否启用 RAG 检索")
    VECTOR_DB_PATH: str = Field("./vector_db", description="向量数据库存储路径")
    
    # ========== 记忆配置 ==========
    MEMORY_SIZE: int = Field(15, description="短期记忆最大消息数")
    MEMORY_WINDOW: int = Field(8, description="短期记忆窗口大小")
    LONG_TERM_MEMORY_PATH: str = Field(
        "./memory/long_term_memory/user_profiles.sqlite",
        description="长期记忆 SQLite 数据库路径"
    )
    DEFAULT_USER_ID: str = Field("anonymous", description="默认用户 ID")
    
    # ========== API 服务配置 ==========
    API_HOST: str = Field("0.0.0.0", description="FastAPI 服务监听地址")
    API_PORT: int = Field(8000, description="FastAPI 服务端口")
    MCP_PORT: int = Field(8001, description="MCP 服务端口（避免与 FastAPI 冲突）")
    MCP_SERVER_URL: str = Field(
        "http://localhost:8001/mcp",
        description="MCP Server URL，可通过环境变量覆盖"
    )
    
    # ========== NASA API 配置 ==========
    NASA_API_KEY: Optional[str] = Field(None, description="NASA API 密钥")
    NASA_BASE_URL: str = Field("https://api.nasa.gov", description="NASA API 基础 URL")
    NASA_APOD_URL: str = Field("https://api.nasa.gov/planetary/apod", description="NASA APOD 接口地址")
    NASA_NEO_URL: str = Field("https://api.nasa.gov/neo/rest/v1/feed", description="NASA NEO 接口地址")
    NASA_NEO_MAX_DAYS: int = Field(7, description="NEO API 最大查询天数限制")
    
    # ========== 高德地图 API 配置 ==========
    AMAP_API_KEY: Optional[str] = Field(None, description="高德地图 API Key")
    AMAP_DEFAULT_CITY: str = Field("北京", description="默认城市")
    AMAP_WEATHER_URL: str = Field(
        "https://restapi.amap.com/v3/weather/weatherInfo",
        description="高德天气查询接口地址"
    )
    AMAP_GEOCODE_URL: str = Field(
        "https://restapi.amap.com/v3/geocode/regeo",
        description="高德逆地理编码接口地址"
    )
    
    # ========== Tavily 联网搜索 API ==========
    TAVILY_API_KEY: Optional[str] = Field(None, description="Tavily 搜索 API Key")
    TAVILY_SEARCH_URL: str = Field(
        "https://api.tavily.com/search",
        description="Tavily 搜索接口地址"
    )
    
    # ========== 天文数据配置 ==========
    EPHEMERIS_FILE: str = Field("./data/ephemeris/de421.bsp", description="星历数据文件路径")
    DEFAULT_LOCATION: Tuple[float, float] = Field(
        (39.9, 116.4),
        description="默认观测位置（纬度, 经度），默认为北京"
    )
    SUPPORTED_YEAR_RANGE: Tuple[int, int] = Field(
        (2026, 2030),
        description="天象数据支持的年份范围"
    )
    
    # ========== 行星相关常量（可配置但通常使用默认值） ==========
    PLANET_MAPPING: Dict[str, int] = Field(
        default_factory=lambda: {
            'mercury': 199,
            'venus': 299,
            'mars': 499,
            'jupiter': 5,
            'saturn': 6,
            'uranus': 7,
            'neptune': 8
        },
        description="行星名称到星历 ID 的映射"
    )
    
    VALID_PLANETS: set = Field(
        default_factory=lambda: {'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'},
        description="有效的行星名称集合"
    )
    
    PLANET_NAMES_CN: Dict[str, str] = Field(
        default_factory=lambda: {
            'mercury': '水星',
            'venus': '金星',
            'mars': '火星',
            'jupiter': '木星',
            'saturn': '土星',
            'uranus': '天王星',
            'neptune': '海王星'
        },
        description="行星英文名到中文名的映射"
    )
    
    PLANET_MAX_MAGNITUDE: Dict[str, float] = Field(
        default_factory=lambda: {
            'mercury': -1.9,
            'venus': -4.6,
            'mars': -2.0,
            'jupiter': -2.7,
            'saturn': -0.3,
        },
        description="行星最大视星等"
    )
    
    SUPPORTED_BODIES: List[str] = Field(
        default_factory=lambda: ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn'],
        description="支持升起落下计算的天体列表"
    )
    
    # ========== 天体名称映射 ==========
    CELESTIAL_NAME_MAPPING: Dict[str, str] = Field(
        default_factory=lambda: {
            'Andromeda Galaxy': 'M31',
            '仙女座星系': 'M31',
            '仙女座大星系': 'M31',
            'Milky Way': 'Milky Way Galaxy',
            '银河系': 'Milky Way Galaxy',
            'Triangulum Galaxy': 'M33',
            '三角座星系': 'M33',
            'Sirius': 'Sirius',
            '天狼星': 'Sirius',
            'Orion Nebula': 'M42',
            '猎户座星云': 'M42',
            '猎户座大星云': 'M42',
        },
        description="天体别名到标准名的映射"
    )
    
    GALAXY_NAME_MAPPING: Dict[str, str] = Field(
        default_factory=lambda: {
            'Milky Way': 'Milky Way Galaxy',
            '银河系': 'Milky Way Galaxy',
            'Andromeda Galaxy': 'M31',
            '仙女座星系': 'M31',
            '仙女座大星系': 'M31',
            'Triangulum Galaxy': 'M33',
            '三角座星系': 'M33'
        },
        description="星系别名到标准名的映射"
    )
    
    # ========== 行星可见性建议（按月份） ==========
    PLANET_VISIBILITY_BY_MONTH: Dict[int, Dict[str, str]] = Field(
        default_factory=lambda: {
            3: {'best': '土星', 'morning': '金星', 'evening': '木星'},
            4: {'best': '土星', 'morning': '金星', 'evening': '木星'},
            5: {'best': '土星', 'morning': '金星', 'evening': '木星'},
            6: {'best': '土星', 'evening': '火星', 'morning': '木星'},
            7: {'best': '土星', 'evening': '火星', 'morning': '木星'},
            8: {'best': '土星', 'evening': '火星', 'morning': '木星'},
            9: {'best': '木星', 'evening': '土星', 'evening2': '火星'},
            10: {'best': '木星', 'evening': '土星', 'evening2': '火星'},
            11: {'best': '木星', 'evening': '土星', 'evening2': '火星'},
            12: {'best': '木星', 'evening': '木星', 'late_night': '土星', 'morning': '金星'},
            1: {'best': '木星', 'evening': '木星', 'late_night': '土星', 'morning': '金星'},
            2: {'best': '木星', 'evening': '木星', 'late_night': '土星', 'morning': '金星'},
        },
        description="各月份最佳观测行星建议"
    )
    
    # ========== 月相阈值（度） ==========
    MOON_PHASE_THRESHOLDS: List[tuple] = Field(
        default_factory=lambda: [
            (22.5, "🌑 新月", "月光微弱，适合观测深空天体"),
            (67.5, "🌒 娥眉月", "傍晚可见，适合观测"),
            (112.5, "🌓 上弦月", "下午至前半夜可见"),
            (157.5, "🌔 盈凸月", "傍晚至后半夜可见"),
            (202.5, "🌕 满月", "整夜可见，月光强，不适合深空观测"),
            (247.5, "🌖 亏凸月", "前半夜可见"),
            (292.5, "🌗 下弦月", "后半夜可见"),
            (337.5, "🌘 残月", "凌晨可见"),
        ],
        description="月相角度阈值及对应描述"
    )
    
    # ========== 观测建议模板 ==========
    OBSERVING_TIPS_TEMPLATES: Dict[str, str] = Field(
        default_factory=lambda: {
            'bad_weather': "天气现象不佳（雨雪雷雾霾等），不建议深空观测；可改观测月亮/行星或室内学习。",
            'good_weather': "若夜间少云、能见度好，可尝试行星/亮星团观测。",
            'high_humidity': "湿度偏高，易起雾/结露，建议准备除露带、镜头加热。",
            'high_wind': "风力偏大，三脚架需加重，长曝光拍摄成功率下降。",
            'new_moon': "**今晚特别适合深空观测！** 无月光干扰，可以挑战星系、星云。",
            'full_moon': "**满月光太强**，建议观测明亮的行星和双星。",
        },
        description="天文观测条件建议模板"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# 创建全局配置实例
settings = Settings()
