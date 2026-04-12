from pydantic_settings import BaseSettings
from typing import Dict, List, Optional, Tuple
from pydantic import Field


class APIConfig(BaseSettings):
    API_HOST: str = Field("0.0.0.0", description="FastAPI 服务监听地址")
    API_PORT: int = Field(8000, description="FastAPI 服务端口")
    MCP_PORT: int = Field(8001, description="MCP 服务端口（避免与 FastAPI 冲突）")
    MCP_SERVER_URL: str = Field(
        "http://localhost:8001/mcp",
        description="MCP Server URL，可通过环境变量覆盖"
    )
    MAX_UPLOAD_SIZE: int = Field(10 * 1024 * 1024, description="文件上传大小限制（字节），默认10MB")
    ALLOWED_IMAGE_EXTENSIONS: set = Field(
        default_factory=lambda: {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'},
        description="允许上传的图片文件扩展名白名单"
    )
    ALLOWED_AUDIO_EXTENSIONS: set = Field(
        default_factory=lambda: {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'},
        description="允许上传的音频文件扩展名白名单"
    )
    RATE_LIMIT_PER_MINUTE: int = Field(30, description="API请求限流：每分钟最大请求数")
    UPLOAD_RATE_LIMIT_PER_MINUTE: int = Field(10, description="文件上传限流：每分钟最大上传数")
    SESSION_MAX_AGE_SECONDS: int = Field(3600, description="用户会话最大存活时间（秒）")
    SESSION_CLEANUP_INTERVAL_SECONDS: int = Field(300, description="会话清理间隔（秒）")
    PROMPT_TEMPLATE_PATH: str = Field(
        "config/prompts/main.txt",
        description="Prompt模板文件路径（相对于项目根目录）"
    )
    PROJECT_ROOT: str = Field(
        "",
        description="项目根目录绝对路径（留空则自动检测）"
    )

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


class AstronomyConfig(BaseSettings):
    EPHEMERIS_FILE: str = Field("./data/ephemeris/de421.bsp", description="星历数据文件路径")
    DEFAULT_LOCATION: Tuple[float, float] = Field(
        (39.9, 116.4),
        description="默认观测位置（纬度, 经度），默认为北京"
    )
    SUPPORTED_YEAR_RANGE: Tuple[int, int] = Field(
        (2026, 2030),
        description="天象数据支持的年份范围"
    )
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
    NASA_API_KEY: Optional[str] = Field(None, description="NASA API 密钥")
    NASA_BASE_URL: str = Field("https://api.nasa.gov", description="NASA API 基础 URL")
    NASA_APOD_URL: str = Field("https://api.nasa.gov/planetary/apod", description="NASA APOD 接口地址")
    NASA_NEO_URL: str = Field("https://api.nasa.gov/neo/rest/v1/feed", description="NASA NEO 接口地址")
    NASA_NEO_MAX_DAYS: int = Field(7, description="NEO API 最大查询天数限制")
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
    TAVILY_API_KEY: Optional[str] = Field(None, description="Tavily 搜索 API Key")
    TAVILY_SEARCH_URL: str = Field(
        "https://api.tavily.com/search",
        description="Tavily 搜索接口地址"
    )
    ASTRONOMY_DATA_DIR: str = Field(
        "config/astronomy",
        description="天文事件数据目录（相对于项目根目录）"
    )

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


class ModelConfig(BaseSettings):
    DASHSCOPE_API_KEY: Optional[str] = Field(None, description="千问模型 API Key")
    MODEL_NAME: str = Field("qwen-max", description="主模型名称")
    EMBEDDING_MODEL_NAME: str = Field("text-embedding-v2", description="嵌入模型名称")
    VISION_MODEL_NAME: str = Field("qwen-vl-plus", description="视觉模型名称")
    SPEECH_MODEL_NAME: str = Field("paraformer-realtime-v2", description="语音模型名称")

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


class MemoryConfig(BaseSettings):
    MEMORY_SIZE: int = Field(15, description="短期记忆最大消息数")
    MEMORY_WINDOW: int = Field(8, description="短期记忆窗口大小")
    LONG_TERM_MEMORY_PATH: str = Field(
        "./memory/long_term_memory/user_profiles.sqlite",
        description="长期记忆 SQLite 数据库路径"
    )
    DEFAULT_USER_ID: str = Field("anonymous", description="默认用户 ID")
    RAG_ENABLED: bool = Field(True, description="是否启用 RAG 检索")
    VECTOR_DB_PATH: str = Field("./vector_db", description="向量数据库存储路径")

    RERANK_MODEL_NAME: str = Field("qwen3-rerank", description="Rerank 模型名称")
    RERANK_ENABLED: bool = Field(True, description="是否启用 Rerank 重排序")
    RERANK_TOP_N: int = Field(3, description="Rerank 返回的最大文档数")
    RRF_K: int = Field(60, description="RRF 算法常数 K，控制排名衰减速度")
    RAG_VECTOR_WEIGHT: float = Field(0.5, description="混合检索中向量检索的权重")
    RAG_BM25_WEIGHT: float = Field(0.5, description="混合检索中 BM25 检索的权重")
    RAG_RETRIEVAL_CANDIDATES: int = Field(20, description="混合检索候选文档数（RRF 前）")
    RAG_CACHE_ENABLED: bool = Field(True, description="是否启用检索结果缓存")
    RAG_CACHE_TTL: int = Field(300, description="检索结果缓存 TTL（秒）")
    RAG_CACHE_MAX_SIZE: int = Field(256, description="检索结果缓存最大条目数")

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


_SUB_CONFIG_FIELDS = {
    "api": APIConfig,
    "astronomy": AstronomyConfig,
    "model": ModelConfig,
    "memory": MemoryConfig,
}

_DELEGATION_MAP: Dict[str, str] = {}
for _group_name, _config_cls in _SUB_CONFIG_FIELDS.items():
    for _field_name in _config_cls.model_fields:
        _DELEGATION_MAP[_field_name] = _group_name


class Settings(BaseSettings):
    api: APIConfig = Field(default_factory=APIConfig)
    astronomy: AstronomyConfig = Field(default_factory=AstronomyConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",
    }

    def __getattr__(self, name: str):
        group_name = _DELEGATION_MAP.get(name)
        if group_name is not None:
            return getattr(getattr(self, group_name), name)
        raise AttributeError(f"'Settings' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        group_name = _DELEGATION_MAP.get(name)
        if group_name is not None:
            setattr(getattr(self, group_name), name, value)
        else:
            super().__setattr__(name, value)


def _resolve_project_root() -> str:
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    return str(root)


settings = Settings()
if not settings.PROJECT_ROOT:
    settings.PROJECT_ROOT = _resolve_project_root()


def resolve_path(relative_path: str) -> str:
    from pathlib import Path
    if Path(relative_path).is_absolute():
        return relative_path
    return str(Path(settings.PROJECT_ROOT) / relative_path)
