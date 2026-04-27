from typing import Dict, List, Optional, Tuple

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


def _legacy_memory_env(name: str) -> str:
    return "S" + "TM_" + name


class APIConfig(BaseSettings):
    API_HOST: str = Field("0.0.0.0", description="FastAPI 服务监听地址")
    API_PORT: int = Field(8002, description="FastAPI 服务端口")
    MCP_PORT: int = Field(8001, description="MCP 服务端口（避免与 FastAPI 冲突）")
    MCP_SERVER_URL: str = Field(
        "http://localhost:8001/mcp", description="MCP Server URL，可通过环境变量覆盖"
    )
    MAX_UPLOAD_SIZE: int = Field(
        10 * 1024 * 1024, description="文件上传大小限制（字节），默认10MB"
    )
    ALLOWED_IMAGE_EXTENSIONS: set = Field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"},
        description="允许上传的图片文件扩展名白名单",
    )
    ALLOWED_AUDIO_EXTENSIONS: set = Field(
        default_factory=lambda: {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"},
        description="允许上传的音频文件扩展名白名单",
    )
    RATE_LIMIT_PER_MINUTE: int = Field(30, description="API请求限流：每分钟最大请求数")
    UPLOAD_RATE_LIMIT_PER_MINUTE: int = Field(
        10, description="文件上传限流：每分钟最大上传数"
    )
    SESSION_MAX_AGE_SECONDS: int = Field(3600, description="用户会话最大存活时间（秒）")
    SESSION_CLEANUP_INTERVAL_SECONDS: int = Field(300, description="会话清理间隔（秒）")
    PROMPT_TEMPLATE_PATH: str = Field(
        "config/prompts/main.txt", description="Prompt模板文件路径（相对于项目根目录）"
    )
    PROJECT_ROOT: str = Field("", description="项目根目录绝对路径（留空则自动检测）")

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


class AstronomyConfig(BaseSettings):
    EPHEMERIS_FILE: str = Field(
        "./data/ephemeris/de421.bsp", description="星历数据文件路径"
    )
    DEFAULT_LOCATION: Tuple[float, float] = Field(
        (39.9, 116.4), description="默认观测位置（纬度, 经度），默认为北京"
    )
    SUPPORTED_YEAR_RANGE: Tuple[int, int] = Field(
        (2026, 2030), description="天象数据支持的年份范围"
    )
    PLANET_MAPPING: Dict[str, int] = Field(
        default_factory=lambda: {
            "mercury": 199,
            "venus": 299,
            "mars": 499,
            "jupiter": 5,
            "saturn": 6,
            "uranus": 7,
            "neptune": 8,
        },
        description="行星名称到星历 ID 的映射",
    )
    VALID_PLANETS: set = Field(
        default_factory=lambda: {
            "mercury",
            "venus",
            "mars",
            "jupiter",
            "saturn",
            "uranus",
            "neptune",
        },
        description="有效的行星名称集合",
    )
    PLANET_NAMES_CN: Dict[str, str] = Field(
        default_factory=lambda: {
            "mercury": "水星",
            "venus": "金星",
            "mars": "火星",
            "jupiter": "木星",
            "saturn": "土星",
            "uranus": "天王星",
            "neptune": "海王星",
        },
        description="行星英文名到中文名的映射",
    )
    PLANET_MAX_MAGNITUDE: Dict[str, float] = Field(
        default_factory=lambda: {
            "mercury": -1.9,
            "venus": -4.6,
            "mars": -2.0,
            "jupiter": -2.7,
            "saturn": -0.3,
        },
        description="行星最大视星等",
    )
    SUPPORTED_BODIES: List[str] = Field(
        default_factory=lambda: [
            "sun",
            "moon",
            "mercury",
            "venus",
            "mars",
            "jupiter",
            "saturn",
        ],
        description="支持升起落下计算的天体列表",
    )
    CELESTIAL_NAME_MAPPING: Dict[str, str] = Field(
        default_factory=lambda: {
            "Andromeda Galaxy": "M31",
            "仙女座星系": "M31",
            "仙女座大星系": "M31",
            "Milky Way": "Milky Way Galaxy",
            "银河系": "Milky Way Galaxy",
            "Triangulum Galaxy": "M33",
            "三角座星系": "M33",
            "Sirius": "Sirius",
            "天狼星": "Sirius",
            "Orion Nebula": "M42",
            "猎户座星云": "M42",
            "猎户座大星云": "M42",
        },
        description="天体别名到标准名的映射",
    )
    GALAXY_NAME_MAPPING: Dict[str, str] = Field(
        default_factory=lambda: {
            "Milky Way": "Milky Way Galaxy",
            "银河系": "Milky Way Galaxy",
            "Andromeda Galaxy": "M31",
            "仙女座星系": "M31",
            "仙女座大星系": "M31",
            "Triangulum Galaxy": "M33",
            "三角座星系": "M33",
        },
        description="星系别名到标准名的映射",
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
        description="月相角度阈值及对应描述",
    )
    OBSERVING_TIPS_TEMPLATES: Dict[str, str] = Field(
        default_factory=lambda: {
            "bad_weather": "天气现象不佳（雨雪雷雾霾等），不建议深空观测；可改观测月亮/行星或室内学习。",
            "good_weather": "若夜间少云、能见度好，可尝试行星/亮星团观测。",
            "high_humidity": "湿度偏高，易起雾/结露，建议准备除露带、镜头加热。",
            "high_wind": "风力偏大，三脚架需加重，长曝光拍摄成功率下降。",
            "new_moon": "**今晚特别适合深空观测！** 无月光干扰，可以挑战星系、星云。",
            "full_moon": "**满月光太强**，建议观测明亮的行星和双星。",
        },
        description="天文观测条件建议模板",
    )
    NASA_API_KEY: Optional[str] = Field(None, description="NASA API 密钥")
    NASA_BASE_URL: str = Field("https://api.nasa.gov", description="NASA API 基础 URL")
    NASA_APOD_URL: str = Field(
        "https://api.nasa.gov/planetary/apod", description="NASA APOD 接口地址"
    )
    NASA_NEO_URL: str = Field(
        "https://api.nasa.gov/neo/rest/v1/feed", description="NASA NEO 接口地址"
    )
    NASA_NEO_MAX_DAYS: int = Field(7, description="NEO API 最大查询天数限制")
    AMAP_API_KEY: Optional[str] = Field(None, description="高德地图 API Key")
    AMAP_DEFAULT_CITY: str = Field("北京", description="默认城市")
    AMAP_WEATHER_URL: str = Field(
        "https://restapi.amap.com/v3/weather/weatherInfo",
        description="高德天气查询接口地址",
    )
    AMAP_GEOCODE_URL: str = Field(
        "https://restapi.amap.com/v3/geocode/regeo",
        description="高德逆地理编码接口地址",
    )
    TAVILY_API_KEY: Optional[str] = Field(None, description="Tavily 搜索 API Key")
    TAVILY_SEARCH_URL: str = Field(
        "https://api.tavily.com/search", description="Tavily 搜索接口地址"
    )
    ASTRONOMY_DATA_DIR: str = Field(
        "config/astronomy", description="天文事件数据目录（相对于项目根目录）"
    )

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


class ModelConfig(BaseSettings):
    DASHSCOPE_API_KEY: Optional[str] = Field(None, description="千问模型 API Key")
    DEFAULT_LLM_PROVIDER: str = Field("dashscope", description="默认主模型提供商")
    MODEL_NAME: str = Field("qwen-max", description="主模型名称")
    SMALL_MODEL_PROVIDER: str = Field(
        "dashscope", description="轻量模型提供商，用于路由/轻量总结等场景"
    )
    SMALL_MODEL_NAME: str = Field("qwen-plus", description="轻量模型名称")
    SYNTHESIS_MODEL_TIER: str = Field(
        "main", description="答案合成器使用的模型层级：main|small"
    )
    OPENAI_COMPATIBLE_BASE_URL: str = Field(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="DashScope OpenAI兼容接口基础地址",
    )
    GLM_API_KEY: Optional[str] = Field(None, description="GLM API Key")
    GLM_API_BASE_URL: str = Field(
        "https://open.bigmodel.cn/api/paas/v4",
        description="GLM OpenAI兼容接口基础地址",
    )
    MINIMAX_API_KEY: Optional[str] = Field(None, description="MiniMax API Key")
    MINIMAX_API_BASE_URL: str = Field(
        "https://api.minimaxi.com/v1",
        description="MiniMax OpenAI兼容接口基础地址",
    )
    EMBEDDING_MODEL_NAME: str = Field("text-embedding-v2", description="嵌入模型名称")
    VISION_MODEL_NAME: str = Field("qwen-vl-plus", description="视觉模型名称")
    SPEECH_MODEL_NAME: str = Field("paraformer-realtime-v2", description="语音模型名称")

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


class MemoryConfig(BaseSettings):
    MEMORY_SIZE: int = Field(15, description="会话记忆最大消息数")
    MEMORY_WINDOW: int = Field(8, description="会话记忆窗口大小")
    LONG_TERM_MEMORY_PATH: str = Field(
        "./memory/long_term_memory/user_profiles.sqlite",
        description="长期记忆 SQLite 数据库路径",
    )
    DEFAULT_USER_ID: str = Field("anonymous", description="默认用户 ID")

    MEMORY_CONTEXT_MAX_TOKENS: int = Field(
        4000,
        validation_alias=AliasChoices(
            "MEMORY_CONTEXT_MAX_TOKENS", _legacy_memory_env("CONTEXT_MAX_TOKENS")
        ),
        description="会话记忆上下文最大 token 数",
    )
    MEMORY_SUMMARY_MAX_TOKENS: int = Field(
        500,
        validation_alias=AliasChoices(
            "MEMORY_SUMMARY_MAX_TOKENS", _legacy_memory_env("SUMMARY_MAX_TOKENS")
        ),
        description="记忆摘要最大 token 数",
    )
    MEMORY_SUMMARY_TRIGGER_MESSAGES: int = Field(
        10,
        validation_alias=AliasChoices(
            "MEMORY_SUMMARY_TRIGGER_MESSAGES",
            _legacy_memory_env("SUMMARY_TRIGGER_MESSAGES"),
        ),
        description="触发摘要的消息数阈值",
    )
    MEMORY_SUMMARY_TRIGGER_TOKENS: int = Field(
        3000,
        validation_alias=AliasChoices(
            "MEMORY_SUMMARY_TRIGGER_TOKENS",
            _legacy_memory_env("SUMMARY_TRIGGER_TOKENS"),
        ),
        description="触发摘要的 token 数阈值",
    )
    MEMORY_PERSISTENCE_ENABLED: bool = Field(
        True,
        validation_alias=AliasChoices(
            "MEMORY_PERSISTENCE_ENABLED", _legacy_memory_env("PERSISTENCE_ENABLED")
        ),
        description="是否启用记忆持久化",
    )
    MEMORY_PERSISTENCE_PATH: str = Field(
        "./memory/sessions.sqlite",
        validation_alias=AliasChoices(
            "MEMORY_PERSISTENCE_PATH", _legacy_memory_env("PERSISTENCE_PATH")
        ),
        description="记忆事件与工件 SQLite 数据库路径",
    )
    MEMORY_IMPORTANCE_HIGH_ROLES: set = Field(
        default_factory=lambda: {"user", "system"},
        validation_alias=AliasChoices(
            "MEMORY_IMPORTANCE_HIGH_ROLES",
            _legacy_memory_env("IMPORTANCE_HIGH_ROLES"),
        ),
        description="高重要性角色集合",
    )
    MEMORY_TOOL_RESULT_MAX_LENGTH: int = Field(
        500,
        validation_alias=AliasChoices(
            "MEMORY_TOOL_RESULT_MAX_LENGTH",
            _legacy_memory_env("TOOL_RESULT_MAX_LENGTH"),
        ),
        description="工具结果摘要最大字符数",
    )

    LTM_MIN_CONFIDENCE_TO_STORE: float = Field(
        0.3, description="长期记忆最低存储置信度"
    )
    LTM_DEDUP_SIMILARITY_THRESHOLD: float = Field(0.85, description="去重相似度阈值")
    LTM_CANDIDATE_OCCURRENCE_THRESHOLD: int = Field(
        2, description="候选记忆提升出现次数阈值"
    )
    LTM_CANDIDATE_CONFIDENCE_THRESHOLD: float = Field(
        0.6, description="候选记忆提升置信度阈值"
    )
    LTM_MAX_PROMPT_TOKENS: int = Field(800, description="长期记忆注入Prompt最大token数")
    LTM_MAX_MEMORIES_IN_PROMPT: int = Field(15, description="Prompt注入最大记忆条数")
    LTM_AUTO_BACKUP_INTERVAL_HOURS: int = Field(24, description="自动备份间隔(小时)")
    LTM_DEFAULT_EXPIRY_DAYS: int = Field(180, description="默认记忆过期天数")
    LTM_CONSTRAINT_EXPIRY_DAYS: int = Field(365, description="约束记忆过期天数")
    LTM_FACT_EXPIRY_DAYS: int = Field(730, description="事实记忆过期天数")
    LTM_ARCHIVE_AFTER_DAYS_UNUSED: int = Field(90, description="未使用记忆归档天数")

    RAG_ENABLED: bool = Field(True, description="是否启用 RAG 检索")
    VECTOR_DB_PATH: str = Field("./vector_db", description="向量数据库存储路径")

    RERANK_MODEL_NAME: str = Field("qwen3-rerank", description="Rerank 模型名称")
    RERANK_ENABLED: bool = Field(True, description="是否启用 Rerank 重排序")
    RERANK_TOP_N: int = Field(3, description="Rerank 返回的最大文档数")
    RRF_K: int = Field(60, description="RRF 算法常数 K，控制排名衰减速度")
    RAG_VECTOR_WEIGHT: float = Field(0.5, description="混合检索中向量检索的权重")
    RAG_BM25_WEIGHT: float = Field(0.5, description="混合检索中 BM25 检索的权重")
    RAG_ENTITY_WEIGHT: float = Field(0.3, description="混合检索中天文实体检索的权重")
    RAG_RETRIEVAL_CANDIDATES: int = Field(
        20, description="混合检索候选文档数（RRF 前）"
    )
    RAG_CACHE_ENABLED: bool = Field(True, description="是否启用检索结果缓存")
    RAG_CACHE_TTL: int = Field(300, description="检索结果缓存 TTL（秒）")
    RAG_CACHE_MAX_SIZE: int = Field(256, description="检索结果缓存最大条目数")

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


class AgentGovernanceConfig(BaseSettings):
    AGENT_MODE: str = Field(
        "hybrid",
        description="Agent 执行模式：react|hybrid|planned",
    )
    ENABLE_STRUCTURED_SKILL_RESULT: bool = Field(
        False,
        description="是否启用结构化 SkillResult 契约",
    )
    ENABLE_PLANNER: bool = Field(
        False,
        description="是否启用 Planner 主链路",
    )
    ENABLE_REACT_FALLBACK: bool = Field(
        True,
        description="当主链路不可用时是否允许回退到 ReAct",
    )
    PHASE0_BENCHMARK_PATH: str = Field(
        "config/benchmarks/agent_phase0_benchmark.json",
        description="阶段0基准数据集路径（相对于项目根目录）",
    )
    AGENT_MAX_LLM_CALLS: int = Field(4, description="单请求允许的最大 LLM 调用次数")
    AGENT_MAX_TOOL_CALLS: int = Field(6, description="单请求允许的最大工具调用次数")
    AGENT_MAX_TOTAL_TIME_MS: int = Field(
        60000, description="单请求最大执行时间（毫秒）"
    )
    AGENT_MAX_PARALLELISM: int = Field(2, description="单请求最大并行度")
    AGENT_MAX_CONTEXT_CHARS: int = Field(
        6000, description="单请求注入到模型的最大上下文字符数"
    )
    AGENT_AUDIT_ENABLED: bool = Field(True, description="是否开启请求审计日志")
    AGENT_AUDIT_LOG_PATH: str = Field(
        "logs/agent_audit/requests.jsonl",
        description="请求审计日志路径（相对于项目根目录）",
    )
    ROUTER_POLICY_VERSION: str = Field("router_v1", description="路由策略版本")
    PLANNER_VERSION: str = Field("planner_v2", description="Planner 版本")
    SCHEMA_VERSION: str = Field("schema_v2", description="输出契约版本")
    SYNTH_PROMPT_VERSION: str = Field(
        "synth_prompt_v2", description="答案合成 Prompt 版本"
    )
    FALLBACK_POLICY_VERSION: str = Field(
        "fallback_v2", description="Fallback 策略版本"
    )
    BUDGET_POLICY_VERSION: str = Field(
        "budget_v1", description="预算策略版本"
    )
    MODEL_POLICY_VERSION: str = Field("model_policy_v1", description="模型策略版本")

    # ── 重构阶段 feature flags（Phase 8 已全部开启，旧路径保留为 flag=False 兼容层）──
    # Phase 1: TaskProfile 任务画像，收敛目标：替代 task_type 字段
    ENABLE_TASK_PROFILE: bool = Field(False, description="[重构flag] 启用 TaskProfile 任务画像")
    # Phase 2: ExecutionContext 统一执行上下文，收敛目标：替代散落的 chat_history/user_profile 参数
    ENABLE_EXECUTION_CONTEXT: bool = Field(False, description="[重构flag] 启用 ExecutionContext 统一上下文")
    # Phase 2: ExecutionDecision 路由决策对象，收敛目标：替代 RouteDecision
    ENABLE_EXECUTION_DECISION: bool = Field(False, description="[重构flag] 启用 ExecutionDecision 路由决策对象")
    # Phase 4/8: UnifiedExecutionEngine 统一执行引擎，Phase 8 起默认开启
    # 收敛计划：下一轮可删除 flag，直接移除旧 TaskOrchestrator 分支
    ENABLE_UNIFIED_EXECUTION_ENGINE: bool = Field(True, description="[重构flag] 启用 UnifiedExecutionEngine（Phase 8 默认开启）")
    # Phase 5/8: WorkflowGraph DAG 执行图，Phase 8 起默认开启
    # 收敛计划：下一轮可删除 flag，移除 StepExecutor 分支
    ENABLE_WORKFLOW_GRAPH: bool = Field(True, description="[重构flag] 启用 WorkflowGraph DAG 执行图（Phase 8 默认开启）")
    # Phase 7/8: UnifiedExecutionTrace 统一执行追踪，Phase 8 起默认开启
    # 收敛计划：下一轮升级 FinalResponse.execution_trace 为 List[ExecutionTraceEntry]
    ENABLE_UNIFIED_EXECUTION_TRACE: bool = Field(True, description="[重构flag] 启用 UnifiedExecutionTrace（Phase 8 默认开启）")
    # Phase 7/8: UnifiedExecutionEvents 统一事件模型，Phase 8 起默认开启
    # 收敛计划：下一轮将 ExecutionEvent.to_frontend_type() 映射迁入 FrontendJsonEventAdapter
    ENABLE_UNIFIED_EXECUTION_EVENTS: bool = Field(True, description="[重构flag] 启用 UnifiedExecutionEvents（Phase 8 默认开启）")

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


_SUB_CONFIG_FIELDS = {
    "api": APIConfig,
    "astronomy": AstronomyConfig,
    "model": ModelConfig,
    "memory": MemoryConfig,
    "agent_governance": AgentGovernanceConfig,
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
    agent_governance: AgentGovernanceConfig = Field(
        default_factory=AgentGovernanceConfig
    )

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
