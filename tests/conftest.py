import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()


@pytest.fixture
def mock_settings():
    with patch("src.core.config.Settings") as MockSettings:
        instance = MagicMock()
        instance.DASHSCOPE_API_KEY = "test-dashscope-key"
        instance.MODEL_NAME = "qwen-max"
        instance.EMBEDDING_MODEL_NAME = "text-embedding-v2"
        instance.VISION_MODEL_NAME = "qwen-vl-plus"
        instance.SPEECH_MODEL_NAME = "paraformer-realtime-v2"
        instance.RAG_ENABLED = False
        instance.VECTOR_DB_PATH = "/tmp/test_vector_db"
        instance.RERANK_MODEL_NAME = "qwen3-rerank"
        instance.RERANK_ENABLED = False
        instance.RERANK_TOP_N = 3
        instance.RRF_K = 60
        instance.RAG_VECTOR_WEIGHT = 0.5
        instance.RAG_BM25_WEIGHT = 0.5
        instance.RAG_ENTITY_WEIGHT = 0.3
        instance.RAG_RETRIEVAL_CANDIDATES = 20
        instance.RAG_CACHE_ENABLED = False
        instance.RAG_CACHE_TTL = 300
        instance.RAG_CACHE_MAX_SIZE = 256
        instance.MEMORY_SIZE = 15
        instance.MEMORY_WINDOW = 8
        instance.LONG_TERM_MEMORY_PATH = "/tmp/test_memory/user_profiles.sqlite"
        instance.DEFAULT_USER_ID = "test_user"
        instance.STM_CONTEXT_MAX_TOKENS = 4000
        instance.STM_SUMMARY_MAX_TOKENS = 500
        instance.STM_SUMMARY_TRIGGER_MESSAGES = 100
        instance.STM_SUMMARY_TRIGGER_TOKENS = 100000
        instance.STM_PERSISTENCE_ENABLED = False
        instance.STM_PERSISTENCE_PATH = "/tmp/test_memory/sessions.sqlite"
        instance.STM_IMPORTANCE_HIGH_ROLES = {"user", "system"}
        instance.STM_TOOL_RESULT_MAX_LENGTH = 500
        instance.API_HOST = "0.0.0.0"
        instance.API_PORT = 8000
        instance.MCP_PORT = 8001
        instance.NASA_API_KEY = "test-nasa-key"
        instance.NASA_BASE_URL = "https://api.nasa.gov"
        instance.NASA_APOD_URL = "https://api.nasa.gov/planetary/apod"
        instance.NASA_NEO_URL = "https://api.nasa.gov/neo/rest/v1/feed"
        instance.NASA_NEO_MAX_DAYS = 7
        instance.AMAP_API_KEY = "test-amap-key"
        instance.AMAP_DEFAULT_CITY = "北京"
        instance.AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
        instance.AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/regeo"
        instance.TAVILY_API_KEY = "test-tavily-key"
        instance.TAVILY_SEARCH_URL = "https://api.tavily.com/search"
        instance.EPHEMERIS_FILE = "de421.bsp"
        instance.DEFAULT_LOCATION = (39.9, 116.4)
        instance.SUPPORTED_YEAR_RANGE = (2026, 2030)
        instance.PLANET_MAPPING = {
            "mercury": 199,
            "venus": 299,
            "mars": 499,
            "jupiter": 5,
            "saturn": 6,
            "uranus": 7,
            "neptune": 8,
        }
        instance.VALID_PLANETS = {
            "mercury",
            "venus",
            "mars",
            "jupiter",
            "saturn",
            "uranus",
            "neptune",
        }
        instance.PLANET_NAMES_CN = {
            "mercury": "水星",
            "venus": "金星",
            "mars": "火星",
            "jupiter": "木星",
            "saturn": "土星",
            "uranus": "天王星",
            "neptune": "海王星",
        }
        instance.PLANET_MAX_MAGNITUDE = {
            "mercury": -1.9,
            "venus": -4.6,
            "mars": -2.0,
            "jupiter": -2.7,
            "saturn": -0.3,
        }
        instance.SUPPORTED_BODIES = [
            "sun",
            "moon",
            "mercury",
            "venus",
            "mars",
            "jupiter",
            "saturn",
        ]
        instance.CELESTIAL_NAME_MAPPING = {
            "Andromeda Galaxy": "M31",
            "仙女座星系": "M31",
            "Milky Way": "Milky Way Galaxy",
            "银河系": "Milky Way Galaxy",
        }
        instance.GALAXY_NAME_MAPPING = {
            "Milky Way": "Milky Way Galaxy",
            "银河系": "Milky Way Galaxy",
            "Andromeda Galaxy": "M31",
            "仙女座星系": "M31",
        }
        instance.MOON_PHASE_THRESHOLDS = [
            (22.5, "🌑 新月", "月光微弱，适合观测深空天体"),
            (67.5, "🌒 娥眉月", "傍晚可见，适合观测"),
            (112.5, "🌓 上弦月", "下午至前半夜可见"),
            (157.5, "🌔 盈凸月", "傍晚至后半夜可见"),
            (202.5, "🌕 满月", "整夜可见，月光强，不适合深空观测"),
            (247.5, "🌖 亏凸月", "前半夜可见"),
            (292.5, "🌗 下弦月", "后半夜可见"),
            (337.5, "🌘 残月", "凌晨可见"),
        ]
        instance.OBSERVING_TIPS_TEMPLATES = {
            "bad_weather": "天气现象不佳，不建议深空观测。",
            "good_weather": "若夜间少云，可尝试行星/亮星团观测。",
            "high_humidity": "湿度偏高，建议准备除露带。",
            "high_wind": "风力偏大，三脚架需加重。",
            "new_moon": "今晚特别适合深空观测！",
            "full_moon": "满月光太强，建议观测明亮的行星和双星。",
        }
        instance.MAX_UPLOAD_SIZE = 10 * 1024 * 1024
        instance.ALLOWED_IMAGE_EXTENSIONS = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".bmp",
        }
        instance.ALLOWED_AUDIO_EXTENSIONS = {
            ".wav",
            ".mp3",
            ".ogg",
            ".flac",
            ".m4a",
            ".aac",
        }
        instance.RATE_LIMIT_PER_MINUTE = 30
        instance.UPLOAD_RATE_LIMIT_PER_MINUTE = 10
        instance.SESSION_MAX_AGE_SECONDS = 3600
        instance.SESSION_CLEANUP_INTERVAL_SECONDS = 300
        MockSettings.return_value = instance
        yield instance


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_profiles.sqlite")
        yield db_path


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_weather_response():
    return {
        "status": "1",
        "count": "1",
        "info": "OK",
        "infocode": "10000",
        "lives": [
            {
                "province": "北京",
                "city": "朝阳区",
                "adcode": "110105",
                "weather": "晴",
                "temperature": "22",
                "winddirection": "北",
                "windpower": "≤3",
                "humidity": "45",
                "reporttime": "2026-04-08 14:30:00",
                "temperature_float": "22.0",
                "humidity_float": "45.0",
            }
        ],
    }


@pytest.fixture
def sample_weather_forecast_response():
    return {
        "status": "1",
        "count": "1",
        "info": "OK",
        "infocode": "10000",
        "forecasts": [
            {
                "city": "朝阳区",
                "adcode": "110105",
                "province": "北京",
                "reporttime": "2026-04-08 14:30:00",
                "casts": [
                    {
                        "date": "2026-04-08",
                        "week": "3",
                        "dayweather": "晴",
                        "nightweather": "晴",
                        "daytemp": "25",
                        "nighttemp": "12",
                        "daywind": "北",
                        "nightwind": "北",
                        "daypower": "≤3",
                        "nightpower": "≤3",
                        "daytemp_float": "25.0",
                        "nighttemp_float": "12.0",
                    }
                ],
            }
        ],
    }


@pytest.fixture
def sample_nasa_apod_response():
    return {
        "copyright": "Test",
        "date": "2026-04-08",
        "explanation": "A test astronomy picture.",
        "hdurl": "https://apod.nasa.gov/test_hd.jpg",
        "media_type": "image",
        "service_version": "v1",
        "title": "Test APOD",
        "url": "https://apod.nasa.gov/test.jpg",
    }


@pytest.fixture
def sample_neo_response():
    return {
        "element_count": 10,
        "near_earth_objects": {
            "2026-04-08": [
                {
                    "links": {"self": "http://api.nasa.gov/neo/1"},
                    "id": "1",
                    "neo_reference_id": "1",
                    "name": "(2026 Test Asteroid)",
                    "nasa_jpl_url": "http://ssd.jpl.nasa.gov/",
                    "absolute_magnitude_h": 22.0,
                    "estimated_diameter": {
                        "meters": {
                            "estimated_diameter_min": 50.0,
                            "estimated_diameter_max": 120.0,
                        }
                    },
                    "is_potentially_hazardous_asteroid": False,
                    "close_approach_data": [
                        {
                            "close_approach_date": "2026-04-08",
                            "close_approach_date_full": "2026-Apr-08 12:00",
                            "epoch_date_close_approach": 1775731200000,
                            "relative_velocity": {
                                "kilometers_per_second": "15.0",
                            },
                            "miss_distance": {
                                "lunar": "5.0",
                                "kilometers": "1920000",
                            },
                            "orbiting_body": "Earth",
                        }
                    ],
                }
            ],
        },
    }


@pytest.fixture
def sample_planet_position():
    return {
        "ra_hours": 10.5,
        "ra_degrees": 157.5,
        "dec": 20.3,
        "distance_au": 1.5,
    }


@pytest.fixture
def sample_user_profile():
    return {
        "user_id": "test_user",
        "preferences": {"response_style": "详细", "knowledge_level": "专业"},
        "habits": {"frequent_topics": ["火星", "木星"]},
        "constraints": ["避免使用专业术语"],
    }


@pytest.fixture
def mock_ephemeris():
    with patch("src.astronomy.base.EphemerisManager") as MockEph:
        instance = MagicMock()
        instance.is_loaded = True
        instance.planets = MagicMock()
        instance.earth = MagicMock()
        instance.timescale = MagicMock()
        MockEph.return_value = instance
        MockEph._instance = None
        MockEph._initialized = False
        yield instance


@pytest.fixture
def mock_requests_get():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        yield mock_get, mock_response


@pytest.fixture
def mock_requests_post():
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response
        yield mock_post, mock_response


class PerformanceTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.durations = []

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        self.durations.append(duration)
        return duration

    @property
    def avg(self):
        return sum(self.durations) / len(self.durations) if self.durations else 0

    @property
    def p95(self):
        if not self.durations:
            return 0
        sorted_d = sorted(self.durations)
        idx = int(len(sorted_d) * 0.95)
        return sorted_d[min(idx, len(sorted_d) - 1)]

    @property
    def max_duration(self):
        return max(self.durations) if self.durations else 0

    @property
    def min_duration(self):
        return min(self.durations) if self.durations else 0

    def report(self):
        return {
            "total_runs": len(self.durations),
            "avg_ms": round(self.avg * 1000, 2),
            "p95_ms": round(self.p95 * 1000, 2),
            "max_ms": round(self.max_duration * 1000, 2),
            "min_ms": round(self.min_duration * 1000, 2),
        }


@pytest.fixture
def perf_timer():
    return PerformanceTimer()


@pytest.fixture
def skill_manager():
    from unittest.mock import MagicMock, patch

    with patch("src.core.config.settings") as mock_settings:
        mock_settings.DASHSCOPE_API_KEY = "test-key"
        mock_settings.MODEL_NAME = "qwen-max"
        mock_settings.RAG_ENABLED = False
        mock_settings.MEMORY_SIZE = 15
        mock_settings.MEMORY_WINDOW = 8
        mock_settings.LONG_TERM_MEMORY_PATH = "/tmp/test_memory/user_profiles.sqlite"
        mock_settings.DEFAULT_USER_ID = "test_user"
        mock_settings.EPHEMERIS_FILE = "de421.bsp"
        mock_settings.DEFAULT_LOCATION = (39.9, 116.4)
        mock_settings.SUPPORTED_YEAR_RANGE = (2026, 2030)
        mock_settings.PLANET_MAPPING = {
            "mercury": 199,
            "venus": 299,
            "mars": 499,
            "jupiter": 5,
            "saturn": 6,
            "uranus": 7,
            "neptune": 8,
        }
        mock_settings.VALID_PLANETS = {
            "mercury",
            "venus",
            "mars",
            "jupiter",
            "saturn",
            "uranus",
            "neptune",
        }
        mock_settings.PLANET_NAMES_CN = {
            "mercury": "水星",
            "venus": "金星",
            "mars": "火星",
            "jupiter": "木星",
            "saturn": "土星",
            "uranus": "天王星",
            "neptune": "海王星",
        }
        mock_settings.PLANET_MAX_MAGNITUDE = {
            "mercury": -1.9,
            "venus": -4.6,
            "mars": -2.0,
            "jupiter": -2.7,
            "saturn": -0.3,
        }
        mock_settings.SUPPORTED_BODIES = [
            "sun",
            "moon",
            "mercury",
            "venus",
            "mars",
            "jupiter",
            "saturn",
        ]
        mock_settings.CELESTIAL_NAME_MAPPING = {}
        mock_settings.GALAXY_NAME_MAPPING = {}
        mock_settings.MOON_PHASE_THRESHOLDS = []
        mock_settings.OBSERVING_TIPS_TEMPLATES = {}
        mock_settings.NASA_API_KEY = "test-nasa-key"
        mock_settings.NASA_BASE_URL = "https://api.nasa.gov"
        mock_settings.NASA_APOD_URL = "https://api.nasa.gov/planetary/apod"
        mock_settings.NASA_NEO_URL = "https://api.nasa.gov/neo/rest/v1/feed"
        mock_settings.NASA_NEO_MAX_DAYS = 7
        mock_settings.AMAP_API_KEY = "test-amap-key"
        mock_settings.AMAP_DEFAULT_CITY = "北京"
        mock_settings.AMAP_WEATHER_URL = (
            "https://restapi.amap.com/v3/weather/weatherInfo"
        )
        mock_settings.AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/regeo"
        mock_settings.TAVILY_API_KEY = "test-tavily-key"
        mock_settings.TAVILY_SEARCH_URL = "https://api.tavily.com/search"
        mock_settings.MCP_SERVER_URL = "http://localhost:8001/mcp"
        mock_settings.MCP_PORT = 8001
        mock_settings.API_HOST = "0.0.0.0"
        mock_settings.API_PORT = 8000
        mock_settings.MAX_UPLOAD_SIZE = 10 * 1024 * 1024
        mock_settings.ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
        mock_settings.ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3"}
        mock_settings.RATE_LIMIT_PER_MINUTE = 30
        mock_settings.UPLOAD_RATE_LIMIT_PER_MINUTE = 10
        mock_settings.SESSION_MAX_AGE_SECONDS = 3600
        mock_settings.SESSION_CLEANUP_INTERVAL_SECONDS = 300
        mock_settings.EMBEDDING_MODEL_NAME = "text-embedding-v2"
        mock_settings.VISION_MODEL_NAME = "qwen-vl-plus"
        mock_settings.SPEECH_MODEL_NAME = "paraformer-realtime-v2"
        mock_settings.VECTOR_DB_PATH = "/tmp/test_vector_db"
        mock_settings.RERANK_MODEL_NAME = "qwen3-rerank"
        mock_settings.RERANK_ENABLED = False
        mock_settings.RERANK_TOP_N = 3
        mock_settings.RRF_K = 60
        mock_settings.RAG_VECTOR_WEIGHT = 0.5
        mock_settings.RAG_BM25_WEIGHT = 0.5
        mock_settings.RAG_ENTITY_WEIGHT = 0.3
        mock_settings.RAG_RETRIEVAL_CANDIDATES = 20
        mock_settings.RAG_CACHE_ENABLED = False
        mock_settings.RAG_CACHE_TTL = 300
        mock_settings.RAG_CACHE_MAX_SIZE = 256

        with patch("src.agent.skill_manager.AstronomySkillRouter") as MockRouter:
            mock_router = MagicMock()
            mock_router.call_mcp_tool.return_value = '{"status": "ok"}'
            MockRouter.return_value = mock_router

            from src.agent.skill_manager import SkillManager

            sm = SkillManager()
            yield sm
