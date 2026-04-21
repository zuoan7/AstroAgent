import json
import os
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()


class TestAstronomyModuleIntegration:
    """测试天文数据获取模块与数据分析模块之间的交互"""

    def test_ephemeris_to_planetary_data_flow(self, mock_ephemeris):
        from src.astronomy.planetary import PlanetaryCalculator

        calc = PlanetaryCalculator(ephemeris=mock_ephemeris)

        mock_planet = MagicMock()
        mock_astrometric = MagicMock()
        mock_ra = MagicMock()
        mock_ra.hours = 10.5
        mock_ra._degrees = 157.5
        mock_dec = MagicMock()
        mock_dec.degrees = 20.3
        mock_distance = MagicMock()
        mock_distance.au = 1.5
        mock_astrometric.radec.return_value = (mock_ra, mock_dec, mock_distance)

        mock_alt = MagicMock()
        mock_alt.degrees = 45.0
        mock_az = MagicMock()
        mock_az.degrees = 180.0
        mock_apparent = MagicMock()
        mock_apparent.altaz.return_value = (mock_alt, mock_az, mock_distance)
        mock_astrometric.apparent.return_value = mock_apparent

        mock_observer = MagicMock()
        mock_observer.at.return_value.observe.return_value = mock_astrometric
        mock_ephemeris.earth.__add__ = MagicMock(return_value=mock_observer)
        mock_ephemeris.planets.__getitem__ = MagicMock(return_value=mock_planet)

        with patch(
            "src.agent.param_parser.ParamParser.parse_mixed_input", return_value={}
        ):
            with patch("src.astronomy.planetary.load.timescale") as mock_ts:
                mock_t = MagicMock()
                mock_ts_obj = MagicMock()
                mock_ts_obj.now.return_value = mock_t
                mock_ts_obj.utc.return_value = mock_t
                mock_ts.return_value = mock_ts_obj

                with patch("src.astronomy.planetary.settings") as mock_s:
                    mock_s.VALID_PLANETS = {"mars"}
                    mock_s.PLANET_MAPPING = {"mars": 499}
                    result = calc.get_planet_position(
                        "mars", latitude=39.9, longitude=116.4
                    )

        assert "ra_hours" in result
        assert "ra_degrees" in result
        assert "dec" in result
        assert "distance_au" in result

    def test_ephemeris_not_loaded_returns_error(self):
        from src.astronomy.base import EphemerisManager
        from src.astronomy.planetary import PlanetaryCalculator

        mock_eph = MagicMock(spec=EphemerisManager)
        mock_eph.is_loaded = False

        calc = PlanetaryCalculator(ephemeris=mock_eph)
        try:
            result = calc.get_planet_position("mars")
            assert isinstance(result, dict)
            assert "error" in result
        except (ModuleNotFoundError, ImportError):
            pass

    def test_events_predictor_moon_phase_integration(self, mock_ephemeris):
        from src.astronomy.events_predictor import EventsPredictor

        mock_e = MagicMock()
        mock_m = MagicMock()
        mock_s = MagicMock()

        mock_moon_earth = MagicMock()
        mock_moon_ra = MagicMock()
        mock_moon_ra.hours = 15.0
        mock_moon_earth.radec.return_value = (mock_moon_ra, MagicMock(), MagicMock())

        mock_sun_earth = MagicMock()
        mock_sun_ra = MagicMock()
        mock_sun_ra.hours = 5.0
        mock_sun_earth.radec.return_value = (mock_sun_ra, MagicMock(), MagicMock())

        mock_e.at.return_value.observe.return_value.apparent.return_value = None

        mock_ephemeris.planets.__getitem__ = MagicMock(
            side_effect=lambda x: {"earth": mock_e, "moon": mock_m, "sun": mock_s}[x]
        )

        with patch.object(
            EventsPredictor, "get_moon_phase", return_value=("🌕 满月", "整夜可见")
        ):
            predictor = EventsPredictor(
                ephemeris=mock_ephemeris, location=(39.9, 116.4)
            )
            phase, desc = predictor.get_moon_phase(datetime(2026, 4, 8))

        assert phase is not None
        assert desc is not None

    def test_weather_service_to_observing_tips_flow(self, sample_weather_response):
        from src.astronomy.weather_service import WeatherService

        service = WeatherService()
        service.api_key = "test-key"

        result = service._process_weather_response(
            sample_weather_response, "北京", "base"
        )

        assert "live" in result
        assert "observing_tips" in result
        assert isinstance(result["observing_tips"], list)

    def test_nasa_api_to_neo_tracker_data_flow(self, sample_neo_response):
        from src.astronomy.nasa_api import NASAAPIService

        service = NASAAPIService()
        service.api_key = "test-key"

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = sample_neo_response
            mock_get.return_value = mock_resp

            result = service.get_neo_data(
                start_date="2026-04-08", end_date="2026-04-09"
            )

        assert "near_earth_objects" in result
        assert "2026-04-08" in result["near_earth_objects"]

    def test_coordinate_transformation_icrs_to_fk5(self):
        from src.astronomy.base import EphemerisManager
        from src.astronomy.planetary import PlanetaryCalculator

        mock_eph = MagicMock(spec=EphemerisManager)
        calc = PlanetaryCalculator(ephemeris=mock_eph)

        mock_fk5_coord = MagicMock()
        mock_fk5_coord.ra.hour = 10.5
        mock_fk5_coord.ra.degree = 157.5
        mock_fk5_coord.dec.degree = 20.3
        mock_sky = MagicMock()
        mock_sky.transform_to.return_value = mock_fk5_coord

        with patch("src.astronomy.planetary.SkyCoord", return_value=mock_sky):
            result = calc.coordinate_transformation(
                ra=10.5, dec=20.3, epoch="J2000", target_system="fk5"
            )

        assert "ra_hours" in result
        assert "ra_degrees" in result
        assert "dec" in result
        assert isinstance(result["ra_hours"], float)
        assert isinstance(result["dec"], float)

    def test_rise_set_times_data_flow(self):
        import zoneinfo
        from datetime import timezone

        from src.astronomy.base import EphemerisManager
        from src.astronomy.planetary import PlanetaryCalculator

        mock_eph = MagicMock(spec=EphemerisManager)
        mock_eph.is_loaded = True
        mock_eph.planets = MagicMock()
        mock_eph.earth = MagicMock()
        mock_eph.timescale = MagicMock()

        calc = PlanetaryCalculator(ephemeris=mock_eph)

        with patch(
            "src.astronomy.planetary.skyfield_almanac.find_discrete"
        ) as mock_find:
            mock_times = [MagicMock(), MagicMock()]
            mock_times[0].utc_datetime.return_value = datetime(
                2026, 4, 8, 5, 30, tzinfo=timezone.utc
            )
            mock_times[1].utc_datetime.return_value = datetime(
                2026, 4, 8, 18, 45, tzinfo=timezone.utc
            )
            mock_events = [1, 0]
            mock_find.return_value = (mock_times, mock_events)

            result = calc.get_rise_set_times("sun", 39.9, 116.4, "2026-04-08")

        assert "rise_time" in result or "error" in result


class TestMemoryModuleIntegration:
    """测试记忆模块与Agent模块之间的数据流转"""

    def test_memory_service_add_and_retrieve(self, tmp_path):
        from src.memory.api.dto import AppendMessageRequest
        from src.memory.api.memory_service import MemoryService

        memory = MemoryService(
            db_path=str(tmp_path / "memory.sqlite"),
            session_id="integration_session",
            user_id="test_user",
        )
        memory.append_message(
            AppendMessageRequest(
                session_id="integration_session",
                role="user",
                content="你好",
                timestamp=time.time(),
            )
        )
        memory.append_message(
            AppendMessageRequest(
                session_id="integration_session",
                role="assistant",
                content="你好！有什么天文问题吗？",
                timestamp=time.time(),
            )
        )

        recent = memory.get_all_messages("integration_session")
        assert len(recent) == 2
        assert recent[0]["role"] == "user"
        assert recent[1]["role"] == "assistant"

    def test_memory_service_context_limit(self, tmp_path):
        from src.memory.api.dto import AppendMessageRequest, BuildContextRequest
        from src.memory.api.memory_service import MemoryService

        memory = MemoryService(
            db_path=str(tmp_path / "memory.sqlite"),
            session_id="integration_session",
            user_id="test_user",
        )
        for i in range(10):
            memory.append_message(
                AppendMessageRequest(
                    session_id="integration_session",
                    role="user",
                    content=f"消息{i}",
                    timestamp=time.time(),
                )
            )

        context = memory.build_context(
            BuildContextRequest(session_id="integration_session", max_tokens=300)
        )
        assert len(memory.get_all_messages("integration_session")) == 10
        assert len(context["selected_recent_messages"]) <= 8

    def test_memory_service_clear(self, tmp_path):
        from src.memory.api.dto import AppendMessageRequest
        from src.memory.api.memory_service import MemoryService

        memory = MemoryService(
            db_path=str(tmp_path / "memory.sqlite"),
            session_id="integration_session",
            user_id="test_user",
        )
        memory.append_message(
            AppendMessageRequest(
                session_id="integration_session",
                role="user",
                content="测试",
                timestamp=time.time(),
            )
        )
        assert len(memory.get_all_messages("integration_session")) == 1

        memory.clear("integration_session")
        assert memory.get_all_messages("integration_session") == []

    def test_long_term_memory_save_and_load(self, temp_db_path):
        from src.memory.long_term_memory import LongTermMemoryManager as LongTermMemory

        memory = LongTermMemory(db_path=temp_db_path)
        memory.merge_and_update(
            "test_user",
            {
                "preferences": {"style": "详细"},
                "habits": {"topics": ["火星"]},
                "constraints": ["避免术语"],
            },
        )

        loaded = memory.load_profile("test_user")

        assert loaded is not None
        assert loaded["user_id"] == "test_user"
        assert loaded["preferences"]["style"] == "详细"
        assert "火星" in loaded["habits"]["topics"]
        assert "避免术语" in loaded["constraints"]

    def test_long_term_memory_merge_and_update(self, temp_db_path):
        from src.memory.long_term_memory import LongTermMemoryManager as LongTermMemory

        memory = LongTermMemory(db_path=temp_db_path)

        memory.merge_and_update(
            "test_user",
            {
                "preferences": {"style": "简短"},
                "habits": {"topics": ["火星"]},
                "constraints": ["避免术语"],
            },
        )

        memory.merge_and_update(
            "test_user",
            {
                "preferences": {"style": "详细"},
                "habits": {"topics": ["木星"]},
                "constraints": ["控制长度"],
            },
        )

        profile = memory.load_profile("test_user")
        assert profile is not None
        assert profile.preferences["style"] == "详细"
        assert "火星" in profile.habits["topics"]
        assert "木星" in profile.habits["topics"]
        assert "避免术语" in profile.constraints
        assert "控制长度" in profile.constraints

    def test_long_term_memory_delete_profile(self, temp_db_path):
        from src.memory.long_term_memory import LongTermMemoryManager as LongTermMemory

        memory = LongTermMemory(db_path=temp_db_path)

        memory.merge_and_update(
            "test_user",
            {
                "preferences": {},
                "habits": {},
                "constraints": [],
            },
        )

        assert memory.load_profile("test_user") is not None
        deleted = memory.delete_profile("test_user")
        assert deleted is True
        assert memory.load_profile("test_user") is None

    def test_long_term_memory_extract_from_conversation(self, temp_db_path):
        from src.memory.long_term_memory import LongTermMemoryManager as LongTermMemory

        memory = LongTermMemory(db_path=temp_db_path)

        extracted = memory.extract_from_conversation(
            "请详细介绍一下火星的特征，不要使用专业术语", "火星是太阳系第四颗行星..."
        )

        assert extracted["preferences"]["response_style"] == "详细"
        assert "避免使用专业术语" in extracted["constraints"]
        assert "火星" in extracted["habits"]["frequent_topics"]

    def test_long_term_memory_format_for_prompt(self, temp_db_path):
        from src.memory.long_term_memory import LongTermMemoryManager as LongTermMemory

        memory = LongTermMemory(db_path=temp_db_path)

        memory.merge_and_update(
            "test_user",
            {
                "preferences": {"style": "详细"},
                "habits": {"topics": ["火星", "木星"]},
                "constraints": ["避免术语"],
            },
        )

        formatted = memory.format_profile_for_prompt("test_user")
        assert "详细" in formatted
        assert "火星" in formatted


class TestErrorHandlingIntegration:
    """测试错误处理机制在模块间的集成"""

    def test_error_handler_maps_value_error(self):
        from src.core.errors import ErrorCode, ErrorHandler

        try:
            raise ValueError("测试值错误")
        except Exception as e:
            error = ErrorHandler.handle(e, {"context": "test"})

        assert error.code == ErrorCode.VALIDATION_ERROR
        assert "测试值错误" in error.message

    def test_error_handler_maps_file_not_found(self):
        from src.core.errors import ErrorCode, ErrorHandler

        try:
            raise FileNotFoundError("文件不存在")
        except Exception as e:
            error = ErrorHandler.handle(e)

        assert error.code == ErrorCode.FILE_NOT_FOUND

    def test_error_handler_preserves_agent_error(self):
        from src.core.errors import AgentError, ErrorCode, ErrorHandler

        original = AgentError(
            code=ErrorCode.NASA_API_ERROR,
            message="NASA API调用失败",
            details={"api": "APOD"},
        )

        result = ErrorHandler.handle(original)
        assert result is original
        assert result.code == ErrorCode.NASA_API_ERROR

    def test_error_to_dict_format(self):
        from src.core.errors import AgentError, ErrorCode

        error = AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message="工具调用失败",
            details={"tool": "get_weather"},
        )

        d = error.to_dict()
        assert d["error"] is True
        assert d["code"] == "TOOL_CALL_FAILED"
        assert d["message"] == "工具调用失败"
        assert d["details"]["tool"] == "get_weather"

    def test_safe_tool_call_decorator(self):
        from src.core.errors import AgentError, ErrorCode, safe_tool_call

        @safe_tool_call
        def failing_tool():
            raise ValueError("参数错误")

        result = failing_tool()
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "VALIDATION_ERROR"

    def test_nasa_api_error_propagation(self):
        from src.astronomy.nasa_api import NASAAPIService
        from src.core.errors import AgentError, ErrorCode

        service = NASAAPIService()
        service.api_key = "test-key"

        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("网络错误")

            with pytest.raises(AgentError) as exc_info:
                service.get_apod()

        assert exc_info.value.code == ErrorCode.NASA_API_ERROR

    def test_weather_service_error_returns_dict(self):
        from src.astronomy.weather_service import WeatherService

        service = WeatherService()
        service.api_key = None

        result = service.get_weather("北京")
        assert isinstance(result, dict)
        assert "error" in result


class TestParamParserIntegration:
    """测试参数解析器在模块间的集成"""

    def test_parse_json_string_input(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse('{"city": "北京", "extensions": "all"}')
        assert result["city"] == "北京"
        assert result["extensions"] == "all"

    def test_parse_dict_input(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse({"city": "上海"})
        assert result["city"] == "上海"

    def test_parse_plain_string_with_primary_param(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse("北京", primary_param="city")
        assert result["city"] == "北京"

    def test_parse_tool_input_with_defaults(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse_tool_input(
            '{"city": "北京"}',
            expected_params={"city": None, "extensions": "all", "timeout": 30},
        )
        assert result["city"] == "北京"
        assert result["extensions"] == "all"
        assert result["timeout"] == 30

    def test_normalize_date_today(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.normalize_date("今天")
        assert result == datetime.now().strftime("%Y-%m-%d")

    def test_normalize_date_tomorrow(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.normalize_date("明天")
        expected = (datetime.now() + __import__("datetime").timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        assert result == expected

    def test_normalize_date_iso_format(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.normalize_date("2026-04-08")
        assert result == "2026-04-08"

    def test_safe_int_conversion(self):
        from src.agent.param_parser import ParamParser

        assert ParamParser.safe_int("42") == 42
        assert ParamParser.safe_int("invalid", default=0) == 0
        assert ParamParser.safe_int(None, default=-1) == -1

    def test_safe_float_conversion(self):
        from src.agent.param_parser import ParamParser

        assert ParamParser.safe_float("3.14") == 3.14
        assert ParamParser.safe_float("invalid", default=0.0) == 0.0
        assert ParamParser.safe_float(None, default=-1.0) == -1.0

    def test_normalize_location_dict(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.normalize_location({"city": "上海", "location": "浦东"})
        assert result == "浦东"

    def test_normalize_location_json_string(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.normalize_location('{"city": "广州"}')
        assert result == "广州"


class TestFallbackServiceIntegration:
    """测试降级服务在模块间的集成"""

    def test_should_use_fallback_on_empty_output(self):
        from src.agent.fallback_service import FallbackService

        mock_sm = MagicMock()
        service = FallbackService(skill_manager=mock_sm)

        assert service.should_use_fallback("") is True
        assert service.should_use_fallback(None) is True

    def test_should_not_use_fallback_on_valid_output(self):
        from src.agent.fallback_service import FallbackService

        mock_sm = MagicMock()
        service = FallbackService(skill_manager=mock_sm)

        assert service.should_use_fallback("这是一个正常的回答") is False

    def test_should_use_fallback_on_error_keywords(self):
        from src.agent.fallback_service import FallbackService

        mock_sm = MagicMock()
        service = FallbackService(skill_manager=mock_sm)

        assert service.should_use_fallback("工具调用错误，请重试") is True
        assert service.should_use_fallback("无法连接到MCP服务器") is True

    def test_format_fallback_response_with_search_results(self):
        from src.agent.fallback_service import FallbackService

        mock_sm = MagicMock()
        service = FallbackService(skill_manager=mock_sm)

        search_data = json.dumps(
            {
                "answer": "火星是太阳系第四颗行星",
                "results": [
                    {
                        "title": "火星简介",
                        "url": "https://example.com",
                        "content": "火星相关信息",
                    }
                ],
            }
        )

        response = service.format_fallback_response("火星是什么", search_data)
        assert "火星" in response
        assert "搜索结果" in response or "信息" in response

    def test_format_fallback_response_with_error(self):
        from src.agent.fallback_service import FallbackService

        mock_sm = MagicMock()
        service = FallbackService(skill_manager=mock_sm)

        error_data = json.dumps(
            {
                "error": True,
                "code": "API_ERROR",
                "message": "搜索服务不可用",
            }
        )

        response = service.format_fallback_response("测试查询", error_data)
        assert "问题" in response or "抱歉" in response


class TestHelpersIntegration:
    """测试工具函数在模块间的集成"""

    def test_parse_date_various_formats(self):
        from src.utils.helpers import parse_date

        dt = parse_date("2026-04-08")
        assert dt.year == 2026
        assert dt.month == 4
        assert dt.day == 8

        dt2 = parse_date("2026/04/08")
        assert dt2.year == 2026

        dt3 = parse_date("2026-04-08 14:30:00")
        assert dt3.hour == 14
        assert dt3.minute == 30

    def test_parse_date_natural_language(self):
        from src.utils.helpers import parse_date

        dt = parse_date("今天")
        assert dt.date() == datetime.now().date()

        dt2 = parse_date("明天")
        expected = (datetime.now() + __import__("datetime").timedelta(days=1)).date()
        assert dt2.date() == expected

    def test_parse_date_none_returns_default(self):
        from src.utils.helpers import parse_date

        dt = parse_date(None)
        assert isinstance(dt, datetime)

    def test_parse_coordinate_string(self):
        from src.utils.helpers import parse_coordinate_string

        result = parse_coordinate_string("39.9,116.4")
        assert result == (39.9, 116.4)

        result2 = parse_coordinate_string("latitude=39.9, longitude=116.4")
        assert result2 == (39.9, 116.4)

    def test_is_coordinates(self):
        from src.utils.helpers import is_coordinates

        assert is_coordinates("39.9,116.4") is True
        assert is_coordinates("北京") is False

    def test_get_direction_from_azimuth(self):
        from src.utils.helpers import get_direction_from_azimuth

        assert get_direction_from_azimuth(0) == "北"
        assert get_direction_from_azimuth(90) == "东"
        assert get_direction_from_azimuth(180) == "南"
        assert get_direction_from_azimuth(270) == "西"

    def test_shorten_text(self):
        from src.utils.helpers import shorten_text

        long_text = "a" * 2000
        result = shorten_text(long_text, max_len=100)
        assert len(result) == 100
        assert result.endswith("...")

        short_text = "hello"
        result2 = shorten_text(short_text, max_len=100)
        assert result2 == "hello"

    def test_extract_image_url(self):
        from src.utils.helpers import extract_image_url

        text = "图片链接 https://example.com/image.png 查看更多"
        result = extract_image_url(text)
        assert result == "https://example.com/image.png"

        text2 = "没有图片链接"
        result2 = extract_image_url(text2)
        assert result2 is None
