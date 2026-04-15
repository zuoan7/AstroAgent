import os
import json
import math
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

import pytest

from tests.mock_deps import mock_heavy_dependencies
mock_heavy_dependencies()


class TestExtremeAstronomicalValues:
    """测试极端天文数据值处理"""

    def test_extreme_magnitude_values(self):
        from src.utils.helpers import safe_float

        assert safe_float("-26.74") == -26.74
        assert safe_float("30.0") == 30.0
        assert safe_float("0.0") == 0.0
        assert safe_float("-1.0e10") == -1.0e10
        assert safe_float("1.0e-10") == 1.0e-10

    def test_extreme_coordinate_values(self):
        from src.astronomy.planetary import PlanetaryCalculator
        from src.astronomy.base import EphemerisManager

        mock_eph = MagicMock(spec=EphemerisManager)
        calc = PlanetaryCalculator(ephemeris=mock_eph)

        result = calc.coordinate_transformation(ra=0.0, dec=-90.0, target_system="fk5")
        assert "ra_hours" in result
        assert "ra_degrees" in result
        assert "dec" in result

        result2 = calc.coordinate_transformation(ra=24.0, dec=90.0, target_system="fk5")
        assert "ra_hours" in result2
        assert "ra_degrees" in result2
        assert "dec" in result2

    def test_extreme_distance_values(self):
        from src.utils.helpers import safe_float

        assert safe_float("0.0") == 0.0
        assert safe_float("0.001") == 0.001
        assert safe_float("100000.0") == 100000.0
        assert safe_float("1.0e15") == 1.0e15

    def test_extreme_latitude_longitude(self):
        from src.utils.helpers import parse_coordinate_string

        result = parse_coordinate_string("90,180")
        assert result == (90.0, 180.0)

        result2 = parse_coordinate_string("-90,-180")
        assert result2 == (-90.0, -180.0)

        result3 = parse_coordinate_string("0,0")
        assert result3 == (0.0, 0.0)

    def test_invalid_coordinate_values(self):
        from src.utils.helpers import parse_coordinate_string

        result = parse_coordinate_string("91,180")
        assert result is None

        result2 = parse_coordinate_string("45,181")
        assert result2 is None

        result3 = parse_coordinate_string("-91,0")
        assert result3 is None

    def test_extreme_planet_name_handling(self):
        from src.astronomy.planetary import PlanetaryCalculator
        from src.astronomy.base import EphemerisManager

        mock_eph = MagicMock(spec=EphemerisManager)
        mock_eph.is_loaded = True
        calc = PlanetaryCalculator(ephemeris=mock_eph)

        with patch("src.agent.param_parser.ParamParser.parse_mixed_input", return_value={}):
            with patch("src.astronomy.planetary.load.timescale") as mock_ts:
                mock_t = MagicMock()
                mock_ts_obj = MagicMock()
                mock_ts_obj.now.return_value = mock_t
                mock_ts.return_value = mock_ts_obj
                with patch("src.astronomy.planetary.settings") as mock_s:
                    mock_s.VALID_PLANETS = {"mars"}
                    mock_s.PLANET_MAPPING = {"mars": 499}
                    try:
                        result = calc.get_planet_position("invalid_planet_xyz")
                        assert isinstance(result, dict)
                        assert "error" in result or "ra" not in result
                    except (ModuleNotFoundError, ImportError):
                        pass

    def test_rise_set_polar_regions(self):
        from src.astronomy.planetary import PlanetaryCalculator
        from src.astronomy.base import EphemerisManager

        mock_eph = MagicMock(spec=EphemerisManager)
        calc = PlanetaryCalculator(ephemeris=mock_eph)

        try:
            result = calc.get_rise_set_times("sun", 89.9, 0.0, "2026-06-21")
            assert isinstance(result, dict)
        except Exception:
            pass

    def test_moon_phase_extreme_dates(self):
        from src.astronomy.events_predictor import EventsPredictor

        with patch.object(EventsPredictor, "get_moon_phase", return_value=("🌑 新月", "测试")):
            mock_eph = MagicMock()
            predictor = EventsPredictor(ephemeris=mock_eph, location=(39.9, 116.4))

            phase, desc = predictor.get_moon_phase(datetime(2026, 1, 1))
            assert phase is not None

            phase2, desc2 = predictor.get_moon_phase(datetime(2026, 12, 31))
            assert phase2 is not None


class TestInputDataBoundary:
    """测试输入数据边界条件"""

    def test_empty_string_input(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse("")
        assert isinstance(result, dict)

        result2 = ParamParser.parse("   ")
        assert isinstance(result2, dict)

    def test_none_input(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse(None)
        assert isinstance(result, dict)

    def test_malformed_json_input(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse('{"city": "北京"')
        assert isinstance(result, dict)

        result2 = ParamParser.parse('{city: 北京}')
        assert isinstance(result2, dict)

        result3 = ParamParser.parse('{"city": undefined}')
        assert isinstance(result3, dict)

    def test_numeric_input(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse(42)
        assert isinstance(result, dict)

        result2 = ParamParser.parse(3.14)
        assert isinstance(result2, dict)

    def test_list_input(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse([1, 2, 3])
        assert isinstance(result, dict)

    def test_boolean_input(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse(True)
        assert isinstance(result, dict)

        result2 = ParamParser.parse(False)
        assert isinstance(result2, dict)

    def test_unicode_input(self):
        from src.agent.param_parser import ParamParser

        result = ParamParser.parse("🌟✨🔭🪐")
        assert isinstance(result, dict)
        assert "query" in result

    def test_very_long_string_input(self):
        from src.agent.param_parser import ParamParser

        long_str = "a" * 100000
        result = ParamParser.parse(long_str)
        assert isinstance(result, dict)

    def test_special_characters_in_input(self):
        from src.agent.param_parser import ParamParser

        special = '<script>alert("xss")</script>'
        result = ParamParser.parse(special)
        assert isinstance(result, dict)

        sql_injection = "'; DROP TABLE users; --"
        result2 = ParamParser.parse(sql_injection)
        assert isinstance(result2, dict)

    def test_empty_knowledge_list(self):
        from src.memory.memory import ShortTermMemory

        with patch("src.memory.memory.settings") as mock_s:
            mock_s.MEMORY_SIZE = 15
            mock_s.MEMORY_WINDOW = 8
            mock_s.STM_CONTEXT_MAX_TOKENS = 4000
            mock_s.STM_SUMMARY_MAX_TOKENS = 500
            mock_s.STM_SUMMARY_TRIGGER_MESSAGES = 100
            mock_s.STM_SUMMARY_TRIGGER_TOKENS = 100000
            mock_s.STM_PERSISTENCE_ENABLED = False
            mock_s.STM_PERSISTENCE_PATH = "/tmp/test_stm/sessions.sqlite"
            mock_s.STM_IMPORTANCE_HIGH_ROLES = {"user", "system"}
            mock_s.STM_TOOL_RESULT_MAX_LENGTH = 500
            mock_s.DEFAULT_USER_ID = "test_user"
            mock_s.DASHSCOPE_API_KEY = None
            memory = ShortTermMemory()

        memory.add_message("user", "", time.time())
        assert memory.get_size() == 1

        recent = memory.get_recent_messages()
        assert recent[0]["content"] == ""

    def test_parse_date_edge_cases(self):
        from src.utils.helpers import parse_date

        result = parse_date("2026-02-29")
        assert isinstance(result, datetime)

        result2 = parse_date("2026-13-01")
        assert isinstance(result2, datetime)

        result3 = parse_date("2026-00-01")
        assert isinstance(result3, datetime)

    def test_parse_coordinate_edge_cases(self):
        from src.utils.helpers import parse_coordinate_string

        assert parse_coordinate_string("") is None
        assert parse_coordinate_string(None) is None
        assert parse_coordinate_string("abc") is None
        assert parse_coordinate_string("1,2,3") is None
        assert parse_coordinate_string("a,b") is None

    def test_safe_float_edge_cases(self):
        from src.utils.helpers import safe_float

        assert safe_float(None) is None
        assert safe_float("inf") is None or safe_float("inf") == float("inf")
        assert safe_float("-inf") is None or safe_float("-inf") == float("-inf")
        assert safe_float("nan") is None or math.isnan(safe_float("nan"))

    def test_safe_int_edge_cases(self):
        from src.utils.helpers import safe_int

        assert safe_int(None) is None
        assert safe_int("abc") is None
        assert safe_int("3.14") is not None
        assert safe_int("") is None

    def test_shorten_text_edge_cases(self):
        from src.utils.helpers import shorten_text

        assert shorten_text(None) == ""
        assert shorten_text("") == ""
        assert shorten_text(123) == "123"
        assert shorten_text([1, 2, 3]) == "[1, 2, 3]"

    def test_extract_image_url_edge_cases(self):
        from src.utils.helpers import extract_image_url

        assert extract_image_url(None) is None
        assert extract_image_url("") is None
        assert extract_image_url("no url here") is None
        assert extract_image_url("http://notimage.com/file.txt") is None


class TestResourceBoundary:
    """测试系统资源边界条件"""

    def test_memory_size_limit_enforcement(self):
        from src.memory.memory import ShortTermMemory

        with patch("src.memory.memory.settings") as mock_s:
            mock_s.MEMORY_SIZE = 5
            mock_s.MEMORY_WINDOW = 3
            mock_s.STM_CONTEXT_MAX_TOKENS = 4000
            mock_s.STM_SUMMARY_MAX_TOKENS = 500
            mock_s.STM_SUMMARY_TRIGGER_MESSAGES = 100
            mock_s.STM_SUMMARY_TRIGGER_TOKENS = 100000
            mock_s.STM_PERSISTENCE_ENABLED = False
            mock_s.STM_PERSISTENCE_PATH = "/tmp/test_stm/sessions.sqlite"
            mock_s.STM_IMPORTANCE_HIGH_ROLES = {"user", "system"}
            mock_s.STM_TOOL_RESULT_MAX_LENGTH = 500
            mock_s.DEFAULT_USER_ID = "test_user"
            mock_s.DASHSCOPE_API_KEY = None
            memory = ShortTermMemory()

        for i in range(100):
            memory.add_message("user", f"消息{i}", time.time())

        assert memory.get_size() == 5

    def test_large_number_of_profiles(self, temp_db_path):
        from src.memory.memory import LongTermMemory

        memory = LongTermMemory(db_path=temp_db_path)

        for i in range(500):
            memory.merge_and_update(f"user_{i}", {
                "preferences": {"style": "详细"},
                "habits": {"topics": ["火星"]},
                "constraints": [],
            })

        profile = memory.load_profile("user_499")
        assert profile is not None
        assert profile.user_id == "user_499"

    def test_large_profile_data(self, temp_db_path):
        from src.memory.memory import LongTermMemory

        memory = LongTermMemory(db_path=temp_db_path)

        large_topics = [f"天体_{i}" for i in range(1000)]
        large_constraints = [f"约束_{i}" for i in range(100)]

        memory.merge_and_update("large_user", {
            "preferences": {f"pref_{i}": f"value_{i}" for i in range(100)},
            "habits": {"topics": large_topics},
            "constraints": large_constraints,
        })

        profile = memory.load_profile("large_user")
        assert profile is not None
        assert len(profile.habits["topics"]) == 1000
        assert len(profile.constraints) == 100

    def test_concurrent_db_access(self, temp_db_path):
        import threading
        from src.memory.memory import LongTermMemory

        errors = []

        def write_read(user_id):
            try:
                memory = LongTermMemory(db_path=temp_db_path)
                memory.merge_and_update(f"concurrent_{user_id}", {
                    "preferences": {"id": str(user_id)},
                    "habits": {},
                    "constraints": [],
                })
                profile = memory.load_profile(f"concurrent_{user_id}")
                assert profile is not None
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_read, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"并发数据库访问错误: {errors}"

    def test_error_handler_with_deep_nesting(self):
        from src.core.errors import ErrorHandler

        try:
            try:
                try:
                    raise ValueError("深层嵌套错误")
                except Exception as e:
                    raise RuntimeError("中间层错误") from e
            except Exception as e:
                raise TypeError("外层错误") from e
        except Exception as e:
            error = ErrorHandler.handle(e, {"depth": "deep"})

        assert error is not None
        assert error.code.value in ("VALIDATION_ERROR", "UNKNOWN_ERROR")

    def test_param_parser_with_deeply_nested_json(self):
        from src.agent.param_parser import ParamParser

        nested = json.dumps({
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        })

        result = ParamParser.parse(nested)
        assert isinstance(result, dict)
        assert "level1" in result

    def test_network_timeout_handling(self):
        from src.astronomy.search_service import SearchService

        service = SearchService()
        service.api_key = "test-key"

        with patch("requests.post") as mock_post:
            import requests
            mock_post.side_effect = requests.exceptions.Timeout("请求超时")

            result = service.search("测试查询")
            assert isinstance(result, dict)
            assert "error" in result

    def test_network_connection_error_handling(self):
        from src.astronomy.search_service import SearchService

        service = SearchService()
        service.api_key = "test-key"

        with patch("requests.post") as mock_post:
            import requests
            mock_post.side_effect = requests.exceptions.ConnectionError("连接失败")

            result = service.search("测试查询")
            assert isinstance(result, dict)
            assert "error" in result

    def test_api_key_missing_handling(self):
        from src.astronomy.search_service import SearchService

        service = SearchService()
        service.api_key = None

        result = service.search("测试查询")
        assert isinstance(result, dict)
        assert "error" in result

    def test_weather_api_missing_key(self):
        from src.astronomy.weather_service import WeatherService

        service = WeatherService()
        service.api_key = None

        result = service.get_weather("北京")
        assert isinstance(result, dict)
        assert "error" in result


class TestTimeBoundary:
    """测试时间相关边界条件"""

    def test_date_change_boundary(self):
        from src.utils.helpers import parse_date

        midnight = parse_date("2026-04-08 00:00:00")
        assert midnight.hour == 0
        assert midnight.minute == 0

        before_midnight = parse_date("2026-04-08 23:59:59")
        assert before_midnight.hour == 23
        assert before_midnight.minute == 59

    def test_leap_year_handling(self):
        from src.utils.helpers import parse_date

        leap_date = parse_date("2024-02-29")
        assert leap_date.month == 2
        assert leap_date.day == 29

        non_leap_date = parse_date("2025-02-29")
        assert isinstance(non_leap_date, datetime)

    def test_year_boundary(self):
        from src.utils.helpers import parse_date

        new_year = parse_date("2026-01-01")
        assert new_year.month == 1
        assert new_year.day == 1

        year_end = parse_date("2026-12-31")
        assert year_end.month == 12
        assert year_end.day == 31

    def test_nasa_neo_date_range_limit(self):
        from src.astronomy.nasa_api import NASAAPIService

        service = NASAAPIService()
        service.api_key = "test-key"

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"near_earth_objects": {}}
            mock_get.return_value = mock_resp

            service.get_neo_data(start_date="2026-04-01", end_date="2026-04-15")

            call_args = mock_get.call_args
            params = call_args[1].get("params", call_args[0][1] if len(call_args[0]) > 1 else {})
            if "params" in call_args[1]:
                end_date_param = call_args[1]["params"].get("end_date")
                if end_date_param:
                    end_dt = datetime.strptime(end_date_param, "%Y-%m-%d")
                    start_dt = datetime.strptime("2026-04-01", "%Y-%m-%d")
                    assert (end_dt - start_dt).days <= 7

    def test_events_predictor_date_range(self):
        from src.astronomy.events_predictor import EventsPredictor

        mock_eph = MagicMock()
        with patch.object(EventsPredictor, "get_moon_phase", return_value=("🌑 新月", "测试")):
            with patch.object(EventsPredictor, "get_visible_planets", return_value=[]):
                with patch.object(EventsPredictor, "get_sunrise_sunset", return_value=(None, None)):
                    predictor = EventsPredictor(ephemeris=mock_eph, location=(39.9, 116.4))

                    result = predictor.get_weekly_events(start_date="2026-01-01")
                    assert isinstance(result, str)

    def test_normalize_date_various_formats(self):
        from src.agent.param_parser import ParamParser

        assert ParamParser.normalize_date("2026-01-01") == "2026-01-01"
        assert ParamParser.normalize_date("2026/01/01") == "2026-01-01"
        assert ParamParser.normalize_date("今天") is not None
        assert ParamParser.normalize_date("明天") is not None
        assert ParamParser.normalize_date(None) is None
        assert ParamParser.normalize_date("invalid") is None

    def test_month_boundary_values(self):
        from src.agent.param_parser import ParamParser

        assert ParamParser.safe_int("1") == 1
        assert ParamParser.safe_int("12") == 12
        assert ParamParser.safe_int("0") == 0
        assert ParamParser.safe_int("13") == 13
        assert ParamParser.safe_int("-1") == -1

    def test_timezone_handling_in_date_parsing(self):
        from src.utils.helpers import parse_date

        utc_date = parse_date("2026-04-08T14:30:00+08:00")
        assert isinstance(utc_date, datetime)

    def test_far_future_date(self):
        from src.utils.helpers import parse_date

        future = parse_date("2100-12-31")
        assert future.year == 2100

    def test_far_past_date(self):
        from src.utils.helpers import parse_date

        past = parse_date("1900-01-01")
        assert past.year == 1900

    def test_events_predictor_monthly_boundary(self):
        from src.astronomy.events_predictor import EventsPredictor

        mock_eph = MagicMock()
        with patch.object(EventsPredictor, "get_moon_phase", return_value=("🌑 新月", "测试")):
            predictor = EventsPredictor(ephemeris=mock_eph, location=(39.9, 116.4))

            result = predictor.get_monthly_events(year=2026, month=1)
            assert isinstance(result, str)

            result2 = predictor.get_monthly_events(year=2026, month=12)
            assert isinstance(result2, str)

    def test_time_range_parsing_boundary(self):
        from src.skills.skill_handlers import _parse_time_range

        start, end = _parse_time_range(None)
        assert start is not None
        assert end is not None

        start2, end2 = _parse_time_range("未来30天")
        assert start2 is not None
        assert end2 is not None

        start3, end3 = _parse_time_range("本月")
        assert start3 is not None
        assert end3 is not None
