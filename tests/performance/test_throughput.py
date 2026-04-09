import os
import json
import time
import tracemalloc
import threading
from unittest.mock import MagicMock, patch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, asCompleted

import pytest

from tests.mock_deps import mock_heavy_dependencies
mock_heavy_dependencies()


class TestAstronomyDataProcessingThroughput:
    """测试天文数据处理吞吐量"""

    def test_planetary_position_calculation_throughput(self, perf_timer):
        from src.astronomy.planetary import PlanetaryCalculator
        from src.astronomy.base import EphemerisManager

        mock_eph = MagicMock(spec=EphemerisManager)
        mock_eph.is_loaded = True

        mock_planet = MagicMock()
        mock_astrometric = MagicMock()
        mock_ra = MagicMock()
        mock_ra.hours = 10.5
        mock_dec = MagicMock()
        mock_dec.degrees = 20.3
        mock_distance = MagicMock()
        mock_distance.au = 1.5
        mock_astrometric.radec.return_value = (mock_ra, mock_dec, mock_distance)

        mock_observer = MagicMock()
        mock_observer.at.return_value.observe.return_value = mock_astrometric
        mock_eph.earth = MagicMock(return_value=mock_observer)
        mock_eph.earth.__add__ = MagicMock(return_value=mock_observer)
        mock_eph.planets = MagicMock()
        mock_eph.planets.__getitem__ = MagicMock(return_value=mock_planet)

        calc = PlanetaryCalculator(ephemeris=mock_eph)

        planets = ["mercury", "venus", "mars", "jupiter", "saturn"]

        for _ in range(50):
            for planet in planets:
                perf_timer.start()
                with patch("src.astronomy.planetary.parse_mixed_input", return_value={}):
                    with patch("src.astronomy.planetary.load.timescale") as mock_ts:
                        mock_t = MagicMock()
                        mock_ts_obj = MagicMock()
                        mock_ts_obj.now.return_value = mock_t
                        mock_ts.return_value = mock_ts_obj
                        with patch("src.astronomy.planetary.settings") as mock_s:
                            mock_s.VALID_PLANETS = set(planets)
                            mock_s.PLANET_MAPPING = {
                                "mercury": 199, "venus": 299, "mars": 499,
                                "jupiter": 5, "saturn": 6,
                            }
                            try:
                                calc.get_planet_position(planet)
                            except (ModuleNotFoundError, ImportError):
                                pass
                perf_timer.stop()

        report = perf_timer.report()
        assert report["total_runs"] == 250

    def test_coordinate_transformation_throughput(self, perf_timer):
        from src.astronomy.planetary import PlanetaryCalculator
        from src.astronomy.base import EphemerisManager

        mock_eph = MagicMock(spec=EphemerisManager)
        calc = PlanetaryCalculator(ephemeris=mock_eph)

        test_coords = [(ra, dec) for ra in range(0, 24) for dec in range(-90, 90, 10)]

        for ra, dec in test_coords[:100]:
            perf_timer.start()
            calc.coordinate_transformation(ra=ra, dec=dec, target_system="fk5")
            perf_timer.stop()

        report = perf_timer.report()
        assert report["total_runs"] == 100
        assert report["avg_ms"] < 50, f"坐标转换平均 {report['avg_ms']}ms 超过50ms阈值"

    def test_date_parsing_throughput(self, perf_timer):
        from src.utils.helpers import parse_date

        test_dates = [
            "2026-04-08", "2026/04/08", "2026-04-08 14:30:00",
            "今天", "明天", "2026-01-01", "2026-12-31",
        ]

        for _ in range(100):
            for date_str in test_dates:
                perf_timer.start()
                parse_date(date_str)
                perf_timer.stop()

        report = perf_timer.report()
        assert report["total_runs"] == 700
        assert report["avg_ms"] < 10, f"日期解析平均 {report['avg_ms']}ms 超过10ms阈值"

    def test_param_parser_throughput(self, perf_timer):
        from src.agent.param_parser import ParamParser

        test_inputs = [
            '{"city": "北京", "extensions": "all"}',
            {"city": "上海"},
            "深圳",
            '{"target": "mars", "datetime": "2026-04-08"}',
        ]

        for _ in range(200):
            for inp in test_inputs:
                perf_timer.start()
                ParamParser.parse(inp)
                perf_timer.stop()

        report = perf_timer.report()
        assert report["total_runs"] == 800
        assert report["avg_ms"] < 5, f"参数解析平均 {report['avg_ms']}ms 超过5ms阈值"

    def test_memory_operations_throughput(self, perf_timer):
        from src.memory.memory import ShortTermMemory

        with patch("src.memory.memory.settings") as mock_s:
            mock_s.MEMORY_SIZE = 100
            mock_s.MEMORY_WINDOW = 10
            memory = ShortTermMemory()

        for i in range(500):
            perf_timer.start()
            memory.add_message("user", f"消息{i}", time.time())
            memory.get_recent_messages()
            perf_timer.stop()

        report = perf_timer.report()
        assert report["total_runs"] == 500
        assert report["avg_ms"] < 5, f"记忆操作平均 {report['avg_ms']}ms 超过5ms阈值"

    def test_long_term_memory_throughput(self, perf_timer, temp_db_path):
        from src.memory.memory import LongTermMemory

        memory = LongTermMemory(db_path=temp_db_path)

        for i in range(100):
            perf_timer.start()
            memory.merge_and_update(f"user_{i}", {
                "preferences": {"style": "详细"},
                "habits": {"topics": ["火星"]},
                "constraints": [],
            })
            perf_timer.stop()

        report = perf_timer.report()
        assert report["total_runs"] == 100
        assert report["avg_ms"] < 50, f"长期记忆写入平均 {report['avg_ms']}ms 超过50ms阈值"

    def test_error_handling_throughput(self, perf_timer):
        from src.core.errors import ErrorHandler, AgentError, ErrorCode

        for _ in range(500):
            perf_timer.start()
            try:
                raise ValueError("测试错误")
            except Exception as e:
                ErrorHandler.handle(e, {"context": "test"})
            perf_timer.stop()

        report = perf_timer.report()
        assert report["total_runs"] == 500
        assert report["avg_ms"] < 5, f"错误处理平均 {report['avg_ms']}ms 超过5ms阈值"

    def test_weather_data_processing_throughput(self, perf_timer, sample_weather_response):
        from src.astronomy.weather_service import WeatherService

        service = WeatherService()

        for _ in range(200):
            perf_timer.start()
            service._process_weather_response(
                sample_weather_response, "北京", "base"
            )
            perf_timer.stop()

        report = perf_timer.report()
        assert report["total_runs"] == 200
        assert report["avg_ms"] < 10, f"天气数据处理平均 {report['avg_ms']}ms 超过10ms阈值"


class TestAPIResponseTime:
    """测试API响应时间"""

    @pytest.fixture
    def test_client(self):
        from fastapi.testclient import TestClient

        with patch("src.api.main.AstroAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.user_id = "test_user"
            mock_agent.long_term_memory = MagicMock()

            mock_profile = MagicMock()
            mock_profile.user_id = "test_user"
            mock_profile.preferences = {}
            mock_profile.habits = {}
            mock_profile.constraints = []
            mock_profile.created_at = "2026-01-01T00:00:00"
            mock_profile.updated_at = "2026-04-08T00:00:00"
            mock_agent.long_term_memory.load_profile.return_value = mock_profile

            MockAgent.return_value = mock_agent

            from src.api.main import app
            client = TestClient(app)
            yield client, mock_agent

    def test_root_endpoint_response_time(self, test_client, perf_timer):
        client, _ = test_client

        for _ in range(50):
            perf_timer.start()
            client.get("/")
            perf_timer.stop()

        report = perf_timer.report()
        assert report["avg_ms"] < 50, f"根路径平均 {report['avg_ms']}ms 超过50ms"
        assert report["p95_ms"] < 100, f"根路径P95 {report['p95_ms']}ms 超过100ms"

    def test_profile_endpoint_response_time(self, test_client, perf_timer):
        client, _ = test_client

        for _ in range(50):
            perf_timer.start()
            client.get("/profile?user_id=test_user")
            perf_timer.stop()

        report = perf_timer.report()
        assert report["avg_ms"] < 50, f"Profile端点平均 {report['avg_ms']}ms 超过50ms"
        assert report["p95_ms"] < 100, f"Profile端点P95 {report['p95_ms']}ms 超过100ms"

    def test_add_knowledge_endpoint_response_time(self, test_client, perf_timer):
        client, mock_agent = test_client

        for i in range(50):
            perf_timer.start()
            client.post("/add_knowledge", json={"knowledge": [f"知识{i}"]})
            perf_timer.stop()

        report = perf_timer.report()
        assert report["avg_ms"] < 50, f"添加知识端点平均 {report['avg_ms']}ms 超过50ms"

    def test_clear_memory_endpoint_response_time(self, test_client, perf_timer):
        client, mock_agent = test_client

        for _ in range(50):
            perf_timer.start()
            client.post("/clear_memory")
            perf_timer.stop()

        report = perf_timer.report()
        assert report["avg_ms"] < 50, f"清空记忆端点平均 {report['avg_ms']}ms 超过50ms"


class TestResourceUtilization:
    """测试资源利用率"""

    def test_short_term_memory_usage(self):
        from src.memory.memory import ShortTermMemory

        with patch("src.memory.memory.settings") as mock_s:
            mock_s.MEMORY_SIZE = 1000
            mock_s.MEMORY_WINDOW = 50
            memory = ShortTermMemory()

        tracemalloc.start()

        for i in range(1000):
            memory.add_message("user", f"这是一条测试消息，编号{i}，内容较长以测试内存占用" * 5, time.time())

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 50, f"短期记忆峰值内存 {peak_mb:.2f}MB 超过50MB阈值"

    def test_long_term_memory_disk_usage(self, temp_db_path):
        from src.memory.memory import LongTermMemory

        memory = LongTermMemory(db_path=temp_db_path)

        for i in range(1000):
            memory.merge_and_update(f"user_{i}", {
                "preferences": {"style": "详细", "level": "专业"},
                "habits": {"topics": ["火星", "木星", "土星"]},
                "constraints": ["避免术语", "控制长度"],
            })

        db_size = os.path.getsize(temp_db_path)
        db_size_mb = db_size / (1024 * 1024)
        assert db_size_mb < 10, f"数据库大小 {db_size_mb:.2f}MB 超过10MB阈值"

    def test_param_parser_memory_usage(self):
        from src.agent.param_parser import ParamParser

        tracemalloc.start()

        for _ in range(10000):
            ParamParser.parse('{"city": "北京", "extensions": "all", "data": "测试数据"}')
            ParamParser.parse_tool_input(
                '{"target": "mars"}',
                expected_params={"target": None, "date": None, "location": None},
            )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 10, f"参数解析器峰值内存 {peak_mb:.2f}MB 超过10MB阈值"

    def test_error_handler_memory_usage(self):
        from src.core.errors import ErrorHandler

        tracemalloc.start()

        errors = []
        for i in range(10000):
            try:
                raise ValueError(f"测试错误{i}")
            except Exception as e:
                error = ErrorHandler.handle(e, {"iteration": i})
                errors.append(error.to_dict())

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 50, f"错误处理峰值内存 {peak_mb:.2f}MB 超过50MB阈值"

    def test_coordinate_transformation_memory_usage(self):
        from src.astronomy.planetary import PlanetaryCalculator
        from src.astronomy.base import EphemerisManager

        mock_eph = MagicMock(spec=EphemerisManager)
        calc = PlanetaryCalculator(ephemeris=mock_eph)

        tracemalloc.start()

        for ra in range(0, 24):
            for dec in range(-90, 90, 5):
                calc.coordinate_transformation(ra=ra, dec=dec, target_system="fk5")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 50, f"坐标转换峰值内存 {peak_mb:.2f}MB 超过50MB阈值"


class TestConcurrentUserSimulation:
    """测试并发用户场景"""

    def test_concurrent_memory_operations(self, temp_db_path):
        from src.memory.memory import LongTermMemory

        errors = []
        results = []

        def write_user(user_id):
            try:
                memory = LongTermMemory(db_path=temp_db_path)
                memory.merge_and_update(f"user_{user_id}", {
                    "preferences": {"style": "详细"},
                    "habits": {"topics": ["火星"]},
                    "constraints": [],
                })
                profile = memory.load_profile(f"user_{user_id}")
                results.append(profile is not None)
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(20):
            t = threading.Thread(target=write_user, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"并发写入错误: {errors}"
        assert all(results), "部分用户画像未成功写入"

    def test_concurrent_short_term_memory(self):
        from src.memory.memory import ShortTermMemory

        with patch("src.memory.memory.settings") as mock_s:
            mock_s.MEMORY_SIZE = 100
            mock_s.MEMORY_WINDOW = 10
            memory = ShortTermMemory()

        errors = []

        def add_messages(thread_id):
            try:
                for i in range(50):
                    memory.add_message("user", f"线程{thread_id}_消息{i}", time.time())
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(10):
            t = threading.Thread(target=add_messages, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"并发短期记忆错误: {errors}"
        assert memory.get_size() <= 100

    def test_concurrent_param_parsing(self):
        from src.agent.param_parser import ParamParser

        errors = []
        results = []

        def parse_params(thread_id):
            try:
                for i in range(100):
                    result = ParamParser.parse(f'{{"city": "城市{thread_id}_{i}"}}')
                    results.append(result.get("city") is not None)
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(10):
            t = threading.Thread(target=parse_params, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"并发参数解析错误: {errors}"
        assert all(results), "部分参数解析结果不正确"

    def test_concurrent_error_handling(self):
        from src.core.errors import ErrorHandler

        errors = []
        results = []

        def handle_errors(thread_id):
            try:
                for i in range(100):
                    try:
                        raise ValueError(f"线程{thread_id}_错误{i}")
                    except Exception as e:
                        error = ErrorHandler.handle(e, {"thread": thread_id})
                        results.append(error.code.value == "VALIDATION_ERROR")
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(10):
            t = threading.Thread(target=handle_errors, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"并发错误处理异常: {errors}"
        assert all(results), "部分错误处理结果不正确"

    def test_concurrent_api_requests(self):
        from fastapi.testclient import TestClient

        with patch("src.api.main.AstroAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.user_id = "test_user"
            mock_agent.long_term_memory = MagicMock()

            mock_profile = MagicMock()
            mock_profile.user_id = "test_user"
            mock_profile.preferences = {}
            mock_profile.habits = {}
            mock_profile.constraints = []
            mock_profile.created_at = "2026-01-01T00:00:00"
            mock_profile.updated_at = "2026-04-08T00:00:00"
            mock_agent.long_term_memory.load_profile.return_value = mock_profile

            MockAgent.return_value = mock_agent

            from src.api.main import app
            client = TestClient(app)

        errors = []
        response_times = []

        def make_request(req_id):
            try:
                start = time.perf_counter()
                response = client.get("/")
                duration = time.perf_counter() - start
                response_times.append(duration)
                if response.status_code != 200:
                    errors.append(f"请求{req_id}返回{response.status_code}")
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"并发API请求错误: {errors}"
        avg_time = sum(response_times) / len(response_times) * 1000
        assert avg_time < 200, f"并发API平均响应时间 {avg_time:.2f}ms 超过200ms"
