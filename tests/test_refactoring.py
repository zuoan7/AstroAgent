# -*- coding: utf-8 -*-
"""
自动化测试套件 - 验证重构后的代码正确性和向后兼容性
"""

import unittest
import sys
import os
from datetime import datetime

# 确保项目根目录在sys.path中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestHelpersModule(unittest.TestCase):
    """测试 utils/helpers.py 工具函数"""
    
    def test_parse_mixed_input_dict(self):
        """测试字典输入解析"""
        from utils.helpers import parse_mixed_input
        
        input_dict = {"planet_name": "mars", "latitude": 39.9}
        result = parse_mixed_input(input_dict, {"planet_name": None, "latitude": None})
        
        self.assertEqual(result["planet_name"], "mars")
        self.assertEqual(result["latitude"], 39.9)
    
    def test_parse_mixed_input_string_json(self):
        """测试JSON字符串输入解析"""
        from utils.helpers import parse_mixed_input
        
        input_str = '{"city": "北京", "extensions": "all"}'
        result = parse_mixed_input(input_str, {"city": None, "extensions": None})
        
        self.assertEqual(result["city"], "北京")
        self.assertEqual(result["extensions"], "all")
    
    def test_parse_mixed_input_invalid(self):
        """测试无效输入处理"""
        from utils.helpers import parse_mixed_input
        
        result = parse_mixed_input("invalid string", {"key": None})
        self.assertEqual(result, {})
        
        result = parse_mixed_input(None)
        self.assertEqual(result, {})
    
    def test_parse_date_none(self):
        """测试None日期解析返回当前时间"""
        from utils.helpers import parse_date
        
        result = parse_date(None)
        self.assertIsInstance(result, datetime)
        # 应该是最近的时间（1秒内）
        time_diff = abs((datetime.now() - result).total_seconds())
        self.assertLessEqual(time_diff, 1.0)
    
    def test_parse_date_datetime_object(self):
        """测试datetime对象直接返回"""
        from utils.helpers import parse_date
        
        dt = datetime(2026, 4, 7, 12, 0, 0)
        result = parse_date(dt)
        
        self.assertEqual(result, dt)
    
    def test_parse_date_iso_format(self):
        """测试ISO格式日期字符串"""
        from utils.helpers import parse_date
        
        result = parse_date("2026-04-07T12:00:00")
        self.assertEqual(result.year, 2026)
        self.assertEqual(result.month, 4)
        self.assertEqual(result.day, 7)
    
    def test_parse_date_standard_format(self):
        """测试标准格式日期字符串"""
        from utils.helpers import parse_date
        
        result = parse_date("2026-04-07")
        self.assertEqual(result.year, 2026)
        self.assertEqual(result.month, 4)
        self.assertEqual(result.day, 7)
    
    def test_parse_date_natural_language(self):
        """测试自然语言日期"""
        from utils.helpers import parse_date
        
        today = datetime.now()
        
        # 测试"今天"
        result_today = parse_date("今天")
        self.assertEqual(result_today.date(), today.date())
        
        # 测试"明天"
        result_tomorrow = parse_date("明天")
        expected_tomorrow = today.replace(day=today.day + 1) if today.day < 28 else today
        self.assertEqual(result_tomorrow.date(), expected_tomorrow.date())
    
    def test_parse_coordinate_string_standard(self):
        """测试标准坐标格式解析"""
        from utils.helpers import parse_coordinate_string
        
        result = parse_coordinate_string("39.9042,116.4074")
        
        self.assertIsNotNone(result)
        lat, lon = result
        self.assertAlmostEqual(lat, 39.9042)
        self.assertAlmostEqual(lon, 116.4074)
    
    def test_coordinate_string_key_value(self):
        """测试键值对坐标格式"""
        from utils.helpers import parse_coordinate_string
        
        result = parse_coordinate_string("latitude=39.9, longitude=116.4")
        
        self.assertIsNotNone(result)
        lat, lon = result
        self.assertAlmostEqual(lat, 39.9)
        self.assertAlmostEqual(lon, 116.4)
    
    def test_is_coordinates_valid(self):
        """测试有效坐标检测"""
        from utils.helpers import is_coordinates
        
        self.assertTrue(is_coordinates("39.9,116.4"))
        self.assertTrue(is_coordinates("latitude=39.9, longitude=116.4"))
        self.assertFalse(is_coordinates(""))
        self.assertFalse(is_coordinates(None))
        self.assertFalse(is_coordinates("invalid"))
    
    def test_safe_float_conversion(self):
        """测试安全浮点数转换"""
        from utils.helpers import safe_float
        
        self.assertEqual(safe_float(3.14), 3.14)
        self.assertEqual(safe_float("2.71"), 2.71)
        self.assertIsNone(safe_float(None))
        self.assertIsNone(safe_float("invalid"))
        self.assertEqual(safe_float("invalid", default=0.0), 0.0)
    
    def test_safe_int_conversion(self):
        """测试安全整数转换"""
        from utils.helpers import safe_int
        
        self.assertEqual(safe_int(42), 42)
        self.assertEqual(safe_int("123"), 123)
        self.assertIsNone(safe_int(None))
        self.assertIsNone(safe_int("invalid"))
    
    def test_shorten_text(self):
        """测试文本截断"""
        from utils.helpers import shorten_text
        
        long_text = "x" * 2000
        result = shorten_text(long_text, max_len=100)
        
        self.assertLessEqual(len(result), 100)
        self.assertTrue(result.endswith("..."))
        
        # 测试短文本不截断
        short_text = "hello"
        result = shorten_text(short_text, max_len=100)
        self.assertEqual(result, short_text)
    
    def test_get_direction_from_azimuth(self):
        """测试方位角转方向"""
        from utils.helpers import get_direction_from_azimuth
        
        self.assertEqual(get_direction_from_azimuth(0), "北")
        self.assertEqual(get_direction_from_azimuth(90), "东")
        self.assertEqual(get_direction_from_azimuth(180), "南")
        self.assertEqual(get_direction_from_azimuth(270), "西")


class TestConstantsModule(unittest.TestCase):
    """测试 constants.py 常量定义"""
    
    def test_planet_mapping_complete(self):
        """测试行星映射完整性"""
        from constants import PLANET_MAPPING, VALID_PLANETS
        
        expected_planets = {'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'}
        self.assertEqual(set(PLANET_MAPPING.keys()), expected_planets)
        self.assertEqual(VALID_PLANETS, expected_planets)
    
    def test_planet_mapping_values(self):
        """测试行星映射值正确性"""
        from constants import PLANET_MAPPING
        
        self.assertEqual(PLANET_MAPPING['mercury'], 199)
        self.assertEqual(PLANET_MAPPING['venus'], 299)
        self.assertEqual(PLANET_MAPPING['mars'], 499)
        self.assertEqual(PLANET_MAPPING['jupiter'], 5)
        self.assertEqual(PLANET_MAPPING['saturn'], 6)
    
    def test_celestial_name_mapping(self):
        """测试天体名称映射"""
        from constants import CELESTIAL_NAME_MAPPING
        
        self.assertIn('仙女座星系', CELESTIAL_NAME_MAPPING)
        self.assertEqual(CELESTIAL_NAME_MAPPING['仙女座星系'], 'M31')
        self.assertIn('天狼星', CELESTIAL_NAME_MAPPING)
    
    def test_supported_bodies(self):
        """测试支持的天体列表"""
        from constants import SUPPORTED_BODIES
        
        self.assertIn('sun', SUPPORTED_BODIES)
        self.assertIn('moon', SUPPORTED_BODIES)
        self.assertIn('mercury', SUPPORTED_BODIES)
        self.assertEqual(len(SUPPORTED_BODIES), 7)


class TestAstronomyModules(unittest.TestCase):
    """测试 astronomy 子模块导入和基本功能"""
    
    def test_import_astronomy_package(self):
        """测试 astronomy 包可以正常导入"""
        try:
            from astronomy import (
                AstronomyTools,
                AstronomyEventsPredictor,
                EphemerisManager,
                PlanetaryCalculator,
                CelestialDatabaseService,
                NASAAPIService,
                WeatherService,
                SearchService,
                EventsPredictor,
            )
            
            self.assertTrue(True)  # 导入成功
        except ImportError as e:
            self.fail(f"无法导入 astronomy 包: {e}")
    
    def test_ephemeris_manager_singleton(self):
        """测试星历管理器单例模式"""
        from astronomy.base import EphemerisManager
        
        instance1 = EphemerisManager()
        instance2 = EphemerisManager()
        
        # 应该是同一个实例
        self.assertIs(instance1, instance2)
    
    def test_backward_compatibility_astronomy_tools(self):
        """测试 AstronomyTools 向后兼容性"""
        from astronomy import AstronomyTools
        
        tools = AstronomyTools()
        
        # 检查兼容属性存在
        self.assertTrue(hasattr(tools, 'data_loaded'))
        self.assertTrue(hasattr(tools, 'planetary'))
        self.assertTrue(hasattr(tools, 'celestial_db'))
        self.assertTrue(hasattr(tools, 'nasa_api'))
        self.assertTrue(hasattr(tools, 'weather'))
        self.assertTrue(hasattr(tools, 'search'))
        self.assertTrue(hasattr(tools, 'events'))
        
        # 检查方法存在
        self.assertTrue(callable(getattr(tools, 'get_planet_position', None)))
        self.assertTrue(callable(getattr(tools, 'coordinate_transformation', None)))
        self.assertTrue(callable(getattr(tools, 'get_weather', None)))
        self.assertTrue(callable(getattr(tools, 'web_search', None)))
    
    def test_backward_compatibility_events_predictor(self):
        """测试 AstronomyEventsPredictor 向后兼容性"""
        from astronomy import AstronomyEventsPredictor
        
        predictor = AstronomyEventsPredictor()
        
        # 检查兼容属性
        self.assertTrue(hasattr(predictor, 'planets'))
        self.assertTrue(hasattr(predictor, '_predictor'))
        
        # 检查方法
        self.assertTrue(callable(getattr(predictor, 'get_moon_phase', None)))
        self.assertTrue(callable(getattr(predictor, 'get_weekly_events', None)))
        self.assertTrue(callable(getattr(predictor, 'get_monthly_events', None)))


class TestPlanetaryCalculator(unittest.TestCase):
    """测试行星计算器"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        from astronomy.base import EphemerisManager
        cls.ephemeris = EphemerisManager()
        
        if not cls.ephemeris.is_loaded:
            raise unittest.SkipTest("星历数据未加载，跳过行星计算测试")
        
        from astronomy.planetary import PlanetaryCalculator
        cls.calculator = PlanetaryCalculator(cls.ephemeris)
    
    def test_get_planet_position_valid(self):
        """测试获取有效行星位置"""
        result = self.calculator.get_planet_position('mars')
        
        self.assertIsInstance(result, dict)
        self.assertIn('ra', result)
        self.assertIn('dec', result)
        self.assertIn('distance_au', result)
        
        # 值应该是数值类型
        self.assertIsInstance(result['ra'], (int, float))
        self.assertIsInstance(result['dec'], (int, float))
    
    def test_get_planet_position_invalid(self):
        """测试获取无效行星名称"""
        result = self.calculator.get_planet_position('invalid_planet')
        
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
    
    def test_coordinate_transformation_icrs_to_fk5(self):
        """测试ICRS到FK5坐标转换"""
        result = self.calculator.coordinate_transformation(
            ra=10.5,
            dec=20.5,
            epoch='J2000',
            target_system='fk5'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('ra', result)
        self.assertIn('dec', result)
    
    def test_coordinate_transformation_icrs_to_icrs(self):
        """测试ICRS到ICRS转换（应保持不变）"""
        result = self.calculator.coordinate_transformation(
            ra=10.5,
            dec=20.5,
            target_system='icrs'
        )
        
        self.assertAlmostEqual(result['ra'], 10.5)
        self.assertAlmostEqual(result['dec'], 20.5)
    
    def test_rise_set_times_sun(self):
        """测试太阳升降时间计算"""
        result = self.calculator.get_rise_set_times(
            body_name='sun',
            latitude=39.9,
            longitude=116.4
        )
        
        self.assertIsInstance(result, dict)
        # 结果应该包含rise_time和set_time，或者包含error字段
        self.assertTrue('rise_time' in result or 'error' in result)


class TestWeatherService(unittest.TestCase):
    """测试天气服务"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        from astronomy.weather_service import WeatherService
        cls.service = WeatherService()
    
    def test_reverse_geocode_valid_coords(self):
        """测试有效坐标逆地理编码"""
        result = self.service.reverse_geocode(116.4074, 39.9042)
        
        # 北京的经纬度应该能返回城市名
        if result:  # 如果API可用
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)
    
    def test_is_coordinates_method(self):
        """测试坐标检测方法（通过AstronomyTools）"""
        from astronomy import AstronomyTools
        tools = AstronomyTools()
        
        self.assertTrue(tools._is_coordinates("39.9,116.4"))
        self.assertFalse(tools._is_coordinates("invalid"))


class TestEventsPredictor(unittest.TestCase):
    """测试天象预测器"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        from astronomy.events_predictor import EventsPredictor
        cls.predictor = EventsPredictor()
    
    def test_get_moon_phase_returns_tuple(self):
        """测试月相计算返回元组"""
        today = datetime.now()
        result = self.predictor.get_moon_phase(today)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)  # 月相名称
        self.assertIsInstance(result[1], str)  # 月相描述
    
    def test_get_moon_phase_contains_emoji(self):
        """测试月相名称包含emoji"""
        today = datetime.now()
        phase_name, _ = self.predictor.get_moon_phase(today)
        
        # 应该包含月亮emoji
        self.assertTrue(any(char in phase_name for char in ['🌑', '🌒', '🌓', '🌔', 
                                                            '🌕', '🌖', '🌗', '🌘']))
    
    def test_get_weekly_events_returns_string(self):
        """测试周预报返回字符串"""
        result = self.predictor.get_weekly_events()
        
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        # 应该包含标题
        self.assertIn("一周天象预报", result)
    
    def test_get_monthly_events_returns_string(self):
        """测试月预报返回字符串"""
        result = self.predictor.get_monthly_events(year=2026, month=4)
        
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        # 应该包含月份信息
        self.assertIn("4月", result) or self.assertIn("天象预报", result)
    
    def test_get_visible_planets_returns_list(self):
        """测试可见行星返回列表"""
        today = datetime.now()
        result = self.predictor.get_visible_planets(today)
        
        self.assertIsInstance(result, list)
        # 列表中的每个元素应该是字符串
        for item in result:
            self.assertIsInstance(item, str)


class TestIntegration(unittest.TestCase):
    """集成测试 - 验证整体功能"""
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        from astronomy import AstronomyTools, AstronomyEventsPredictor
        
        # 1. 创建工具实例
        tools = AstronomyTools()
        predictor = AstronomyEventsPredictor()
        
        # 2. 使用工具查询行星位置（如果数据已加载）
        if tools.data_loaded:
            pos = tools.get_planet_position('jupiter')
            self.assertIsInstance(pos, dict)
            if 'error' not in pos:
                self.assertIn('ra', pos)
        
        # 3. 获取周预报
        weekly = predictor.get_weekly_events()
        self.assertIsInstance(weekly, str)
        self.assertTrue(len(weekly) > 0)
        
        # 4. 获取月相
        moon = predictor.get_moon_phase(datetime.now())
        self.assertIsInstance(moon, tuple)
    
    def test_module_imports_work(self):
        """测试所有模块都可以正常导入"""
        modules_to_test = [
            'utils.helpers',
            'constants',
            'astronomy',
            'astronomy.base',
            'astronomy.planetary',
            'astronomy.celestial_databases',
            'astronomy.nasa_api',
            'astronomy.weather_service',
            'astronomy.search_service',
            'astronomy.events_predictor',
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                self.fail(f"无法导入模块 {module_name}: {e}")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
