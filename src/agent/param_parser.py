"""
统一参数解析器
用于处理各种格式的工具输入，减少重复代码
合并了原 helpers.py 中的参数解析、日期处理和辅助功能
"""

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from src.core.errors import ErrorHandler


class ParamParser:
    """统一的参数解析器，用于处理各种格式的工具输入"""

    @staticmethod
    def parse(input_data: Any, primary_param: Optional[str] = None) -> Dict[str, Any]:
        """
        统一解析输入参数，支持多种格式：
        - dict: 直接返回
        - str (JSON): 解析为dict
        - str (非JSON): 作为primary_param的值

        Args:
            input_data: 输入数据
            primary_param: 主参数名称，当输入为非JSON字符串时使用

        Returns:
            解析后的参数字典
        """
        if isinstance(input_data, dict):
            return input_data

        if isinstance(input_data, str):
            text = input_data.strip()
            if text.startswith('{'):
                try:
                    clean_text = text.split('#')[0].strip()
                    data = json.loads(clean_text)
                    if isinstance(data, dict):
                        return data
                except (json.JSONDecodeError, ValueError):
                    pass

            if primary_param:
                return {primary_param: input_data}
            return {"query": input_data}

        return {}

    @staticmethod
    def extract_param(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
        """
        从字典中提取参数，支持多个可能的键名

        Args:
            data: 参数字典
            keys: 可能的键名列表
            default: 默认值

        Returns:
            找到的第一个非None值，或默认值
        """
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return default

    @staticmethod
    def parse_tool_input(
        input_data: Any,
        expected_params: Optional[Dict[str, Any]] = None,
        primary_param: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        解析工具输入，支持多种格式并进行验证

        Args:
            input_data: 输入数据（可以是dict、str、JSON字符串等）
            expected_params: 期望的参数及其默认值
            primary_param: 主参数名称

        Returns:
            解析后的参数字典
        """
        try:
            parsed = ParamParser.parse(input_data, primary_param)

            if expected_params:
                result = {}
                for param_name, default_value in expected_params.items():
                    result[param_name] = parsed.get(param_name, default_value)
                return result

            return parsed

        except Exception as e:
            raise ErrorHandler.create_param_error(
                param_name="input",
                error_message=str(e),
                details={"input_type": type(input_data).__name__}
            )

    @staticmethod
    def normalize_location(location: Any) -> Optional[str]:
        """
        规范化位置输入，支持字典、JSON字符串和普通字符串

        Args:
            location: 位置输入

        Returns:
            规范化后的位置字符串
        """
        if location is None:
            return None

        if isinstance(location, dict):
            return (
                location.get("location") or
                location.get("city") or
                location.get("adcode") or
                location.get("citycode")
            )

        if isinstance(location, str):
            text = location.strip()
            if text.startswith('{') and text.endswith('}'):
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        return (
                            obj.get("location") or
                            obj.get("city") or
                            obj.get("adcode") or
                            obj.get("citycode")
                        )
                except (json.JSONDecodeError, ValueError):
                    pass
            return text

        return str(location)

    @staticmethod
    def normalize_date(date_str: Any, default: Optional[datetime] = None) -> Optional[str]:
        """
        规范化日期字符串，返回 YYYY-MM-DD 格式

        Args:
            date_str: 日期输入
            default: 默认值

        Returns:
            YYYY-MM-DD 格式的日期字符串，或 None
        """
        if date_str is None:
            if default is not None:
                return default.strftime("%Y-%m-%d")
            return None

        if isinstance(date_str, datetime):
            return date_str.strftime("%Y-%m-%d")

        if not isinstance(date_str, str):
            return None

        text = date_str.strip()

        natural_dates = {
            '今天': 0, '今日': 0, 'today': 0,
            '明天': 1, '次日': 1, 'tomorrow': 1,
        }

        text_lower = text.lower()
        if text_lower in natural_dates:
            base = default or datetime.now()
            return (base + timedelta(days=natural_dates[text_lower])).strftime("%Y-%m-%d")

        m = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", text)
        if m:
            return m.group(1).replace("/", "-")

        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(text, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        try:
            dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

        return None

    @staticmethod
    def parse_date(date_str: Any, default: Optional[datetime] = None) -> datetime:
        """
        统一的日期解析函数，支持多种格式，返回 datetime 对象

        Args:
            date_str: 日期输入，可以是字符串、datetime对象或None
            default: 默认值，默认为当前时间

        Returns:
            datetime对象
        """
        if default is None:
            default = datetime.now()

        if date_str is None:
            return default

        if isinstance(date_str, datetime):
            return date_str

        if not isinstance(date_str, str):
            return default

        text = date_str.strip()

        natural_dates = {
            '今天': 0, '今日': 0, 'today': 0,
            '明天': 1, '次日': 1, 'tomorrow': 1,
        }

        if text.lower() in natural_dates:
            return default + timedelta(days=natural_dates[text.lower()])

        try:
            dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
            return dt
        except ValueError:
            pass

        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue

        match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', text)
        if match:
            try:
                candidate = match.group(1).replace('/', '-')
                return datetime.strptime(candidate, "%Y-%m-%d")
            except ValueError:
                pass

        return default

    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        """
        安全转换为整数

        Args:
            value: 输入值
            default: 转换失败的默认值

        Returns:
            整数或默认值
        """
        if value is None:
            return default

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        if isinstance(value, str):
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return default

        return default

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """
        安全转换为浮点数

        Args:
            value: 输入值
            default: 转换失败的默认值

        Returns:
            浮点数或默认值
        """
        if value is None:
            return default

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        return default

    @staticmethod
    def safe_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
        """
        安全转换为布尔值

        Args:
            value: 输入值
            default: 转换失败的默认值

        Returns:
            布尔值或默认值
        """
        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            lower_val = value.lower().strip()
            if lower_val in ('true', '1', 'yes'):
                return True
            elif lower_val in ('false', '0', 'no'):
                return False
            return default

        return default

    @staticmethod
    def parse_mixed_input(value: Any, expected_params: Optional[Dict] = None) -> Dict:
        """
        统一处理多种格式的输入（dict/str/其他）

        Args:
            value: 输入值，可以是字典、字符串或其他类型
            expected_params: 期望的参数列表

        Returns:
            解析后的参数字典
        """
        if isinstance(value, dict):
            if expected_params:
                return {k: value.get(k) for k in expected_params.keys()}
            return value

        elif isinstance(value, str):
            parsed = ParamParser.parse_json_string(value)
            if parsed:
                if expected_params:
                    return {k: parsed.get(k) for k in expected_params.keys()}
                return parsed

        return {}

    @staticmethod
    def parse_json_string(text: str) -> Optional[Dict]:
        """
        安全解析JSON字符串

        Args:
            text: JSON字符串

        Returns:
            解析后的字典或None
        """
        if not text or not isinstance(text, str):
            return None

        text = text.strip()

        if not text.startswith('{'):
            return None

        try:
            clean_text = text.split('#')[0].strip()
            data = json.loads(clean_text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return None

    @staticmethod
    def parse_coordinate_string(text: str) -> Optional[tuple]:
        """
        解析坐标字符串（如 "latitude=39.9, longitude=116.4" 或 "39.9,116.4"）

        Args:
            text: 坐标字符串

        Returns:
            (lat, lon) 元组，失败返回None
        """
        if not text or not isinstance(text, str):
            return None

        text = text.strip()

        lat_match = re.search(r'(?:latitude|lat)\s*=\s*([-\d.]+)', text, re.IGNORECASE)
        lon_match = re.search(r'(?:longitude|lon)\s*=\s*([-\d.]+)', text, re.IGNORECASE)

        if lat_match and lon_match:
            try:
                lat = float(lat_match.group(1))
                lon = float(lon_match.group(1))
                return (lat, lon)
            except ValueError:
                pass

        parts = text.split(',')
        if len(parts) == 2:
            try:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            except ValueError:
                pass

        return None

    @staticmethod
    def is_coordinates(text: str) -> bool:
        """
        检测字符串是否为经纬度格式

        Args:
            text: 待检测字符串

        Returns:
            是否为有效的经纬度格式
        """
        result = ParamParser.parse_coordinate_string(text)
        return result is not None

    @staticmethod
    def extract_key_value_pairs(text: str, keys: list) -> Dict[str, Any]:
        """
        从字符串中提取键值对（如 "ra=10.5, dec=20.5"）

        Args:
            text: 包含键值对的字符串
            keys: 要提取的键列表

        Returns:
            提取到的键值对字典
        """
        if not text or not isinstance(text, str):
            return {}

        result = {}
        for key in keys:
            pattern = rf'{key}\s*=\s*([-\d.]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    result[key] = float(match.group(1))
                except ValueError:
                    result[key] = match.group(1)

        return result

    @staticmethod
    def shorten_text(text: Any, max_len: int = 1200) -> str:
        """
        截断文本到指定长度

        Args:
            text: 输入文本（任意类型）
            max_len: 最大长度

        Returns:
            截断后的字符串
        """
        if text is None:
            return ""

        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)

        if len(s) <= max_len:
            return s

        return s[:max_len - 3] + "..."

    @staticmethod
    def get_direction_from_azimuth(az_degrees: float) -> str:
        """
        将方位角转为中文方向

        Args:
            az_degrees: 方位角（度）

        Returns:
            中文方向（北/东/南/西）
        """
        if az_degrees < 45 or az_degrees >= 315:
            return "北"
        elif az_degrees < 135:
            return "东"
        elif az_degrees < 225:
            return "南"
        else:
            return "西"

    @staticmethod
    def extract_image_url(text: str) -> Optional[str]:
        """
        从文本中提取图片URL

        Args:
            text: 包含URL的文本

        Returns:
            图片URL或None
        """
        if not text or not isinstance(text, str):
            return None

        pattern = r"(https?://\S+\.(?:png|jpg|jpeg|webp))"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None
