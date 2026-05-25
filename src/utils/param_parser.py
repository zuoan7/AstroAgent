"""
统一参数解析器
用于处理各种格式的工具输入，减少重复代码
合并了原 helpers.py 中的参数解析、日期处理和辅助功能
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.errors import ErrorHandler

logger = logging.getLogger("AstroAgent")


class ParseErrorType(Enum):
    DATE_PARSE_ERROR = "DATE_PARSE_ERROR"
    BOOL_PARSE_ERROR = "BOOL_PARSE_ERROR"
    COORDINATE_PARSE_ERROR = "COORDINATE_PARSE_ERROR"
    JSON_PARSE_ERROR = "JSON_PARSE_ERROR"
    TYPE_CONVERT_ERROR = "TYPE_CONVERT_ERROR"
    UNKNOWN_PARSE_ERROR = "UNKNOWN_PARSE_ERROR"


@dataclass
class ParseError:
    error_type: ParseErrorType
    position: str
    reason: str
    raw_value: Any = None
    details: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "error_type": self.error_type.value,
            "position": self.position,
            "reason": self.reason,
        }
        if self.raw_value is not None:
            result["raw_value"] = str(self.raw_value)
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        return f"[{self.error_type.value}] 位置: {self.position}, 原因: {self.reason}"


@dataclass
class ParseResult:
    success: bool
    value: Any = None
    error: Optional[ParseError] = None

    @staticmethod
    def ok(value: Any) -> "ParseResult":
        return ParseResult(success=True, value=value)

    @staticmethod
    def fail(
        error_type: ParseErrorType,
        position: str,
        reason: str,
        raw_value: Any = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> "ParseResult":
        error = ParseError(
            error_type=error_type,
            position=position,
            reason=reason,
            raw_value=raw_value,
            details=details or {},
        )
        logger.warning(str(error))
        return ParseResult(success=False, error=error)


_NATURAL_DATE_OFFSETS = {
    '今天': 0, '今日': 0, 'today': 0,
    '明天': 1, '次日': 1, 'tomorrow': 1,
    '后天': 2, 'the day after tomorrow': 2,
    '昨天': -1, 'yesterday': -1,
}

_TRUE_VALUES = {'true', '1', 'yes', '是', '开', '需要', '对', '真', '有'}
_FALSE_VALUES = {'false', '0', 'no', '否', '关', '不需要', '错', '假', '无'}


def _resolve_natural_date(text: str, base: datetime) -> Optional[datetime]:
    text_lower = text.lower().strip()

    if text_lower in _NATURAL_DATE_OFFSETS:
        return base + timedelta(days=_NATURAL_DATE_OFFSETS[text_lower])

    if text in ('今晚', '明晚'):
        offset = 0 if text == '今晚' else 1
        return base.replace(hour=20, minute=0, second=0, microsecond=0) + timedelta(days=offset)

    if text == '本周末':
        weekday = base.weekday()
        days_to_saturday = (5 - weekday) % 7
        if days_to_saturday == 0 and base.hour >= 18:
            days_to_saturday = 7
        return base + timedelta(days=days_to_saturday)

    if text == '下周一':
        weekday = base.weekday()
        days_to_monday = (7 - weekday) % 7
        if days_to_monday == 0:
            days_to_monday = 7
        return base + timedelta(days=days_to_monday)

    return None


def _dms_to_degrees(degrees: float, minutes: float, seconds: float, sign: str = '') -> float:
    value = abs(degrees) + abs(minutes) / 60.0 + abs(seconds) / 3600.0
    if sign and sign.upper() in ('S', 'W', '-'):
        value = -value
    elif degrees < 0:
        value = -value
    return value


def _parse_ra_to_degrees(ra_str: str) -> Optional[float]:
    h = m = s = 0.0
    m_h = re.match(r'(\d+(?:\.\d+)?)\s*h', ra_str, re.IGNORECASE)
    m_m = re.search(r'(\d+(?:\.\d+)?)\s*m', ra_str, re.IGNORECASE)
    m_s = re.search(r'(\d+(?:\.\d+)?)\s*s', ra_str, re.IGNORECASE)

    if m_h:
        h = float(m_h.group(1))
    elif re.match(r'(\d+(?:\.\d+)?)\s', ra_str):
        h = float(re.match(r'(\d+(?:\.\d+)?)', ra_str).group(1))

    if m_m:
        m = float(m_m.group(1))
    if m_s:
        s = float(m_s.group(1))

    if not m_h and not m_m and not m_s:
        return None

    return (h + m / 60.0 + s / 3600.0) * 15.0


def _parse_dec_to_degrees(dec_str: str) -> Optional[float]:
    sign = 1.0
    dec_clean = dec_str.strip()
    if dec_clean.startswith('+'):
        dec_clean = dec_clean[1:]
    elif dec_clean.startswith('-'):
        sign = -1.0
        dec_clean = dec_clean[1:]

    m_d = re.match(r'(\d+(?:\.\d+)?)\s*[°d]', dec_clean, re.IGNORECASE)
    m_m = re.search(r'(\d+(?:\.\d+)?)\s*[\'′m]', dec_clean, re.IGNORECASE)
    m_s = re.search(r'(\d+(?:\.\d+)?)\s*["″s]', dec_clean, re.IGNORECASE)

    if m_d:
        d = float(m_d.group(1))
    else:
        return None

    minutes = float(m_m.group(1)) if m_m else 0.0
    seconds = float(m_s.group(1)) if m_s else 0.0

    return sign * (d + minutes / 60.0 + seconds / 3600.0)


class ParamParser:
    """统一的参数解析器，用于处理各种格式的工具输入"""

    @staticmethod
    def parse(input_data: Any, primary_param: Optional[str] = None) -> Dict[str, Any]:
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
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"JSON解析失败，回退到字符串模式: {e}")

            if primary_param:
                return {primary_param: input_data}
            return {"query": input_data}

        return {}

    @staticmethod
    def extract_param(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
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
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"位置JSON解析失败: {e}")
            return text

        return str(location)

    @staticmethod
    def normalize_date(date_str: Any, default: Optional[datetime] = None) -> Optional[str]:
        if date_str is None:
            if default is not None:
                return default.strftime("%Y-%m-%d")
            return None

        if isinstance(date_str, datetime):
            return date_str.strftime("%Y-%m-%d")

        if not isinstance(date_str, str):
            return None

        text = date_str.strip()
        base = default or datetime.now()

        resolved = _resolve_natural_date(text, base)
        if resolved is not None:
            return resolved.strftime("%Y-%m-%d")

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
        if default is None:
            default = datetime.now()

        if date_str is None:
            return default

        if isinstance(date_str, datetime):
            return date_str

        if not isinstance(date_str, str):
            return default

        text = date_str.strip()

        resolved = _resolve_natural_date(text, default)
        if resolved is not None:
            return resolved

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
        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, int):
            return value != 0

        if isinstance(value, str):
            lower_val = value.lower().strip()
            if lower_val in _TRUE_VALUES:
                return True
            elif lower_val in _FALSE_VALUES:
                return False
            return default

        return default

    @staticmethod
    def safe_bool_with_result(value: Any, default: Optional[bool] = None) -> ParseResult:
        if value is None:
            if default is not None:
                return ParseResult.ok(default)
            return ParseResult.fail(
                ParseErrorType.BOOL_PARSE_ERROR,
                "safe_bool",
                "输入为None且无默认值",
                raw_value=value,
            )

        if isinstance(value, bool):
            return ParseResult.ok(value)

        if isinstance(value, int):
            return ParseResult.ok(value != 0)

        if isinstance(value, str):
            lower_val = value.lower().strip()
            if lower_val in _TRUE_VALUES:
                return ParseResult.ok(True)
            elif lower_val in _FALSE_VALUES:
                return ParseResult.ok(False)
            return ParseResult.fail(
                ParseErrorType.BOOL_PARSE_ERROR,
                "safe_bool",
                f"无法识别的布尔值: '{value}'",
                raw_value=value,
                details={"recognized_true": list(_TRUE_VALUES), "recognized_false": list(_FALSE_VALUES)},
            )

        return ParseResult.fail(
            ParseErrorType.BOOL_PARSE_ERROR,
            "safe_bool",
            f"不支持的类型: {type(value).__name__}",
            raw_value=value,
        )

    @staticmethod
    def parse_mixed_input(value: Any, expected_params: Optional[Dict] = None) -> Dict:
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
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"JSON字符串解析失败: {e}")

        return None

    @staticmethod
    def parse_coordinate_string(text: str) -> Optional[Tuple[float, float]]:
        if not text or not isinstance(text, str):
            return None

        text = text.strip()
        if not text:
            return None

        result = ParamParser._parse_coord_radec(text)
        if result:
            return result

        result = ParamParser._parse_coord_dms_with_direction(text)
        if result:
            return result

        result = ParamParser._parse_coord_decimal_with_direction(text)
        if result:
            return result

        result = ParamParser._parse_coord_key_value(text)
        if result:
            return result

        result = ParamParser._parse_coord_simple(text)
        if result:
            return result

        return None

    @staticmethod
    def parse_coordinate_with_result(text: str) -> ParseResult:
        if not text or not isinstance(text, str):
            return ParseResult.fail(
                ParseErrorType.COORDINATE_PARSE_ERROR,
                "parse_coordinate",
                "输入为空或非字符串类型",
                raw_value=text,
            )

        text = text.strip()
        if not text:
            return ParseResult.fail(
                ParseErrorType.COORDINATE_PARSE_ERROR,
                "parse_coordinate",
                "输入为空字符串",
                raw_value=text,
            )

        result = ParamParser.parse_coordinate_string(text)
        if result is not None:
            return ParseResult.ok(result)

        return ParseResult.fail(
            ParseErrorType.COORDINATE_PARSE_ERROR,
            "parse_coordinate",
            f"无法识别的坐标格式: '{text}'",
            raw_value=text,
            details={
                "supported_formats": [
                    "RA/Dec (如: RA 12h30m45s, Dec +45d15m30s)",
                    "度分秒 (如: 39°54'00\"N, 116°24'00\"E)",
                    "带方向小数 (如: 39.9N, 116.4E)",
                    "键值对 (如: lat=39.9, lon=116.4)",
                    "简单小数 (如: 39.9, 116.4)",
                ]
            },
        )

    @staticmethod
    def _parse_coord_radec(text: str) -> Optional[Tuple[float, float]]:
        ra_match = re.search(
            r'(?:RA|赤经|ra)\s*[:=]?\s*([0-9hms\d.\s°d\'\"″′]+)',
            text, re.IGNORECASE
        )
        dec_match = re.search(
            r'(?:Dec|赤纬|dec)\s*[:=]?\s*([+\-0-9dms\d.\s°\'\"″′]+)',
            text, re.IGNORECASE
        )

        if not ra_match or not dec_match:
            return None

        ra_deg = _parse_ra_to_degrees(ra_match.group(1))
        dec_deg = _parse_dec_to_degrees(dec_match.group(1))

        if ra_deg is None or dec_deg is None:
            return None

        if 0 <= ra_deg < 360 and -90 <= dec_deg <= 90:
            return (dec_deg, ra_deg)

        return None

    @staticmethod
    def _parse_coord_dms_with_direction(text: str) -> Optional[Tuple[float, float]]:
        dms_pattern = (
            r'(\d+(?:\.\d+)?)\s*[°d]\s*'
            r'(\d+(?:\.\d+)?)\s*[\'′m]\s*'
            r'(\d+(?:\.\d+)?)\s*["″s]'
            r'\s*([NSEW])'
        )
        matches = re.findall(dms_pattern, text, re.IGNORECASE)
        if len(matches) < 2:
            return None

        lat_val = None
        lon_val = None

        for deg_str, min_str, sec_str, direction in matches:
            d = float(deg_str)
            m = float(min_str)
            s = float(sec_str)
            value = d + m / 60.0 + s / 3600.0
            dir_upper = direction.upper()

            if dir_upper in ('N', 'S'):
                if dir_upper == 'S':
                    value = -value
                lat_val = value
            elif dir_upper in ('E', 'W'):
                if dir_upper == 'W':
                    value = -value
                lon_val = value

        if lat_val is not None and lon_val is not None:
            if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                return (lat_val, lon_val)

        return None

    @staticmethod
    def _parse_coord_decimal_with_direction(text: str) -> Optional[Tuple[float, float]]:
        pattern = r'(\d+(?:\.\d+)?)\s*([NSEW])'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if len(matches) < 2:
            return None

        lat_val = None
        lon_val = None

        for num_str, direction in matches:
            value = float(num_str)
            dir_upper = direction.upper()

            if dir_upper in ('N', 'S'):
                if dir_upper == 'S':
                    value = -value
                lat_val = value
            elif dir_upper in ('E', 'W'):
                if dir_upper == 'W':
                    value = -value
                lon_val = value

        if lat_val is not None and lon_val is not None:
            if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                return (lat_val, lon_val)

        return None

    @staticmethod
    def _parse_coord_key_value(text: str) -> Optional[Tuple[float, float]]:
        lat_match = re.search(r'(?:latitude|lat|纬度)\s*=\s*([-\d.]+)', text, re.IGNORECASE)
        lon_match = re.search(r'(?:longitude|lon|经度)\s*=\s*([-\d.]+)', text, re.IGNORECASE)

        if lat_match and lon_match:
            try:
                lat = float(lat_match.group(1))
                lon = float(lon_match.group(1))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            except ValueError:
                pass

        return None

    @staticmethod
    def _parse_coord_simple(text: str) -> Optional[Tuple[float, float]]:
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
        result = ParamParser.parse_coordinate_string(text)
        return result is not None

    @staticmethod
    def extract_key_value_pairs(text: str, keys: list) -> Dict[str, Any]:
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
        if text is None:
            return ""

        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)

        if len(s) <= max_len:
            return s

        return s[:max_len - 3] + "..."

    @staticmethod
    def get_direction_from_azimuth(az_degrees: float) -> str:
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
        if not text or not isinstance(text, str):
            return None

        pattern = r"(https?://\S+\.(?:png|jpg|jpeg|webp))"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None
