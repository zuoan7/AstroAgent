# -*- coding: utf-8 -*-
"""
工具函数模块 - 统一的参数解析、日期处理和辅助功能
"""

import re
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union


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
        parsed = parse_json_string(value)
        if parsed:
            if expected_params:
                return {k: parsed.get(k) for k in expected_params.keys()}
            return parsed
    
    return {}


def parse_json_string(text: str) -> Optional[Dict]:
    """
    解析JSON格式的字符串
    
    Args:
        text: JSON字符串
        
    Returns:
        解析后的字典，失败返回None
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    if not (text.startswith('{') and text.endswith('}')):
        return None
    
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    
    return None


def parse_date(date_str: Any, default: Optional[datetime] = None) -> datetime:
    """
    统一的日期解析函数，支持多种格式
    
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
    
    # 处理自然语言日期
    natural_dates = {
        '今天': 0, '今日': 0, 'today': 0,
        '明天': 1, '次日': 1, 'tomorrow': 1,
    }
    
    if text.lower() in natural_dates:
        return default + timedelta(days=natural_dates[text.lower()])
    
    # 尝试ISO格式
    try:
        dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
        return dt
    except ValueError:
        pass
    
    # 尝试常见格式
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
    
    # 从文本中提取日期（如 "2026-08-01 天象预报"）
    match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', text)
    if match:
        try:
            candidate = match.group(1).replace('/', '-')
            return datetime.strptime(candidate, "%Y-%m-%d")
        except ValueError:
            pass
    
    return default


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
    
    # 尝试 "key=value, key=value" 格式
    lat_match = re.search(r'(?:latitude|lat)\s*=\s*([-\d.]+)', text, re.IGNORECASE)
    lon_match = re.search(r'(?:longitude|lon)\s*=\s*([-\d.]+)', text, re.IGNORECASE)
    
    if lat_match and lon_match:
        try:
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))
            return (lat, lon)
        except ValueError:
            pass
    
    # 尝试 "lat,lon" 格式
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


def is_coordinates(text: str) -> bool:
    """
    检测字符串是否为经纬度格式
    
    Args:
        text: 待检测字符串
        
    Returns:
        是否为有效的经纬度格式
    """
    result = parse_coordinate_string(text)
    return result is not None


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


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
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


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
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


def normalize_location(location: Optional[str]) -> Optional[str]:
    """
    标准化位置描述
    
    Args:
        location: 位置信息
        
    Returns:
        标准化后的位置字符串
    """
    if location is None:
        return None
    
    if isinstance(location, dict):
        return location.get('city') or location.get('location')
    
    return str(location).strip()


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
