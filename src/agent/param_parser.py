"""
统一参数解析器
用于处理各种格式的工具输入，减少重复代码
"""

import json
import re
from typing import Any, Dict, Optional, List
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
    def parse_json_string(text: str) -> Optional[Dict[str, Any]]:
        """尝试解析JSON字符串"""
        if not isinstance(text, str):
            return None
        
        text = text.strip()
        if not text.startswith('{'):
            return None
        
        try:
            clean_text = text.split('#')[0].strip()
            data = json.loads(clean_text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    
    @staticmethod
    def normalize_location(location: Any) -> Optional[str]:
        """规范化位置信息"""
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
                    return (
                        obj.get("location") or
                        obj.get("city") or
                        obj.get("adcode") or
                        obj.get("citycode")
                    )
                except:
                    pass
            return text
        
        return str(location)
    
    @staticmethod
    def normalize_date(date_str: Any) -> Optional[str]:
        """规范化日期字符串"""
        if date_str is None:
            return None
        
        from datetime import datetime, timedelta
        
        text = str(date_str).strip()
        
        if text in ("今天", "今日", "today"):
            return datetime.now().strftime("%Y-%m-%d")
        
        if text in ("明天", "次日", "tomorrow"):
            return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        m = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", text)
        if m:
            return m.group(1).replace("/", "-")
        
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(text, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        """安全地转换为整数"""
        if value is None:
            return default
        
        if isinstance(value, int):
            return value
        
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        
        try:
            return int(value)
        except:
            return default
    
    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """安全地转换为浮点数"""
        if value is None:
            return default
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        
        try:
            return float(value)
        except:
            return default
