import json
from typing import Any, Dict, Optional


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
    def extract_param(data: Dict[str, Any], keys: list, default: Any = None) -> Any:
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
