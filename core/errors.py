"""
统一错误处理机制
提供标准化的错误码、错误信息和错误处理
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


class ErrorCode(Enum):
    """错误码枚举"""
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    MCP_SESSION_ERROR = "MCP_SESSION_ERROR"
    MCP_CONNECTION_ERROR = "MCP_CONNECTION_ERROR"
    MCP_TIMEOUT_ERROR = "MCP_TIMEOUT_ERROR"
    LLM_ERROR = "LLM_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PARAM_PARSE_ERROR = "PARAM_PARSE_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    RAG_ERROR = "RAG_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class AgentError(Exception):
    """
    统一的Agent错误类
    
    Attributes:
        code: 错误码
        message: 错误信息
        details: 详细信息（可选）
        original_error: 原始异常（可选）
    """
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = field(default_factory=dict)
    original_error: Optional[Exception] = None
    
    def __post_init__(self):
        super().__init__(self.message)
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "error": True,
            "code": self.code.value,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"
    
    def __repr__(self) -> str:
        return f"AgentError(code={self.code}, message={self.message!r})"


class ErrorHandler:
    """统一错误处理器"""
    
    @staticmethod
    def handle(error: Exception, context: Optional[Dict[str, Any]] = None) -> AgentError:
        """
        将异常转换为AgentError
        
        Args:
            error: 原始异常
            context: 错误上下文信息
        
        Returns:
            AgentError实例
        """
        context = context or {}
        
        if isinstance(error, AgentError):
            return error
        
        error_type = type(error).__name__
        error_message = str(error)
        
        if 'MCP' in error_type or 'mcp' in error_message.lower():
            if 'timeout' in error_message.lower():
                return AgentError(
                    code=ErrorCode.MCP_TIMEOUT_ERROR,
                    message=f"MCP调用超时: {error_message}",
                    details=context,
                    original_error=error
                )
            elif 'connection' in error_message.lower() or 'connect' in error_message.lower():
                return AgentError(
                    code=ErrorCode.MCP_CONNECTION_ERROR,
                    message=f"MCP连接失败: {error_message}",
                    details=context,
                    original_error=error
                )
            else:
                return AgentError(
                    code=ErrorCode.MCP_SESSION_ERROR,
                    message=f"MCP会话错误: {error_message}",
                    details=context,
                    original_error=error
                )
        
        if 'validation' in error_message.lower() or 'invalid' in error_message.lower():
            return AgentError(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"参数验证失败: {error_message}",
                details=context,
                original_error=error
            )
        
        if 'timeout' in error_message.lower():
            return AgentError(
                code=ErrorCode.TOOL_CALL_FAILED,
                message=f"工具调用超时: {error_message}",
                details=context,
                original_error=error
            )
        
        return AgentError(
            code=ErrorCode.UNKNOWN_ERROR,
            message=f"未知错误: {error_message}",
            details=context,
            original_error=error
        )
    
    @staticmethod
    def create_tool_error(tool_name: str, error_message: str, details: Optional[Dict] = None) -> AgentError:
        """
        创建工具调用错误
        
        Args:
            tool_name: 工具名称
            error_message: 错误信息
            details: 详细信息
        
        Returns:
            AgentError实例
        """
        return AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message=f"工具 '{tool_name}' 调用失败: {error_message}",
            details={"tool_name": tool_name, **(details or {})}
        )
    
    @staticmethod
    def create_param_error(param_name: str, error_message: str, details: Optional[Dict] = None) -> AgentError:
        """
        创建参数解析错误
        
        Args:
            param_name: 参数名称
            error_message: 错误信息
            details: 详细信息
        
        Returns:
            AgentError实例
        """
        return AgentError(
            code=ErrorCode.PARAM_PARSE_ERROR,
            message=f"参数 '{param_name}' 解析失败: {error_message}",
            details={"param_name": param_name, **(details or {})}
        )


def safe_tool_call(func):
    """
    工具调用装饰器，自动捕获异常并转换为AgentError
    
    Usage:
        @safe_tool_call
        def my_tool(param: str) -> str:
            # 工具逻辑
            pass
    """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, AgentError):
                return result.to_json()
            return result
        except AgentError as e:
            return e.to_json()
        except Exception as e:
            error = ErrorHandler.handle(e, {"function": func.__name__})
            return error.to_json()
    return wrapper
