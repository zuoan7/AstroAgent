"""
统一错误处理机制
提供标准化的错误码、错误信息和错误处理
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Type
import json
import functools


class ErrorCode(Enum):
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    MCP_SESSION_ERROR = "MCP_SESSION_ERROR"
    MCP_CONNECTION_ERROR = "MCP_CONNECTION_ERROR"
    MCP_TIMEOUT_ERROR = "MCP_TIMEOUT_ERROR"
    LLM_ERROR = "LLM_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PARAM_PARSE_ERROR = "PARAM_PARSE_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    RAG_ERROR = "RAG_ERROR"
    VISION_ERROR = "VISION_ERROR"
    SPEECH_ERROR = "SPEECH_ERROR"
    API_ERROR = "API_ERROR"
    NASA_API_ERROR = "NASA_API_ERROR"
    WEATHER_API_ERROR = "WEATHER_API_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    SECURITY_ERROR = "SECURITY_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    FILE_TYPE_NOT_ALLOWED = "FILE_TYPE_NOT_ALLOWED"
    PATH_TRAVERSAL_ERROR = "PATH_TRAVERSAL_ERROR"
    MEMORY_NOT_FOUND = "MEMORY_NOT_FOUND"
    MEMORY_ACCESS_DENIED = "MEMORY_ACCESS_DENIED"
    MEMORY_CONFLICT = "MEMORY_CONFLICT"
    MEMORY_VALIDATION_ERROR = "MEMORY_VALIDATION_ERROR"
    MEMORY_CANDIDATE_ERROR = "MEMORY_CANDIDATE_ERROR"
    MEMORY_CONFIRMATION_ERROR = "MEMORY_CONFIRMATION_ERROR"
    MEMORY_BACKUP_ERROR = "MEMORY_BACKUP_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


_EXCEPTION_MAP: Dict[Type[Exception], ErrorCode] = {}


def register_exception_mapping(exc_type: Type[Exception], code: ErrorCode):
    _EXCEPTION_MAP[exc_type] = code


def _init_default_exception_map():
    try:
        import httpx
        register_exception_mapping(httpx.TimeoutException, ErrorCode.MCP_TIMEOUT_ERROR)
        register_exception_mapping(httpx.ConnectError, ErrorCode.MCP_CONNECTION_ERROR)
    except ImportError:
        pass

    register_exception_mapping(json.JSONDecodeError, ErrorCode.PARAM_PARSE_ERROR)
    register_exception_mapping(FileNotFoundError, ErrorCode.FILE_NOT_FOUND)
    register_exception_mapping(ValueError, ErrorCode.VALIDATION_ERROR)
    register_exception_mapping(TypeError, ErrorCode.VALIDATION_ERROR)

    try:
        import sqlite3
        register_exception_mapping(sqlite3.Error, ErrorCode.MEMORY_ERROR)
    except ImportError:
        pass

    try:
        import requests
        register_exception_mapping(requests.Timeout, ErrorCode.API_ERROR)
        register_exception_mapping(requests.ConnectionError, ErrorCode.API_ERROR)
    except ImportError:
        pass


_init_default_exception_map()


@dataclass
class AgentError(Exception):
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = field(default_factory=dict)
    original_error: Optional[Exception] = None

    def __post_init__(self):
        super().__init__(self.message)
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "error": True,
            "code": self.code.value,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def __repr__(self) -> str:
        return f"AgentError(code={self.code}, message={self.message!r})"


class ErrorHandler:
    """统一错误处理器"""

    @staticmethod
    def handle(error: Exception, context: Optional[Dict[str, Any]] = None) -> AgentError:
        context = context or {}

        if isinstance(error, AgentError):
            return error

        for exc_type, code in _EXCEPTION_MAP.items():
            if isinstance(error, exc_type):
                return AgentError(
                    code=code,
                    message=str(error),
                    details=context,
                    original_error=error
                )

        return AgentError(
            code=ErrorCode.UNKNOWN_ERROR,
            message=str(error),
            details=context,
            original_error=error
        )

    @staticmethod
    def create_tool_error(tool_name: str, error_message: str, details: Optional[Dict] = None) -> AgentError:
        return AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message=f"工具 '{tool_name}' 调用失败: {error_message}",
            details={"tool_name": tool_name, **(details or {})}
        )

    @staticmethod
    def create_param_error(param_name: str, error_message: str, details: Optional[Dict] = None) -> AgentError:
        return AgentError(
            code=ErrorCode.PARAM_PARSE_ERROR,
            message=f"参数 '{param_name}' 解析失败: {error_message}",
            details={"param_name": param_name, **(details or {})}
        )

    @staticmethod
    def create_api_error(api_name: str, error_message: str, details: Optional[Dict] = None) -> AgentError:
        return AgentError(
            code=ErrorCode.API_ERROR,
            message=f"API '{api_name}' 调用失败: {error_message}",
            details={"api_name": api_name, **(details or {})}
        )

    @staticmethod
    def is_error_response(data: Any) -> bool:
        if isinstance(data, dict):
            return data.get("error") is True or "error" in data
        if isinstance(data, AgentError):
            return True
        return False

    @staticmethod
    def extract_error_code(data: Any) -> Optional[str]:
        if isinstance(data, AgentError):
            return data.code.value
        if isinstance(data, dict) and "code" in data:
            return data["code"]
        return None


def safe_tool_call(func=None, *, error_code: ErrorCode = ErrorCode.TOOL_CALL_FAILED):
    """
    工具调用装饰器，自动捕获异常并转换为 AgentError dict

    Usage:
        @safe_tool_call
        def my_tool(param: str) -> dict:
            pass

        @safe_tool_call(error_code=ErrorCode.VISION_ERROR)
        def describe_image(path: str) -> dict:
            pass
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, AgentError):
                    return result.to_dict()
                return result
            except AgentError as e:
                return e.to_dict()
            except Exception as e:
                error = ErrorHandler.handle(e, {"function": fn.__name__})
                if error.code == ErrorCode.UNKNOWN_ERROR:
                    error = AgentError(
                        code=error_code,
                        message=str(e),
                        details={"function": fn.__name__},
                        original_error=e
                    )
                return error.to_dict()
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
