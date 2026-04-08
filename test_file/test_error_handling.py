"""
统一错误处理机制测试
覆盖 core/errors.py 及各模块对 AgentError 的集成使用
"""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.errors import (
    AgentError, ErrorCode, ErrorHandler,
    safe_tool_call, register_exception_mapping, _EXCEPTION_MAP
)


# ============================================================
# 1. ErrorCode 枚举测试
# ============================================================

class TestErrorCode:
    def test_all_error_codes_have_value(self):
        for code in ErrorCode:
            assert isinstance(code.value, str)
            assert len(code.value) > 0

    def test_new_error_codes_exist(self):
        assert hasattr(ErrorCode, "VISION_ERROR")
        assert hasattr(ErrorCode, "SPEECH_ERROR")
        assert hasattr(ErrorCode, "API_ERROR")
        assert hasattr(ErrorCode, "NASA_API_ERROR")
        assert hasattr(ErrorCode, "WEATHER_API_ERROR")
        assert hasattr(ErrorCode, "FILE_NOT_FOUND")

    def test_error_code_values_are_unique(self):
        values = [code.value for code in ErrorCode]
        assert len(values) == len(set(values))


# ============================================================
# 2. AgentError 测试
# ============================================================

class TestAgentError:
    def test_basic_creation(self):
        err = AgentError(code=ErrorCode.UNKNOWN_ERROR, message="test error")
        assert err.code == ErrorCode.UNKNOWN_ERROR
        assert err.message == "test error"
        assert err.details == {}
        assert err.original_error is None

    def test_creation_with_details(self):
        err = AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message="tool failed",
            details={"tool_name": "test_tool"},
            original_error=ValueError("bad value")
        )
        assert err.details["tool_name"] == "test_tool"
        assert isinstance(err.original_error, ValueError)

    def test_to_dict(self):
        err = AgentError(
            code=ErrorCode.MCP_TIMEOUT_ERROR,
            message="timeout",
            details={"tool_name": "get_weather"}
        )
        d = err.to_dict()
        assert d["error"] is True
        assert d["code"] == "MCP_TIMEOUT_ERROR"
        assert d["message"] == "timeout"
        assert d["details"]["tool_name"] == "get_weather"

    def test_to_dict_no_details(self):
        err = AgentError(code=ErrorCode.UNKNOWN_ERROR, message="no details")
        d = err.to_dict()
        assert "details" not in d

    def test_to_json(self):
        err = AgentError(code=ErrorCode.VALIDATION_ERROR, message="bad param")
        j = err.to_json()
        parsed = json.loads(j)
        assert parsed["error"] is True
        assert parsed["code"] == "VALIDATION_ERROR"

    def test_str_representation(self):
        err = AgentError(code=ErrorCode.MCP_SESSION_ERROR, message="session lost")
        assert "[MCP_SESSION_ERROR]" in str(err)
        assert "session lost" in str(err)

    def test_is_exception(self):
        err = AgentError(code=ErrorCode.UNKNOWN_ERROR, message="test")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(AgentError) as exc_info:
            raise AgentError(code=ErrorCode.TOOL_CALL_FAILED, message="raised")
        assert exc_info.value.code == ErrorCode.TOOL_CALL_FAILED


# ============================================================
# 3. ErrorHandler 测试
# ============================================================

class TestErrorHandler:
    def test_handle_agent_error_passthrough(self):
        original = AgentError(code=ErrorCode.MCP_TIMEOUT_ERROR, message="original")
        result = ErrorHandler.handle(original)
        assert result is original

    def test_handle_value_error(self):
        err = ErrorHandler.handle(ValueError("bad value"))
        assert err.code == ErrorCode.VALIDATION_ERROR
        assert "bad value" in err.message
        assert isinstance(err.original_error, ValueError)

    def test_handle_file_not_found(self):
        err = ErrorHandler.handle(FileNotFoundError("missing file"))
        assert err.code == ErrorCode.FILE_NOT_FOUND

    def test_handle_json_decode_error(self):
        err = ErrorHandler.handle(json.JSONDecodeError("bad json", "", 0))
        assert err.code == ErrorCode.PARAM_PARSE_ERROR

    def test_handle_unknown_exception(self):
        err = ErrorHandler.handle(RuntimeError("something weird"))
        assert err.code == ErrorCode.UNKNOWN_ERROR
        assert err.original_error is not None

    def test_handle_with_context(self):
        err = ErrorHandler.handle(ValueError("test"), context={"key": "value"})
        assert err.details["key"] == "value"

    def test_create_tool_error(self):
        err = ErrorHandler.create_tool_error("get_weather", "API key missing")
        assert err.code == ErrorCode.TOOL_CALL_FAILED
        assert "get_weather" in err.message
        assert err.details["tool_name"] == "get_weather"

    def test_create_param_error(self):
        err = ErrorHandler.create_param_error("city", "invalid format")
        assert err.code == ErrorCode.PARAM_PARSE_ERROR
        assert "city" in err.message
        assert err.details["param_name"] == "city"

    def test_create_api_error(self):
        err = ErrorHandler.create_api_error("NASA", "timeout")
        assert err.code == ErrorCode.API_ERROR
        assert "NASA" in err.message
        assert err.details["api_name"] == "NASA"

    def test_is_error_response_dict(self):
        assert ErrorHandler.is_error_response({"error": True, "code": "X", "message": "m"}) is True
        assert ErrorHandler.is_error_response({"error": "something went wrong"}) is True
        assert ErrorHandler.is_error_response({"status": "ok"}) is False

    def test_is_error_response_agent_error(self):
        err = AgentError(code=ErrorCode.UNKNOWN_ERROR, message="test")
        assert ErrorHandler.is_error_response(err) is True

    def test_is_error_response_other(self):
        assert ErrorHandler.is_error_response("just a string") is False
        assert ErrorHandler.is_error_response(None) is False

    def test_extract_error_code_from_dict(self):
        assert ErrorHandler.extract_error_code({"code": "MCP_TIMEOUT_ERROR"}) == "MCP_TIMEOUT_ERROR"
        assert ErrorHandler.extract_error_code({"no_code": True}) is None

    def test_extract_error_code_from_agent_error(self):
        err = AgentError(code=ErrorCode.MCP_SESSION_ERROR, message="test")
        assert ErrorHandler.extract_error_code(err) == "MCP_SESSION_ERROR"


# ============================================================
# 4. safe_tool_call 装饰器测试
# ============================================================

class TestSafeToolCall:
    def test_normal_return(self):
        @safe_tool_call
        def my_tool(x):
            return {"result": x}

        result = my_tool(42)
        assert result == {"result": 42}

    def test_exception_returns_error_dict(self):
        @safe_tool_call
        def failing_tool():
            raise ValueError("bad input")

        result = failing_tool()
        assert isinstance(result, dict)
        assert result["error"] is True
        assert "bad input" in result["message"]

    def test_agent_error_exception_returns_dict(self):
        @safe_tool_call
        def agent_error_tool():
            raise AgentError(code=ErrorCode.MCP_TIMEOUT_ERROR, message="timeout")

        result = agent_error_tool()
        assert isinstance(result, dict)
        assert result["code"] == "MCP_TIMEOUT_ERROR"

    def test_agent_error_return_value(self):
        @safe_tool_call
        def returns_agent_error():
            return AgentError(code=ErrorCode.TOOL_CALL_FAILED, message="soft error")

        result = returns_agent_error()
        assert isinstance(result, dict)
        assert result["error"] is True

    def test_custom_error_code(self):
        @safe_tool_call(error_code=ErrorCode.VISION_ERROR)
        def vision_tool():
            raise RuntimeError("vision failed")

        result = vision_tool()
        assert result["code"] == "VISION_ERROR"

    def test_preserves_function_name(self):
        @safe_tool_call
        def my_named_tool():
            return "ok"

        assert my_named_tool.__name__ == "my_named_tool"

    def test_known_exception_maps_correctly(self):
        @safe_tool_call
        def json_tool():
            raise json.JSONDecodeError("bad", "", 0)

        result = json_tool()
        assert result["code"] == "PARAM_PARSE_ERROR"

    def test_file_not_found_maps_correctly(self):
        @safe_tool_call
        def file_tool():
            raise FileNotFoundError("no file")

        result = file_tool()
        assert result["code"] == "FILE_NOT_FOUND"


# ============================================================
# 5. 异常映射注册测试
# ============================================================

class TestExceptionMapping:
    def test_register_custom_mapping(self):
        class CustomError(Exception):
            pass

        register_exception_mapping(CustomError, ErrorCode.RAG_ERROR)
        err = ErrorHandler.handle(CustomError("custom"))
        assert err.code == ErrorCode.RAG_ERROR

    def test_httpx_timeout_mapped(self):
        try:
            import httpx
            err = ErrorHandler.handle(httpx.TimeoutException("timeout"))
            assert err.code == ErrorCode.MCP_TIMEOUT_ERROR
        except ImportError:
            pytest.skip("httpx not installed")

    def test_httpx_connect_error_mapped(self):
        try:
            import httpx
            err = ErrorHandler.handle(httpx.ConnectError("connection failed"))
            assert err.code == ErrorCode.MCP_CONNECTION_ERROR
        except ImportError:
            pytest.skip("httpx not installed")


# ============================================================
# 6. FallbackService 集成测试
# ============================================================

class TestFallbackServiceIntegration:
    def test_should_use_fallback_with_structured_error(self):
        from agent.fallback_service import FallbackService

        fs = FallbackService(skill_manager=None)

        error_dict = AgentError(
            code=ErrorCode.MCP_TIMEOUT_ERROR,
            message="timeout"
        ).to_dict()
        assert fs.should_use_fallback(error_dict) is True

    def test_should_use_fallback_with_non_error_dict(self):
        from agent.fallback_service import FallbackService

        fs = FallbackService(skill_manager=None)
        assert fs.should_use_fallback({"status": "ok", "data": "result"}) is False

    def test_should_use_fallback_with_error_string(self):
        from agent.fallback_service import FallbackService

        fs = FallbackService(skill_manager=None)
        assert fs.should_use_fallback("工具调用错误: something") is True
        assert fs.should_use_fallback("正常回答内容") is False

    def test_should_use_fallback_with_empty(self):
        from agent.fallback_service import FallbackService

        fs = FallbackService(skill_manager=None)
        assert fs.should_use_fallback("") is True
        assert fs.should_use_fallback(None) is True

    def test_should_use_fallback_with_json_error_string(self):
        from agent.fallback_service import FallbackService

        fs = FallbackService(skill_manager=None)
        error_json = json.dumps(AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message="tool failed"
        ).to_dict())
        assert fs.should_use_fallback(error_json) is True

    def test_format_fallback_with_structured_error(self):
        from agent.fallback_service import FallbackService

        fs = FallbackService(skill_manager=None)
        error_json = json.dumps(AgentError(
            code=ErrorCode.MCP_TIMEOUT_ERROR,
            message="timeout"
        ).to_dict())
        result = fs.format_fallback_response("test query", error_json)
        assert "遇到了问题" in result


# ============================================================
# 7. VisionService / SpeechService 错误类型测试
# ============================================================

class TestServiceErrorTypes:
    def test_vision_file_not_found_raises_agent_error(self):
        from agent.vision_service import VisionService
        from unittest.mock import patch

        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test-key'}):
            vs = VisionService(api_key="test-key")
            with pytest.raises(Exception) as exc_info:
                vs.describe_image("/nonexistent/path.png", "describe")
            err = exc_info.value
            assert hasattr(err, 'code')
            assert hasattr(err, 'message')
            assert err.code.value == "FILE_NOT_FOUND"
            assert "图片文件不存在" in err.message

    def test_speech_file_not_found_raises_agent_error(self):
        from agent.speech_service import SpeechService
        from unittest.mock import patch

        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test-key'}):
            ss = SpeechService(api_key="test-key")
            with pytest.raises(Exception) as exc_info:
                ss.transcribe_audio("/nonexistent/audio.wav")
            err = exc_info.value
            assert hasattr(err, 'code')
            assert hasattr(err, 'message')
            assert err.code.value == "FILE_NOT_FOUND"
            assert "音频文件不存在" in err.message

    def test_vision_build_query_handles_error_gracefully(self):
        from agent.vision_service import VisionService
        from unittest.mock import patch

        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test-key'}):
            vs = VisionService(api_key="test-key")
            result = vs.build_vision_query("test query", "/nonexistent/path.png")
            assert "test query" in result
            assert "图片理解失败" in result

    def test_speech_build_query_handles_error_gracefully(self):
        from agent.speech_service import SpeechService
        from unittest.mock import patch

        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test-key'}):
            ss = SpeechService(api_key="test-key")
            result = ss.build_speech_query("test query", "/nonexistent/audio.wav")
            assert "test query" in result
            assert "语音识别失败" in result


# ============================================================
# 8. API 全局异常处理器测试
# ============================================================

class TestAPIErrorHandlers:
    def test_api_imports_succeed(self):
        from core.errors import AgentError, ErrorCode, ErrorHandler
        assert AgentError is not None
        assert ErrorCode is not None
        assert ErrorHandler is not None


# ============================================================
# 9. StreamingService 错误检测集成测试
# ============================================================

class TestStreamingServiceErrorDetection:
    def test_error_response_detection_in_observation(self):
        error_dict = AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message="tool failed"
        ).to_dict()
        assert ErrorHandler.is_error_response(error_dict) is True

        normal_dict = {"result": "success", "data": "some data"}
        assert ErrorHandler.is_error_response(normal_dict) is False

    def test_string_observation_not_false_positive(self):
        normal_string = "This is a normal observation result"
        assert ErrorHandler.is_error_response(normal_string) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
