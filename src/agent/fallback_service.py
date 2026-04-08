import json
from typing import Any, Optional
from src.core.logger import logger
from src.core.errors import ErrorCode, ErrorHandler
from src.utils.helpers import extract_image_url


_FALLBACK_ERROR_CODES = {
    ErrorCode.TOOL_CALL_FAILED.value,
    ErrorCode.MCP_SESSION_ERROR.value,
    ErrorCode.MCP_CONNECTION_ERROR.value,
    ErrorCode.MCP_TIMEOUT_ERROR.value,
    ErrorCode.API_ERROR.value,
    ErrorCode.NASA_API_ERROR.value,
    ErrorCode.WEATHER_API_ERROR.value,
    ErrorCode.LLM_ERROR.value,
}

_FALLBACK_ERROR_KEYWORDS = [
    "工具调用错误",
    "调用工具失败",
    "调用工具超时",
    "无法连接到MCP服务器",
    "MCP会话未初始化",
    "HTTP错误",
    "Agent stopped due to iteration limit",
    "Agent stopped due to time limit",
]

_LOW_CONFIDENCE_PHRASES = [
    "当前模型服务暂时不可用",
    "无法回答你的问题",
]


class FallbackService:
    def __init__(self, skill_manager: Any):
        self._skill_manager = skill_manager

    def should_use_fallback(self, output: Any) -> bool:
        if not output:
            return True

        if isinstance(output, dict):
            if ErrorHandler.is_error_response(output):
                code = ErrorHandler.extract_error_code(output)
                if code and code in _FALLBACK_ERROR_CODES:
                    return True
            return False

        if isinstance(output, str):
            condensed = output.strip()
            if not condensed:
                return True
            for kw in _FALLBACK_ERROR_KEYWORDS:
                if kw in condensed:
                    return True
            if len(condensed) < 60:
                for kw in _LOW_CONFIDENCE_PHRASES:
                    if kw in condensed:
                        return True
            try:
                parsed = json.loads(condensed)
                if isinstance(parsed, dict) and ErrorHandler.is_error_response(parsed):
                    code = ErrorHandler.extract_error_code(parsed)
                    if code and code in _FALLBACK_ERROR_CODES:
                        return True
            except (json.JSONDecodeError, TypeError):
                pass

        return False

    def try_web_search_fallback(self, query: str) -> str:
        logger.warning("检测到工具调用可能失败，尝试使用联网搜索...")
        try:
            search_result = self._skill_manager.call_mcp_tool("web_search", query=query, max_results=5)
            logger.info("联网搜索降级方案执行成功")
            return search_result
        except Exception as e:
            logger.error(f"联网搜索降级也失败: {e}")
            error = ErrorHandler.handle(e, {"fallback_query": query})
            return error.to_json()

    def format_fallback_response(self, query: str, search_result: str) -> str:
        try:
            result_data = json.loads(search_result)

            if ErrorHandler.is_error_response(result_data):
                msg = result_data.get("message", str(result_data))
                return f"抱歉，我在处理您的查询「{query}」时遇到了问题：{msg}。请稍后再试或尝试其他问题。"

            answer = result_data.get("answer", "")
            results = result_data.get("results", [])

            if answer:
                response = f"根据搜索结果：\n\n{answer}\n\n"
            else:
                response = f"关于「{query}」，我找到以下信息：\n\n"

            for i, item in enumerate(results[:3], 1):
                title = item.get("title", "")
                url = item.get("url", "")
                content = item.get("content", "")
                if title:
                    response += f"{i}. {title}\n"
                    if content:
                        response += f"   {content[:150]}...\n"
                    response += f"   来源: {url}\n\n"

            return response

        except Exception as e:
            logger.error(f"格式化降级结果失败: {e}")
            return f"抱歉，处理搜索结果时出现问题。请稍后再试。"

    def extract_image_url(self, text: str) -> Optional[str]:
        return extract_image_url(text)
