import json
from typing import Any, Optional

from src.agent.policies.fallback_policy import FallbackDecision, FallbackPolicy
from src.agent.tool_observation import normalize_observation
from src.core.errors import ErrorCode, ErrorHandler
from src.core.logger import logger

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
    def __init__(self, capability_kit: Any = None):
        self._capability_kit = capability_kit
        self._policy = FallbackPolicy()

    def should_use_fallback(self, output: Any) -> bool:
        if not output:
            return True

        normalized = normalize_observation(output)
        if normalized.is_error:
            if normalized.error_code and normalized.error_code in _FALLBACK_ERROR_CODES:
                return True

        if isinstance(output, dict):
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
            result = self._capability_kit.call_tool(
                "web_search",
                query=query,
                max_results=5,
            )
            normalized = normalize_observation(result)
            search_result = (
                normalized.text
                if normalized.text
                else json.dumps(normalized.data, ensure_ascii=False)
            )
            logger.info("联网搜索降级方案执行成功")
            return search_result
        except Exception as e:
            logger.error(f"联网搜索降级也失败: {e}")
            error = ErrorHandler.handle(e, {"fallback_query": query})
            return error.to_json()

    def classify_web_fallback(self, reason: str) -> FallbackDecision:
        return FallbackDecision(
            strategy="web_fallback",
            reason=reason,
            metadata={"policy_version": self._policy.version},
        )

    def format_fallback_response(self, query: str, search_result: str) -> str:
        try:
            normalized = normalize_observation(search_result)
            result_data = normalized.data
            if isinstance(result_data, str):
                try:
                    result_data = json.loads(result_data)
                except (json.JSONDecodeError, TypeError):
                    result_data = {"answer": result_data}

            if normalized.is_error or ErrorHandler.is_error_response(result_data):
                msg = normalized.error_message or result_data.get(
                    "message",
                    str(result_data),
                )
                return f"抱歉，我在处理您的查询「{query}」时遇到了问题：{msg}。请稍后再试或尝试其他问题。"

            if not isinstance(result_data, dict):
                result_data = {"answer": str(result_data)}

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
        from src.utils.param_parser import ParamParser

        return ParamParser.extract_image_url(text)
