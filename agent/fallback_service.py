import json
import re
from typing import Any, Generator, Optional
from logger import logger


class FallbackService:
    def __init__(self, skill_manager: Any):
        self._skill_manager = skill_manager
        self._error_patterns = [
            "工具调用错误",
            "调用工具失败",
            "调用工具超时",
            "无法连接到MCP服务器",
            "MCP会话未初始化",
            "HTTP错误",
            "Agent stopped due to iteration limit",
            "Agent stopped due to time limit",
        ]
        self._low_confidence_phrases = [
            "当前模型服务暂时不可用",
            "无法回答你的问题",
        ]

    def should_use_fallback(self, output: str) -> bool:
        if not output:
            return True
        condensed = output.strip()
        if not condensed:
            return True
        for kw in self._error_patterns:
            if kw in condensed:
                return True
        if len(condensed) < 60:
            for kw in self._low_confidence_phrases:
                if kw in condensed:
                    return True
        return False

    def try_web_search_fallback(self, query: str) -> str:
        logger.warning("检测到工具调用可能失败，尝试使用联网搜索...")
        try:
            search_result = self._skill_manager.call_mcp_tool("web_search", query=query, max_results=5)
            logger.info("联网搜索降级方案执行成功")
            return search_result
        except Exception as e:
            logger.error(f"联网搜索降级也失败: {e}")
            return json.dumps({"error": f"降级搜索失败: {str(e)}"}, ensure_ascii=False)

    def format_fallback_response(self, query: str, search_result: str) -> str:
        try:
            result_data = json.loads(search_result)

            if "error" in result_data:
                return f"抱歉，我在处理您的查询「{query}」时遇到了问题：{result_data['error']}。请稍后再试或尝试其他问题。"

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
        m = re.search(r"(https?://\S+\.(?:png|jpg|jpeg|webp))", text, re.IGNORECASE)
        return m.group(1) if m else None
