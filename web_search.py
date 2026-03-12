"""
联网搜索工具模块 - 作为Agent的降级机制
使用Tavily Search API进行网络搜索
"""
import os
import json
import requests
from typing import Optional, List, Dict, Any
from logger import logger


class WebSearchTool:
    """联网搜索工具"""

    def __init__(self):
        self.base_url = "https://api.tavily.com/search"

    def _get_api_key(self) -> Optional[str]:
        """获取API密钥 - 动态获取，支持运行时设置"""
        # 首先尝试从环境变量获取
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            return api_key
        # 然后尝试从settings获取
        try:
            from config import settings
            return getattr(settings, "TAVILY_API_KEY", None)
        except:
            return None

    def search(self, query: str, max_results: int = 5) -> str:
        """
        执行网络搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            格式化的搜索结果
        """
        api_key = self._get_api_key()

        if not api_key:
            return json.dumps({
                "error": "TAVILY_API_KEY 未配置",
                "suggestion": "请在.env文件中配置 TAVILY_API_KEY"
            }, ensure_ascii=False)

        try:
            payload = {
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False
            }

            response = requests.post(self.base_url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            results = []
            if "results" in data:
                for item in data["results"]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", "")[:500]
                    })

            answer = data.get("answer", "")

            formatted = {
                "query": query,
                "answer": answer,
                "results": results,
                "total": len(results)
            }

            logger.info(f"联网搜索成功: {query}, 获取到 {len(results)} 条结果")
            return json.dumps(formatted, ensure_ascii=False)

        except requests.exceptions.Timeout:
            logger.error("搜索请求超时")
            return json.dumps({"error": "搜索请求超时，请稍后重试"}, ensure_ascii=False)
        except requests.exceptions.RequestException as e:
            logger.error(f"搜索请求失败: {e}")
            return json.dumps({"error": f"搜索请求失败: {str(e)}"}, ensure_ascii=False)
        except Exception as e:
            logger.error(f"搜索异常: {e}")
            return json.dumps({"error": f"搜索异常: {str(e)}"}, ensure_ascii=False)


web_search_tool = WebSearchTool()


def web_search(query: str, max_results: int = 5) -> str:
    """
    联网搜索函数 - 供Agent调用
    """
    return web_search_tool.search(query, max_results)
