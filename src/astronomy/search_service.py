# -*- coding: utf-8 -*-
"""
搜索服务模块 - Tavily联网搜索
"""

import os
import logging

import requests

from src.core.config import settings

logger = logging.getLogger(__name__)


class SearchService:
    """
    联网搜索服务
    
    使用Tavily API提供网络搜索功能。
    """
    
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY") or getattr(settings, "TAVILY_API_KEY", None)
    
    def search(self, query: str, max_results: int = 5) -> dict:
        """
        执行网络搜索
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数（默认5）
            
        Returns:
            搜索结果字典
        """
        try:
            # 验证API密钥
            if not self.api_key:
                return {
                    "error": "TAVILY_API_KEY 未配置",
                    "suggestion": "请在.env文件中配置 TAVILY_API_KEY"
                }
            
            # 构建请求
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False
            }
            
            # 发送请求
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            # 处理响应
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
            
            return {
                "query": query,
                "answer": answer,
                "results": results,
                "total": len(results)
            }
            
        except requests.exceptions.Timeout:
            logger.error(f"搜索超时: {query}")
            return {"error": "搜索请求超时，请稍后重试"}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"搜索请求失败: {e}")
            return {"error": f"搜索请求失败: {str(e)}"}
            
        except Exception as e:
            logger.error(f"搜索异常: {e}")
            return {"error": f"搜索异常: {str(e)}"}


__all__ = ['SearchService']
