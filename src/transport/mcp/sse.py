"""SSE parsing helpers for MCP Streamable HTTP responses."""

from __future__ import annotations

import json
from typing import Optional

from src.core.logger import logger


def parse_sse_response(response_text: str) -> Optional[dict]:
    """Parse a JSON payload from an MCP Streamable HTTP SSE response."""
    try:
        lines = response_text.strip().split("\n")
        for line in lines:
            if line.startswith("data: "):
                json_str = line[6:]
                return json.loads(json_str)
        return None
    except Exception as e:
        logger.error(f"解析 SSE 响应失败: {e}")
        return None

