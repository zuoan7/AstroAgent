"""Deprecated MCP client import path.

Use ``src.transport.mcp.client`` for new code. This compatibility module keeps
existing imports working during the Skill/Tool boundary refactor.
"""

from src.transport.mcp.client import MCPClient, _AsyncBridge
from src.transport.mcp.sse import parse_sse_response as _parse_sse_response

__all__ = ["MCPClient", "_AsyncBridge", "_parse_sse_response"]
