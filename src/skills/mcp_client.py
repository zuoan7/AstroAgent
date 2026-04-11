from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Optional

import httpx
from src.core.logger import logger
from src.core.errors import AgentError, ErrorHandler, ErrorCode
from src.core.config import settings


MCP_RECONNECT_MAX_RETRIES = 3
MCP_RECONNECT_DELAY = 2.0


class _AsyncBridge:
    """
    Safe bridge to run async MCP operations from synchronous context.

    Creates a dedicated background event loop in a daemon thread, avoiding
    the dangerous pattern of calling asyncio.run() inside an already-running
    event loop (which causes RuntimeError / deadlock in FastAPI).

    Uses asyncio.run_coroutine_threadsafe() to submit coroutines from
    sync code to the background loop.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self) -> None:
        if self._loop is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=10)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run(self, coro: Any, timeout: float = 60.0) -> Any:
        self.start()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            future.cancel()
            raise e

    def shutdown(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None


class MCPClient:
    """
    MCP protocol client handling session lifecycle, SSE parsing, and tool invocation.

    Responsibilities:
    - Initialize and maintain MCP sessions (SSE handshake, protocol init, tool listing)
    - Reconnect with exponential backoff on session failures
    - Parse SSE responses from the MCP server
    - Call MCP tools with proper session headers and error handling
    """

    def __init__(self) -> None:
        self._session_id: Optional[str] = None
        self._initialized = False
        self._http_client: Optional[httpx.AsyncClient] = None
        self._reconnect_attempts = 0
        self._async_bridge = _AsyncBridge()

    def call_tool(self, tool_name: str, **kwargs) -> str:
        return self._async_bridge.run(
            self._async_call_tool(tool_name, **kwargs)
        )

    def call_tools_parallel(self, calls: list[dict]) -> list[str]:
        """
        Parallel MCP tool calls using asyncio.gather.

        Args:
            calls: list of {"tool_name": str, "kwargs": dict}

        Returns:
            Results list corresponding to calls order
        """
        async def _gather():
            coros = [
                self._async_call_tool(c["tool_name"], **c.get("kwargs", {}))
                for c in calls
            ]
            return await asyncio.gather(*coros, return_exceptions=True)

        results = self._async_bridge.run(_gather())
        final = []
        for r in results:
            if isinstance(r, Exception):
                error = ErrorHandler.handle(r, {"parallel_call": True})
                final.append(json.dumps(error.to_dict(), ensure_ascii=False))
            else:
                final.append(r)
        return final

    async def async_call_tool(self, tool_name: str, **kwargs) -> str:
        return await self._async_call_tool(tool_name, **kwargs)

    def shutdown(self) -> None:
        self._async_bridge.shutdown()
        if self._http_client and not self._http_client.is_closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._http_client.aclose())
                else:
                    loop.run_until_complete(self._http_client.aclose())
            except Exception:
                pass
        logger.info("✅ MCPClient已关闭")

    async def _init_session(self) -> bool:
        if self._initialized and self._is_session_valid():
            return True

        try:
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()

            client = httpx.AsyncClient(timeout=30.0)

            logger.info("正在建立SSE连接（异步）...")
            sse_response = await client.get(
                settings.MCP_SERVER_URL,
                headers={"Accept": "text/event-stream"}
            )

            session_id = sse_response.headers.get("mcp-session-id")
            if not session_id:
                raise Exception("无法获取session ID")

            logger.info(f"✅ 获取到session ID: {session_id}")

            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "AstroAgent-SkillRouter",
                        "version": "1.0.0"
                    }
                },
                "id": 1
            }

            response = await client.post(
                settings.MCP_SERVER_URL,
                json=init_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id
                }
            )

            if response.status_code != 200:
                raise Exception(f"初始化失败: {response.status_code}")

            init_result = _parse_sse_response(response.text)
            if init_result:
                logger.debug(f"初始化成功，服务器信息: {init_result.get('result', {}).get('serverInfo', {})}")

            notif_request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }

            await client.post(
                settings.MCP_SERVER_URL,
                json=notif_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id
                }
            )

            logger.info("获取工具列表...")
            list_request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2
            }

            response = await client.post(
                settings.MCP_SERVER_URL,
                json=list_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id
                }
            )

            tools_result = _parse_sse_response(response.text)
            if tools_result:
                tools_list = tools_result.get("result", {}).get("tools", [])
                logger.info(f"✅ 从服务器获取到 {len(tools_list)} 个工具")

            self._session_id = session_id
            self._initialized = True
            self._http_client = client
            self._reconnect_attempts = 0

            logger.info(f"✅ MCP会话初始化成功，会话ID: {session_id}")
            return True

        except Exception as e:
            logger.error(f"❌ MCP会话初始化失败: {e}")
            self._initialized = False
            self._http_client = None
            return False

    def _is_session_valid(self) -> bool:
        if not self._initialized or not self._session_id:
            return False
        if self._http_client is None or self._http_client.is_closed:
            return False
        return True

    async def _ensure_session(self) -> bool:
        if self._is_session_valid():
            return True

        logger.warning("MCP会话无效或未初始化，尝试建立连接...")
        return await self._reconnect()

    async def _reconnect(self) -> bool:
        for attempt in range(1, MCP_RECONNECT_MAX_RETRIES + 1):
            logger.info(f"MCP连接尝试 {attempt}/{MCP_RECONNECT_MAX_RETRIES}...")
            try:
                success = await self._init_session()
                if success:
                    return True
            except Exception as e:
                logger.error(f"❌ MCP连接第 {attempt} 次失败: {e}")

            self._initialized = False
            self._http_client = None
            if attempt < MCP_RECONNECT_MAX_RETRIES:
                await asyncio.sleep(MCP_RECONNECT_DELAY * attempt)

        logger.error(f"❌ MCP连接失败，已尝试 {MCP_RECONNECT_MAX_RETRIES} 次")
        self._reconnect_attempts += MCP_RECONNECT_MAX_RETRIES
        return False

    async def _async_call_tool(self, tool_name: str, **kwargs) -> str:
        session_ok = await self._ensure_session()
        if not session_ok:
            logger.error("❌ MCP会话不可用且重连失败")
            return json.dumps(AgentError(
                code=ErrorCode.MCP_SESSION_ERROR,
                message="MCP会话不可用且重连失败，请检查桥服务器是否运行",
                details={"tool_name": tool_name}
            ).to_dict(), ensure_ascii=False)

        try:
            processed_kwargs = {}
            for key, value in kwargs.items():
                if key in ['year', 'month', 'limit']:
                    try:
                        if isinstance(value, str) and value.isdigit():
                            processed_kwargs[key] = int(value)
                        else:
                            processed_kwargs[key] = value
                    except Exception:
                        processed_kwargs[key] = value
                else:
                    processed_kwargs[key] = value

            request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": processed_kwargs
                },
                "id": int(time.time() * 1000)
            }

            logger.debug(f"调用工具 {tool_name}，处理后的参数: {processed_kwargs}")

            response = await self._http_client.post(
                settings.MCP_SERVER_URL,
                json=request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": self._session_id
                },
                timeout=30.0
            )

            if response.status_code != 200:
                if response.status_code in (404, 410, 503):
                    self._initialized = False
                    logger.warning(f"MCP会话可能已失效(HTTP {response.status_code})，标记需要重连")
                return json.dumps(AgentError(
                    code=ErrorCode.MCP_SESSION_ERROR,
                    message=f"MCP服务器返回HTTP错误: {response.status_code}",
                    details={"tool_name": tool_name, "status_code": response.status_code}
                ).to_dict(), ensure_ascii=False)

            result = _parse_sse_response(response.text)
            if not result:
                logger.error(f"无法解析响应: {response.text[:200]}")
                return json.dumps(AgentError(
                    code=ErrorCode.MCP_SESSION_ERROR,
                    message="MCP响应解析失败",
                    details={"tool_name": tool_name}
                ).to_dict(), ensure_ascii=False)

            logger.debug(f"工具响应: {json.dumps(result, ensure_ascii=False)[:500]}")

            if "error" in result:
                error_msg = result["error"].get("message", "未知错误")
                error_code = result["error"].get("code", "")
                return json.dumps(AgentError(
                    code=ErrorCode.TOOL_CALL_FAILED,
                    message=f"工具调用错误 [{error_code}]: {error_msg}",
                    details={"tool_name": tool_name, "mcp_error_code": error_code}
                ).to_dict(), ensure_ascii=False)

            if "result" in result:
                res = result["result"]

                if isinstance(res, dict):
                    if "content" in res:
                        content = res["content"]
                        if isinstance(content, list) and len(content) > 0:
                            for item in content:
                                if item.get("type") == "text":
                                    text = item.get("text", "")
                                    try:
                                        parsed = json.loads(text)
                                        if isinstance(parsed, dict) and parsed.get("error"):
                                            return json.dumps(AgentError(
                                                code=ErrorCode.TOOL_CALL_FAILED,
                                                message=parsed.get("message", text),
                                                details={"tool_name": tool_name}
                                            ).to_dict(), ensure_ascii=False)
                                    except (json.JSONDecodeError, TypeError):
                                        pass
                                    return text

                    return json.dumps(res, ensure_ascii=False)

                if isinstance(res, str):
                    return res

                return str(res)

            logger.warning(f"未知响应格式: {result}")
            return str(result)

        except httpx.TimeoutException:
            logger.error(f"❌ MCP工具调用超时: {tool_name}")
            return json.dumps(AgentError(
                code=ErrorCode.MCP_TIMEOUT_ERROR,
                message=f"工具 '{tool_name}' 调用超时",
                details={"tool_name": tool_name}
            ).to_dict(), ensure_ascii=False)
        except httpx.ConnectError:
            logger.error(f"❌ 无法连接到MCP服务器: {settings.MCP_SERVER_URL}")
            self._initialized = False
            return json.dumps(AgentError(
                code=ErrorCode.MCP_CONNECTION_ERROR,
                message="无法连接到MCP服务器",
                details={"tool_name": tool_name, "server_url": settings.MCP_SERVER_URL}
            ).to_dict(), ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ 调用工具 {tool_name} 失败: {e}")
            error = ErrorHandler.handle(e, {"tool_name": tool_name})
            return json.dumps(error.to_dict(), ensure_ascii=False)


def _parse_sse_response(response_text: str) -> Optional[dict]:
    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith("data: "):
                json_str = line[6:]
                return json.loads(json_str)
        return None
    except Exception as e:
        logger.error(f"解析 SSE 响应失败: {e}")
        return None
