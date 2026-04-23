from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
from typing import Any, Dict, Optional

import httpx

from src.core.logger import logger
from src.core.errors import ErrorHandler, ErrorCode
from src.core.config import settings
from src.core.mcp_protocol import (
    error_envelope,
    parse_tool_response,
    serialize_envelope,
)

MCP_RECONNECT_MAX_RETRIES = 3
MCP_RECONNECT_DELAY = 2.0


class _AsyncBridge:
    """
    Safe bridge to run async MCP operations from synchronous context.

    Creates a dedicated background event loop in a daemon thread, avoiding the
    dangerous pattern of calling asyncio.run() inside an already-running event loop
    (which causes RuntimeError / deadlock in FastAPI).
    Uses asyncio.run_coroutine_threadsafe() to submit coroutines from sync code
    to the background loop.
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
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"异步操作超时({timeout}s)，可能原因：MCP服务器无响应、死锁或LLM调用阻塞"
            ) from None
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
        self._init_lock: Optional[asyncio.Lock] = None
        self._initializing = False
        self._metrics_lock = threading.Lock()
        self._runtime_metrics: Dict[str, float] = {
            "mcp_session_init_ms": 0.0,
            "tool_exec_ms": 0.0,
            "tool_call_count": 0.0,
        }

    def _get_init_lock(self) -> asyncio.Lock:
        if self._init_lock is None:
            try:
                asyncio.get_running_loop()
                self._init_lock = asyncio.Lock()
            except RuntimeError:
                self._init_lock = asyncio.Lock()
        return self._init_lock

    def call_tool(self, tool_name: str, **kwargs) -> str:
        return self._async_bridge.run(self._async_call_tool(tool_name, **kwargs))

    def call_tools_parallel(self, calls: list[dict]) -> list[str]:
        """
        Batch MCP tool calls using isolated sessions for each parallel request.

        Args:
            calls: list of {"tool_name": str, "kwargs": dict}
        Returns:
            Results list corresponding to calls order
        """

        async def _gather():
            tasks = [
                self._dispatch_parallel_tool_call(
                    call["tool_name"],
                    **call.get("kwargs", {}),
                )
                for call in calls
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        batch_timeout = max(90.0, 35.0 * len(calls))

        try:
            logger.info(
                f"MCP批量调用开始（独立会话并行），calls={len(calls)}，timeout={batch_timeout}s"
            )
            results = self._async_bridge.run(_gather(), timeout=batch_timeout)
        except TimeoutError:
            logger.warning("MCP批量调用超时，回退为同步串行调用")
            # 废弃主会话，避免沿用潜在坏状态
            self._initialized = False
            self._session_id = None
            return [
                self.call_tool(c["tool_name"], **c.get("kwargs", {}))
                for c in calls
            ]

        final = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                tool_name = calls[i].get("tool_name", "parallel_call")
                error = ErrorHandler.handle(
                    r,
                    {"parallel_call": True, "tool_name": tool_name},
                )
                final.append(
                    serialize_envelope(
                        error_envelope(
                            tool_name=tool_name,
                            code=error.code.value,
                            message=error.message,
                            details=error.details,
                        )
                    )
                )
            else:
                final.append(r)
        return final

    async def _dispatch_parallel_tool_call(self, tool_name: str, **kwargs) -> str:
        """
        Dedicated entrypoint for batch tool dispatch.

        Keeping this indirection makes the parallel path easier to test and lets
        future routing changes avoid coupling tests to a concrete private
        implementation detail.
        """
        return await self._async_call_tool_isolated(tool_name, **kwargs)

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

    def prewarm(self) -> bool:
        try:
            return bool(self._async_bridge.run(self._ensure_session(), timeout=20.0))
        except Exception as e:
            logger.warning(f"MCP预热失败: {e}")
            return False

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        with self._metrics_lock:
            return dict(self._runtime_metrics)

    def _add_metric(self, key: str, value: float) -> None:
        with self._metrics_lock:
            self._runtime_metrics[key] = self._runtime_metrics.get(key, 0.0) + value

    async def _init_session(self) -> bool:
        if self._initialized and self._is_session_valid():
            return True

        async with self._get_init_lock():
            if self._initialized and self._is_session_valid():
                return True

            if self._initializing:
                logger.debug("MCP会话正在初始化中，等待完成...")
                for _ in range(30):
                    await asyncio.sleep(0.1)
                    if self._initialized and self._is_session_valid():
                        return True
                    if not self._initializing:
                        break
                return self._initialized and self._is_session_valid()

            self._initializing = True
            try:
                return await self._do_init_session()
            finally:
                self._initializing = False

    async def _do_init_session(self) -> bool:
        init_started = time.perf_counter()
        try:
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()

            client = httpx.AsyncClient(timeout=30.0)

            logger.info("正在连接MCP服务器...")
            try:
                health_resp = await client.get(
                    settings.MCP_SERVER_URL,
                    headers={"Accept": "text/event-stream"},
                    timeout=5.0,
                )
                logger.debug(f"MCP服务器健康检查: HTTP {health_resp.status_code}")
            except httpx.ConnectError:
                raise Exception(f"MCP服务器不可达: {settings.MCP_SERVER_URL}")
            except httpx.TimeoutException:
                raise Exception(f"MCP服务器连接超时: {settings.MCP_SERVER_URL}")

            logger.info("正在初始化MCP会话（POST initialize）...")
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "AstroAgent-SkillRouter",
                        "version": "1.0.0",
                    },
                },
                "id": 1,
            }

            response = await client.post(
                settings.MCP_SERVER_URL,
                json=init_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )

            if response.status_code != 200:
                raise Exception(f"初始化请求失败: HTTP {response.status_code}")

            session_id = response.headers.get("mcp-session-id")
            if not session_id:
                raise Exception("初始化响应中未返回session ID")

            logger.info(f"✅ 获取到session ID: {session_id}")

            init_result = _parse_sse_response(response.text)
            if init_result:
                server_info = init_result.get("result", {}).get("serverInfo", {})
                logger.debug(f"初始化成功，服务器信息: {server_info}")

            notif_request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            notif_resp = await client.post(
                settings.MCP_SERVER_URL,
                json=notif_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id,
                },
            )
            if notif_resp.status_code not in (200, 202):
                logger.warning(f"initialized通知返回非预期状态: {notif_resp.status_code}")

            logger.info("获取工具列表...")
            list_request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2,
            }
            response = await client.post(
                settings.MCP_SERVER_URL,
                json=list_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id,
                },
            )

            tools_result = _parse_sse_response(response.text)
            if tools_result:
                tools_list = tools_result.get("result", {}).get("tools", [])
                logger.info(f"✅ 从服务器获取到 {len(tools_list)} 个工具")

            self._session_id = session_id
            self._initialized = True
            self._http_client = client
            self._reconnect_attempts = 0
            self._add_metric(
                "mcp_session_init_ms",
                (time.perf_counter() - init_started) * 1000.0,
            )
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

    async def _async_call_tool(
        self,
        tool_name: str,
        _skip_session_check: bool = False,
        **kwargs,
    ) -> str:
        tool_started = time.perf_counter()

        if _skip_session_check:
            session_ok = self._is_session_valid()
        else:
            session_ok = await self._ensure_session()

        if not session_ok:
            logger.error("❌ MCP会话不可用且重连失败")
            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=ErrorCode.MCP_SESSION_ERROR.value,
                    message="MCP会话不可用且重连失败，请检查桥服务器是否运行",
                    details={"tool_name": tool_name},
                )
            )

        try:
            return await self._execute_tool_call(
                http_client=self._http_client,
                session_id=self._session_id,
                tool_name=tool_name,
                kwargs=kwargs,
            )
        except httpx.TimeoutException:
            logger.error(f"❌ MCP工具调用超时: {tool_name}")
            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=ErrorCode.MCP_TIMEOUT_ERROR.value,
                    message=f"工具 '{tool_name}' 调用超时",
                    details={"tool_name": tool_name},
                )
            )
        except httpx.ConnectError:
            logger.error(f"❌ 无法连接到MCP服务器: {settings.MCP_SERVER_URL}")
            self._initialized = False
            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=ErrorCode.MCP_CONNECTION_ERROR.value,
                    message="无法连接到MCP服务器",
                    details={
                        "tool_name": tool_name,
                        "server_url": settings.MCP_SERVER_URL,
                    },
                )
            )
        except Exception as e:
            logger.error(f"❌ 调用工具 {tool_name} 失败: {e}")
            error = ErrorHandler.handle(e, {"tool_name": tool_name})
            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=error.code.value,
                    message=error.message,
                    details=error.details,
                )
            )
        finally:
            self._add_metric(
                "tool_exec_ms",
                (time.perf_counter() - tool_started) * 1000.0,
            )
            self._add_metric("tool_call_count", 1.0)

    async def _async_call_tool_isolated(self, tool_name: str, **kwargs) -> str:
        """
        Parallel-safe tool call:
        each request uses its own AsyncClient and its own MCP session.
        """
        tool_started = time.perf_counter()
        client: Optional[httpx.AsyncClient] = None
        session_id: Optional[str] = None

        try:
            logger.info(f"[MCP][ISOLATED] 开始独立会话调用 tool={tool_name}")
            client, session_id = await self._create_ephemeral_session()
            return await self._execute_tool_call(
                http_client=client,
                session_id=session_id,
                tool_name=tool_name,
                kwargs=kwargs,
            )
        except httpx.TimeoutException:
            logger.error(f"❌ MCP独立会话工具调用超时: {tool_name}")
            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=ErrorCode.MCP_TIMEOUT_ERROR.value,
                    message=f"工具 '{tool_name}' 调用超时",
                    details={"tool_name": tool_name, "isolated_session": True},
                )
            )
        except httpx.ConnectError:
            logger.error(f"❌ 无法连接到MCP服务器: {settings.MCP_SERVER_URL}")
            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=ErrorCode.MCP_CONNECTION_ERROR.value,
                    message="无法连接到MCP服务器",
                    details={
                        "tool_name": tool_name,
                        "server_url": settings.MCP_SERVER_URL,
                        "isolated_session": True,
                    },
                )
            )
        except Exception as e:
            logger.error(f"❌ 独立会话调用工具 {tool_name} 失败: {e}")
            error = ErrorHandler.handle(
                e,
                {"tool_name": tool_name, "isolated_session": True},
            )
            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=error.code.value,
                    message=error.message,
                    details=error.details,
                )
            )
        finally:
            if client is not None and not client.is_closed:
                try:
                    await client.aclose()
                except Exception:
                    pass
            self._add_metric(
                "tool_exec_ms",
                (time.perf_counter() - tool_started) * 1000.0,
            )
            self._add_metric("tool_call_count", 1.0)

    async def _create_ephemeral_session(self) -> tuple[httpx.AsyncClient, str]:
        """
        Create a short-lived MCP session for isolated parallel calls.
        """
        client = httpx.AsyncClient(timeout=30.0)

        try:
            logger.debug("独立会话：正在连接MCP服务器...")
            try:
                health_resp = await client.get(
                    settings.MCP_SERVER_URL,
                    headers={"Accept": "text/event-stream"},
                    timeout=5.0,
                )
                logger.debug(f"独立会话健康检查: HTTP {health_resp.status_code}")
            except httpx.ConnectError:
                raise Exception(f"MCP服务器不可达: {settings.MCP_SERVER_URL}")
            except httpx.TimeoutException:
                raise Exception(f"MCP服务器连接超时: {settings.MCP_SERVER_URL}")

            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "AstroAgent-SkillRouter",
                        "version": "1.0.0",
                    },
                },
                "id": 1,
            }

            response = await client.post(
                settings.MCP_SERVER_URL,
                json=init_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )

            if response.status_code != 200:
                raise Exception(f"独立会话初始化请求失败: HTTP {response.status_code}")

            session_id = response.headers.get("mcp-session-id")
            if not session_id:
                raise Exception("独立会话初始化响应中未返回session ID")

            init_result = _parse_sse_response(response.text)
            if init_result:
                server_info = init_result.get("result", {}).get("serverInfo", {})
                logger.debug(f"独立会话初始化成功，服务器信息: {server_info}")

            notif_request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            notif_resp = await client.post(
                settings.MCP_SERVER_URL,
                json=notif_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id,
                },
            )
            if notif_resp.status_code not in (200, 202):
                logger.warning(
                    f"独立会话 initialized 通知返回非预期状态: {notif_resp.status_code}"
                )

            return client, session_id

        except Exception:
            if not client.is_closed:
                try:
                    await client.aclose()
                except Exception:
                    pass
            raise

    async def _execute_tool_call(
        self,
        http_client: httpx.AsyncClient,
        session_id: str,
        tool_name: str,
        kwargs: dict,
    ) -> str:
        processed_kwargs = {}
        for key, value in kwargs.items():
            if key in ["year", "month", "limit"]:
                try:
                    if isinstance(value, str) and value.isdigit():
                        processed_kwargs[key] = int(value)
                    else:
                        processed_kwargs[key] = value
                except Exception:
                    processed_kwargs[key] = value
            else:
                processed_kwargs[key] = value

        request_id = int(time.time() * 1000)
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": processed_kwargs,
            },
            "id": request_id,
        }

        logger.debug(
            f"[MCP][CALL] tool={tool_name}, session={session_id}, "
            f"request_id={request_id}, args={processed_kwargs}"
        )

        response = await http_client.post(
            settings.MCP_SERVER_URL,
            json=request,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Mcp-Session-Id": session_id,
            },
            timeout=30.0,
        )

        if response.status_code != 200:
            if response.status_code in (404, 410, 503):
                if session_id == self._session_id:
                    self._initialized = False
                logger.warning(
                    f"MCP会话可能已失效(HTTP {response.status_code})，"
                    f"tool={tool_name}, session={session_id}"
                )

            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=ErrorCode.MCP_SESSION_ERROR.value,
                    message=f"MCP服务器返回HTTP错误: {response.status_code}",
                    details={
                        "tool_name": tool_name,
                        "status_code": response.status_code,
                        "session_id": session_id,
                    },
                )
            )

        result = _parse_sse_response(response.text)
        if not result:
            logger.error(f"无法解析响应: {response.text[:200]}")
            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=ErrorCode.MCP_SESSION_ERROR.value,
                    message="MCP响应解析失败",
                    details={"tool_name": tool_name, "session_id": session_id},
                )
            )

        logger.debug(f"工具响应: {json.dumps(result, ensure_ascii=False)[:500]}")

        if "error" in result:
            error_msg = result["error"].get("message", "未知错误")
            error_code = result["error"].get("code", "")
            return serialize_envelope(
                error_envelope(
                    tool_name=tool_name,
                    code=ErrorCode.TOOL_CALL_FAILED.value,
                    message=f"工具调用错误 [{error_code}]: {error_msg}",
                    details={
                        "tool_name": tool_name,
                        "mcp_error_code": error_code,
                        "session_id": session_id,
                    },
                )
            )

        if "result" in result:
            res = result["result"]

            if isinstance(res, dict):
                if "content" in res:
                    content = res["content"]
                    if isinstance(content, list) and len(content) > 0:
                        for item in content:
                            if item.get("type") == "text":
                                text = item.get("text", "")

                                envelope = parse_tool_response(text)
                                if envelope is not None:
                                    return serialize_envelope(envelope)

                                try:
                                    parsed = json.loads(text)
                                    if isinstance(parsed, dict) and parsed.get("error"):
                                        return serialize_envelope(
                                            error_envelope(
                                                tool_name=tool_name,
                                                code=str(
                                                    parsed.get("code")
                                                    or ErrorCode.TOOL_CALL_FAILED.value
                                                ),
                                                message=parsed.get("message", text),
                                                details=parsed.get("details")
                                                or {"tool_name": tool_name},
                                            )
                                        )
                                except (json.JSONDecodeError, TypeError):
                                    pass

                                return text

                return json.dumps(res, ensure_ascii=False)

            if isinstance(res, str):
                return res

            return str(res)

        logger.warning(f"未知响应格式: {result}")
        return str(result)


def _parse_sse_response(response_text: str) -> Optional[dict]:
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
