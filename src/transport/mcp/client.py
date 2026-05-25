"""MCP 客户端，负责 Streamable HTTP 会话初始化、SSE 响应解析和工具调用。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
from typing import Any, Dict, Optional

import httpx

from src.core.config import settings
from src.core.errors import ErrorCode, ErrorHandler
from src.core.logger import logger
from src.transport.mcp.envelope import (
    error_envelope,
    parse_tool_response,
    serialize_envelope,
)
from src.transport.mcp.sse import parse_sse_response

MCP_RECONNECT_MAX_RETRIES = 3
MCP_RECONNECT_DELAY = 2.0


class _AsyncBridge:
    """
    在同步上下文中安全运行异步 MCP 操作的桥接器。

    通过后台守护线程创建专用事件循环，避免在 FastAPI 已运行事件循环中
    调用 asyncio.run() 导致 RuntimeError 或死锁。
    """

    def __init__(self) -> None:
        """初始化后台事件循环、线程和启动同步信号。"""
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self) -> None:
        """启动后台事件循环线程。"""
        if self._loop is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=10)

    def _run_loop(self) -> None:
        """在线程中创建并持续运行 asyncio 事件循环。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run(self, coro: Any, timeout: float = 60.0) -> Any:
        """把协程提交到后台事件循环并同步等待结果。"""
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
        """停止后台事件循环并等待线程退出。"""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None


class MCPClient:
    """
    MCP 协议客户端，处理会话生命周期、SSE 解析和工具调用。

    职责：
    - 初始化并维护 MCP 会话（SSE 握手、协议初始化、工具列表）
    - 会话失败时重连
    - 解析 MCP server 返回的 SSE 响应
    - 携带会话头调用 MCP 工具并统一错误处理
    """

    def __init__(self) -> None:
        """初始化 MCP 会话状态、HTTP 客户端、异步桥和运行时指标。"""
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
        """获取用于串行化 MCP 会话初始化的异步锁。"""
        if self._init_lock is None:
            try:
                asyncio.get_running_loop()
                self._init_lock = asyncio.Lock()
            except RuntimeError:
                self._init_lock = asyncio.Lock()
        return self._init_lock

    def invoke(self, tool_name: str, **kwargs) -> str:
        """从同步上下文调用单个 MCP 工具。"""
        return self._async_bridge.run(self._async_call_tool(tool_name, **kwargs))

    def invoke_parallel(self, calls: list[dict]) -> list[str]:
        """
        使用独立会话批量并行调用 MCP 工具。

        calls 的元素格式为 {"tool_name": str, "kwargs": dict}，返回值顺序与输入一致。
        """

        async def _gather():
            """并发派发本批 MCP 工具调用。"""
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
            return [self.invoke(c["tool_name"], **c.get("kwargs", {})) for c in calls]

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
        批量工具派发的独立入口。

        保留这一层便于测试并隔离后续并行路由实现调整。
        """
        return await self._async_call_tool_isolated(tool_name, **kwargs)

    async def ainvoke(self, tool_name: str, **kwargs) -> str:
        """从异步上下文调用单个 MCP 工具。"""
        return await self._async_call_tool(tool_name, **kwargs)

    async def ainvoke_parallel(self, calls: list[dict]) -> list[str]:
        """从异步上下文批量并行调用 MCP 工具。"""
        tasks = [
            self._dispatch_parallel_tool_call(
                call["tool_name"],
                **call.get("kwargs", {}),
            )
            for call in calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_name = calls[i].get("tool_name", "parallel_call")
                error = ErrorHandler.handle(
                    result,
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
                final.append(result)
        return final

    def shutdown(self) -> None:
        """关闭异步桥和 HTTP 客户端。"""
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
        """预先建立 MCP 会话，降低首个工具调用延迟。"""
        try:
            return bool(self._async_bridge.run(self._ensure_session(), timeout=20.0))
        except Exception as e:
            logger.warning(f"MCP预热失败: {e}")
            return False

    def get_runtime_metrics_snapshot(self) -> Dict[str, float]:
        """返回 MCP 会话初始化和工具调用耗时指标。"""
        with self._metrics_lock:
            return dict(self._runtime_metrics)

    def _add_metric(self, key: str, value: float) -> None:
        """累加一个运行时指标。"""
        with self._metrics_lock:
            self._runtime_metrics[key] = self._runtime_metrics.get(key, 0.0) + value

    async def _init_session(self) -> bool:
        """确保 MCP 主会话初始化完成并可用。"""
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
        """执行 MCP initialize、initialized 通知和 tools/list 流程。"""
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

            init_result = parse_sse_response(response.text)
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
                logger.warning(
                    f"initialized通知返回非预期状态: {notif_resp.status_code}"
                )

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

            tools_result = parse_sse_response(response.text)
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
        """判断当前 MCP 主会话和 HTTP 客户端是否可继续使用。"""
        if not self._initialized or not self._session_id:
            return False
        if self._http_client is None or self._http_client.is_closed:
            return False
        return True

    async def _ensure_session(self) -> bool:
        """确保当前存在可用 MCP 会话，不可用时触发重连。"""
        if self._is_session_valid():
            return True
        logger.warning("MCP会话无效或未初始化，尝试建立连接...")
        return await self._reconnect()

    async def _reconnect(self) -> bool:
        """按配置次数重试 MCP 会话初始化。"""
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
        """在主 MCP 会话上执行一次工具调用并封装错误。"""
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
        并行安全的工具调用。

        每个请求使用独立 AsyncClient 和独立 MCP 会话，避免共享会话并发干扰。
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
        为独立并行工具调用创建短生命周期 MCP 会话。
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

            init_result = parse_sse_response(response.text)
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
        """向 MCP server 发送 tools/call 请求并解析统一工具结果。"""
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

        result = parse_sse_response(response.text)
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
