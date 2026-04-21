import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from src.agent.latency import LatencyTracker
from src.agent.streaming_events import (
    FrontendJsonEventAdapter,
    PlainTextStreamAdapter,
    SSEEventAdapter,
    StreamEvent,
    StreamEventAdapter,
    StreamEventProcessor,
    apply_event_processors,
)
from src.core.errors import ErrorHandler
from src.core.logger import logger
from src.core.mcp_protocol import extract_tool_data, is_tool_error
from src.memory.api.dto import (
    AppendMessageRequest,
    AppendToolCallRequest,
    BuildContextRequest,
)

MAX_ACTION_HISTORY_ENTRIES = 100
TOOL_INPUT_LOG_PREVIEW_CHARS = 300
TOOL_OUTPUT_LOG_PREVIEW_CHARS = 200
FALLBACK_TOOL_OUTPUT_PREVIEW_CHARS = 500


class BaseStreamingGenerator:
    def __init__(
        self,
        agent_executor: Any,
        memory: Any,
        long_term_memory: Any = None,
        user_id: str = "anonymous",
        fallback_service: Optional[Any] = None,
        request_router: Optional[Any] = None,
        task_orchestrator: Optional[Any] = None,
        skill_manager: Optional[Any] = None,
        rag_retriever: Optional[Any] = None,
        event_processors: Optional[list[StreamEventProcessor]] = None,
    ):
        self._agent_executor = agent_executor
        self._memory = memory
        self._long_term_memory = long_term_memory
        self._user_id = user_id
        self._fallback_service = fallback_service
        self._request_router = request_router
        self._task_orchestrator = task_orchestrator
        self._skill_manager = skill_manager
        self._rag_retriever = rag_retriever
        self._tool_runs: Dict[str, Dict[str, Any]] = {}
        self._current_request_id: Optional[str] = None
        self._current_query: str = ""
        self._action_history: OrderedDict[str, list] = OrderedDict()
        self._max_same_action_count = 2
        self._event_processors = list(event_processors or [])

    def _cleanup_action_history(self, request_id: Optional[str] = None):
        if request_id and request_id in self._action_history:
            del self._action_history[request_id]
            logger.debug(f"已清理请求 {request_id} 的操作历史")

        while len(self._action_history) > MAX_ACTION_HISTORY_ENTRIES:
            oldest_key, _ = self._action_history.popitem(last=False)
            logger.debug(f"LRU淘汰最旧的操作历史: {oldest_key}")

    def _log_memory_usage(self):
        total_entries = len(self._action_history)
        total_actions = sum(len(v) for v in self._action_history.values())
        logger.debug(
            f"操作历史内存状态: {total_entries} 个请求, 共 {total_actions} 条动作记录"
        )

    def _format_chat_history(self) -> str:
        try:
            context = self._memory.build_context(
                BuildContextRequest(
                    session_id=self._memory_session_id(),
                    query=self._current_query or "",
                )
            )
            context_text = context.get("context_text", "")
            return context_text or "无历史对话"
        except Exception as e:
            logger.warning(f"build_context失败，使用空历史: {type(e).__name__}: {e}")
            return "无历史对话"

    def _format_user_profile(self) -> str:
        if not self._long_term_memory:
            return "暂无用户偏好信息"
        if hasattr(self._long_term_memory, "format_smart_prompt"):
            return self._long_term_memory.format_smart_prompt(
                self._user_id, self._current_query or ""
            )
        if hasattr(self._long_term_memory, "format_profile_for_prompt"):
            return self._long_term_memory.format_profile_for_prompt(self._user_id)
        return "暂无用户偏好信息"

    def _extract_and_update_long_term_memory(
        self, user_message: str, assistant_message: str
    ):
        if not self._long_term_memory:
            return

        try:
            if hasattr(self._long_term_memory, "extract_and_store"):
                results = self._long_term_memory.extract_and_store(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    user_id=self._user_id,
                )
                if results:
                    logger.info(
                        f"✅ 已提取并更新长期记忆 (user_id: {self._user_id}, count: {len(results)})"
                    )
            elif hasattr(self._long_term_memory, "extract_from_conversation"):
                extracted = self._long_term_memory.extract_from_conversation(
                    user_message, assistant_message
                )
                has_info = (
                    extracted.get("preferences")
                    or extracted.get("habits")
                    or extracted.get("constraints")
                )
                if has_info:
                    self._long_term_memory.merge_and_update(self._user_id, extracted)
                    logger.info(f"✅ 已提取并更新长期记忆 (user_id: {self._user_id})")
        except Exception as e:
            logger.error(f"❌ 更新长期记忆失败: {e}")

    def _schedule_long_term_memory_update(
        self, user_message: str, assistant_message: str
    ) -> None:
        if not self._long_term_memory:
            return

        def _runner() -> None:
            started = time.perf_counter()
            self._extract_and_update_long_term_memory(user_message, assistant_message)
            logger.info(
                "长期记忆后台更新完成: user_id=%s, ltm_extract_ms=%.2f",
                self._user_id,
                (time.perf_counter() - started) * 1000.0,
            )

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

    def _check_repeated_action(
        self, request_id: str, tool_name: str, tool_input: str
    ) -> bool:
        if request_id not in self._action_history:
            self._action_history[request_id] = []

        action_key = f"{tool_name}:{tool_input}"
        history = self._action_history[request_id]

        history.append(action_key)
        same_count = sum(1 for h in history if h == action_key)

        if same_count >= self._max_same_action_count:
            logger.warning(
                f"[{request_id}] ⚠️ 检测到重复操作：{tool_name} 已执行 {same_count} 次，强制终止"
            )
            return True

        return False

    def _build_response_from_intermediate_steps(
        self, query: str, intermediate_steps: list
    ) -> str:
        if not intermediate_steps:
            return ""

        tool_results = []
        for step in intermediate_steps:
            if hasattr(step, "__iter__") and len(step) >= 2:
                action, observation = step[0], step[1]
                tool_name = getattr(action, "tool", "unknown")
                tool_input = getattr(action, "tool_input", "")
                if observation and not ErrorHandler.is_error_response(observation):
                    if is_tool_error(observation):
                        continue
                    try:
                        obs_data = extract_tool_data(str(observation))
                        if isinstance(obs_data, str):
                            obs_data = json.loads(obs_data)
                        if isinstance(obs_data, dict) and obs_data.get("error"):
                            continue
                    except (json.JSONDecodeError, TypeError):
                        pass
                    tool_results.append(
                        {
                            "tool": tool_name,
                            "input": self._preview_text(tool_input, 100),
                            "output": self._preview_text(
                                observation, FALLBACK_TOOL_OUTPUT_PREVIEW_CHARS
                            ),
                        }
                    )

        if not tool_results:
            return ""

        response_parts = [f"根据已获取的信息，为您回答「{query}」：\n"]

        for i, result in enumerate(tool_results, 1):
            tool_name = result["tool"]
            output = result["output"]

            try:
                output_data = json.loads(output)
                if isinstance(output_data, dict):
                    if "error" in output_data:
                        continue
                    if "answer" in output_data:
                        response_parts.append(f"\n{output_data['answer']}")
                        continue
                    if "name" in output_data:
                        response_parts.append(
                            f"\n目标名称：{output_data.get('name', '未知')}"
                        )
                    if "ra" in output_data and "dec" in output_data:
                        response_parts.append(
                            f"位置：赤经 {output_data['ra']}°，赤纬 {output_data['dec']}°"
                        )
                    for key, value in list(output_data.items())[:5]:
                        if key not in ["name", "ra", "dec", "error", "answer"]:
                            response_parts.append(f"\n{key}：{value}")
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            if len(output) > 50:
                response_parts.append(f"\n{output}")

        response_parts.append(
            "\n\n（注：由于处理时间限制，以上是基于已获取数据整理的信息。）"
        )
        return "".join(response_parts)

    def _prepare_input(
        self, query: str, use_long_term_memory: bool = True
    ) -> Dict[str, str]:
        self._current_query = query
        chat_history = self._format_chat_history()
        user_profile = (
            self._format_user_profile()
            if use_long_term_memory
            else "本轮对话已禁用长期记忆"
        )
        return {
            "input": query,
            "chat_history": chat_history,
            "user_profile": user_profile,
        }

    def _infer_plan_steps(
        self, query: str, use_long_term_memory: bool
    ) -> List[Dict[str, Any]]:
        steps = [
            {
                "id": "understand",
                "title": "解析任务",
                "description": f"理解用户问题并构建执行路径: {self._preview_text(query, 80)}",
                "status": "running",
            }
        ]
        if use_long_term_memory:
            steps.append(
                {
                    "id": "memory",
                    "title": "读取记忆",
                    "description": "检索与当前问题相关的长期记忆与用户偏好",
                    "status": "pending",
                }
            )
        steps.extend(
            [
                {
                    "id": "tools",
                    "title": "调用工具",
                    "description": "按需执行检索、观测或多模态工具",
                    "status": "pending",
                },
                {
                    "id": "answer",
                    "title": "生成答案",
                    "description": "汇总证据、记忆和工具结果形成最终回答",
                    "status": "pending",
                },
            ]
        )
        return steps

    def _update_plan_status(
        self, steps: List[Dict[str, Any]], step_id: str, status: str
    ) -> List[Dict[str, Any]]:
        updated: List[Dict[str, Any]] = []
        for step in steps:
            clone = dict(step)
            if clone["id"] == step_id:
                clone["status"] = status
            updated.append(clone)
        return updated

    def _explain_memory_hits(
        self, query: str, use_long_term_memory: bool
    ) -> List[Dict[str, Any]]:
        if not use_long_term_memory or not self._long_term_memory:
            return []
        if hasattr(self._long_term_memory, "explain_memory_hits"):
            try:
                hits = self._long_term_memory.explain_memory_hits(self._user_id, query)
                if isinstance(hits, list):
                    return [hit for hit in hits if isinstance(hit, dict)]
            except Exception as e:
                logger.warning(f"memory explain 失败: {type(e).__name__}: {e}")
        return []

    def _extract_evidence(
        self, tool_name: str, tool_output: str, run_id: str
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        preview = self._preview_text(tool_output, 240)
        if preview:
            evidence.append(
                {
                    "source_id": run_id,
                    "kind": "tool_output",
                    "title": tool_name,
                    "snippet": preview,
                    "tool": tool_name,
                }
            )
        if tool_output.strip().startswith("{"):
            try:
                payload = json.loads(tool_output)
                if isinstance(payload, dict):
                    for key in ("source", "sources", "url", "hdurl", "reference"):
                        value = payload.get(key)
                        if value:
                            evidence.append(
                                {
                                    "source_id": f"{run_id}:{key}",
                                    "kind": "reference",
                                    "title": f"{tool_name}.{key}",
                                    "snippet": self._preview_text(value, 240),
                                    "tool": tool_name,
                                }
                            )
            except Exception:
                pass
        return evidence

    def _compute_confidence(
        self,
        tool_runs: List[Dict[str, Any]],
        memory_hits: List[Dict[str, Any]],
        evidence_items: List[Dict[str, Any]],
    ) -> float:
        confidence = 0.35
        if tool_runs:
            confidence += min(len(tool_runs) * 0.1, 0.3)
            if all(item.get("status") == "success" for item in tool_runs):
                confidence += 0.1
        if memory_hits:
            confidence += 0.1
        if evidence_items:
            confidence += 0.1
        return min(round(confidence, 2), 0.95)

    @staticmethod
    def _preview_text(value: Any, max_len: int) -> str:
        text = "" if value is None else str(value)
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _handle_tool_start(
        self, request_id: str, run_id: str, data: dict, check_repeated: bool = False
    ) -> Optional[str]:
        tool_name = data.get("name") or data.get("tool")
        tool_input = data.get("input")
        tool_input_str = str(tool_input) if tool_input else ""

        if check_repeated and self._check_repeated_action(
            request_id, tool_name or "unknown_tool", tool_input_str
        ):
            return "repeated"

        self._tool_runs[run_id] = {
            "name": tool_name or "unknown_tool",
            "input": tool_input_str,
            "start_time": time.time(),
            "request_id": request_id,
        }
        logger.info(
            json.dumps(
                {
                    "type": "tool_start",
                    "request_id": request_id,
                    "run_id": run_id,
                    "tool_name": tool_name or "unknown_tool",
                    "input": self._preview_text(
                        tool_input, TOOL_INPUT_LOG_PREVIEW_CHARS
                    ),
                },
                ensure_ascii=False,
            )
        )
        return None

    def _handle_tool_end(
        self, request_id: str, run_id: str, data: dict
    ) -> Dict[str, Any]:
        meta = self._tool_runs.pop(run_id, {})
        duration = None
        if meta.get("start_time") is not None:
            duration = time.time() - meta["start_time"]
        tool_output = data.get("output")
        tool_output_str = "" if tool_output is None else str(tool_output)
        success = True
        if ErrorHandler.is_error_response(tool_output):
            success = False
        elif tool_output_str.strip().startswith("{"):
            try:
                parsed_output = json.loads(tool_output_str)
                success = not ErrorHandler.is_error_response(parsed_output)
            except Exception:
                success = "error" not in tool_output_str.lower()

        logger.info(
            json.dumps(
                {
                    "type": "tool_end",
                    "request_id": request_id,
                    "run_id": run_id,
                    "tool_name": meta.get("name"),
                    "duration_sec": duration,
                    "success": success,
                    "output_preview": self._preview_text(
                        tool_output_str, TOOL_OUTPUT_LOG_PREVIEW_CHARS
                    ),
                },
                ensure_ascii=False,
            )
        )

        if meta:
            try:
                self._memory.append_tool_call(
                    AppendToolCallRequest(
                        session_id=self._memory_session_id(),
                        user_id=self._user_id,
                        turn_id=request_id,
                        tool_name=meta.get("name") or "unknown_tool",
                        tool_input=meta.get("input", ""),
                        raw_output=tool_output_str,
                        timestamp=time.time(),
                        success=success,
                        content_type=self._infer_content_type(tool_output_str),
                    )
                )
            except Exception as e:
                logger.error(
                    f"❌ memory tool-call append failed: {type(e).__name__}: {e}"
                )

        extracted_url = None
        if tool_output_str.strip().startswith("{"):
            try:
                obj = json.loads(tool_output_str)
                if isinstance(obj, dict):
                    extracted_url = obj.get("hdurl") or obj.get("url")
            except Exception:
                extracted_url = None
        if not extracted_url and self._fallback_service:
            extracted_url = self._fallback_service.extract_image_url(tool_output_str)
        elif not extracted_url:
            from src.agent.param_parser import ParamParser

            extracted_url = ParamParser.extract_image_url(tool_output_str)

        return {
            "meta": meta,
            "duration": duration,
            "tool_output_str": tool_output_str,
            "extracted_url": extracted_url,
        }

    def _save_to_memory(
        self,
        query: str,
        response: str,
        *,
        use_long_term_memory: bool = True,
        latency: Optional[LatencyTracker] = None,
    ):
        try:
            with latency.measure("memory_save_ms") if latency else _nullcontext():
                self._memory.append_message(
                    AppendMessageRequest(
                        session_id=self._memory_session_id(),
                        user_id=self._user_id,
                        turn_id=self._current_request_id,
                        role="user",
                        content=query,
                        timestamp=time.time(),
                    )
                )
                self._memory.append_message(
                    AppendMessageRequest(
                        session_id=self._memory_session_id(),
                        user_id=self._user_id,
                        turn_id=self._current_request_id,
                        role="assistant",
                        content=response,
                        timestamp=time.time(),
                    )
                )
        except Exception as e:
            logger.error(f"❌ memory message append failed: {type(e).__name__}: {e}")

        if use_long_term_memory:
            self._schedule_long_term_memory_update(query, response)
            if latency:
                latency.set_meta("ltm_async", True)

    def _finalize_request(self, request_id: Optional[str]):
        self._current_request_id = None
        self._cleanup_action_history(request_id)
        self._log_memory_usage()

    def _memory_session_id(self) -> str:
        session_id = getattr(self._memory, "session_id", None)
        if session_id:
            return session_id
        return f"mem_{self._user_id}"

    @staticmethod
    def _infer_content_type(text: str) -> str:
        stripped = (text or "").strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(stripped)
                return "application/json"
            except Exception:
                return "text/plain"
        return "text/plain"

    def _extract_stream_text(self, data: dict) -> Optional[str]:
        chunk = data.get("chunk")
        if not chunk:
            return None
        text = getattr(chunk, "content", None) or getattr(chunk, "text", None)
        return text

    def _parse_thinking_and_final_answer(
        self,
        text: str,
        thinking_buffer: list[str],
        final_answer_started: bool,
        final_answer_extracted: bool,
        thinking_logged: bool,
        request_id: str,
    ) -> Dict[str, Any]:
        thinking_buffer.append(text)
        combined_thinking = "".join(thinking_buffer)

        result = {
            "thinking_buffer": thinking_buffer,
            "final_answer_started": final_answer_started,
            "final_answer_extracted": final_answer_extracted,
            "thinking_logged": thinking_logged,
            "final_answer_text": None,
            "thinking_text": None,
            "is_thinking": False,
            "is_final_answer_chunk": False,
            "should_continue": True,
        }

        if not final_answer_started:
            if re.search(
                r"(Thought:|Action:|Observation:)\s*$", combined_thinking, re.IGNORECASE
            ):
                result["is_thinking"] = True
                result["thinking_text"] = text
            elif re.search(r"Final Answer:\s*", combined_thinking, re.IGNORECASE):
                result["final_answer_started"] = True
                if not thinking_logged and combined_thinking:
                    result["thinking_logged"] = True
                    logger.info(f"[{request_id}] 🔍 Thinking: {combined_thinking}")
                match = re.search(
                    r"Final Answer:\s*(.*)",
                    combined_thinking,
                    re.IGNORECASE | re.DOTALL,
                )
                if match and not final_answer_extracted:
                    result["final_answer_extracted"] = True
                    final_answer_text = match.group(1).strip()
                    if final_answer_text:
                        result["final_answer_text"] = final_answer_text
                        logger.info(
                            f"[{request_id}] Final Answer: {final_answer_text[:100]}..."
                        )
                result["thinking_buffer"] = []
                result["should_continue"] = False
                return result
            else:
                result["is_thinking"] = True
                result["thinking_text"] = text
        else:
            result["is_final_answer_chunk"] = True

        return result

    def register_event_processor(self, processor: StreamEventProcessor):
        self._event_processors.append(processor)

    def clear_event_processors(self):
        self._event_processors.clear()

    def _build_stream_event(
        self,
        event_type: str,
        sequence: int,
        request_id: str,
        *,
        content: Any = None,
        meta: Optional[Dict[str, Any]] = None,
        modality: str = "text",
    ) -> StreamEvent:
        payload = {"request_id": request_id}
        if meta:
            payload.update(meta)
        return StreamEvent(
            type=event_type,
            content=content,
            meta=payload,
            sequence=sequence,
            modality=modality,
        ).validate()

    def _snapshot_runtime_metrics(self) -> Dict[str, Dict[str, float]]:
        mcp = {}
        rag = {}
        if self._skill_manager and hasattr(
            self._skill_manager, "get_runtime_metrics_snapshot"
        ):
            try:
                mcp = self._skill_manager.get_runtime_metrics_snapshot()
            except Exception:
                mcp = {}
        if self._rag_retriever and hasattr(
            self._rag_retriever, "get_runtime_metrics_snapshot"
        ):
            try:
                rag = self._rag_retriever.get_runtime_metrics_snapshot()
            except Exception:
                rag = {}
        return {"mcp": mcp, "rag": rag}

    @staticmethod
    def _metric_delta(
        after: Dict[str, float], before: Dict[str, float], key: str
    ) -> float:
        return round(after.get(key, 0.0) - before.get(key, 0.0), 2)

    def _record_runtime_metric_deltas(
        self,
        latency: LatencyTracker,
        before: Dict[str, Dict[str, float]],
        after: Dict[str, Dict[str, float]],
    ) -> None:
        latency.record_ms(
            "mcp_session_init_ms",
            self._metric_delta(
                after.get("mcp", {}), before.get("mcp", {}), "mcp_session_init_ms"
            ),
        )
        latency.record_ms(
            "tool_exec_ms",
            self._metric_delta(
                after.get("mcp", {}), before.get("mcp", {}), "tool_exec_ms"
            ),
        )
        latency.record_ms(
            "rag_total_ms",
            self._metric_delta(
                after.get("rag", {}), before.get("rag", {}), "rag_total_ms"
            ),
        )
        latency.record_ms(
            "rerank_ms",
            self._metric_delta(
                after.get("rag", {}), before.get("rag", {}), "rerank_ms"
            ),
        )

    async def _run_direct_path(
        self,
        query: str,
        decision: Any,
        *,
        use_long_term_memory: bool,
        latency: LatencyTracker,
    ) -> Dict[str, Any]:
        if not self._task_orchestrator:
            raise ValueError("task orchestrator is not configured")
        with latency.measure("agent_prepare_ms"):
            chat_history = self._format_chat_history()
            self._current_query = query
            user_profile = (
                self._format_user_profile()
                if use_long_term_memory
                else "本轮对话已禁用长期记忆"
            )
        with latency.measure("agent_total_ms"):
            result = await self._task_orchestrator.run(
                decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
            )
        return result

    async def _stream_with_adapter(
        self,
        query: str,
        adapter: StreamEventAdapter,
        *,
        stop_on_repeated_action: bool,
        response_mode: str,
        use_long_term_memory: bool = True,
    ) -> AsyncGenerator[Any, None]:
        async for event in self._generate_internal_events(
            query,
            stop_on_repeated_action=stop_on_repeated_action,
            response_mode=response_mode,
            use_long_term_memory=use_long_term_memory,
        ):
            adapted = adapter.adapt(event)
            if adapted is not None:
                yield adapted

    async def _generate_internal_events(
        self,
        query: str,
        *,
        stop_on_repeated_action: bool,
        response_mode: str,
        use_long_term_memory: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        request_id = uuid.uuid4().hex[:8]
        request_started_at = time.time()
        latency = LatencyTracker()
        self._current_request_id = request_id
        self._current_query = query
        logger.info(f"[{request_id}] 开始处理统一流式查询：{query[:200]}")

        sequence = 0
        response_chunks: list[str] = []
        thinking_buffer: list[str] = []
        final_answer_started = False
        final_answer_extracted = False
        thinking_logged = False
        response_saved = False
        plan_steps = self._infer_plan_steps(query, use_long_term_memory)
        memory_hits = self._explain_memory_hits(query, use_long_term_memory)
        tool_timeline: List[Dict[str, Any]] = []
        evidence_items: List[Dict[str, Any]] = []
        reasoning_fragments: List[str] = []
        runtime_metrics_before = self._snapshot_runtime_metrics()

        def next_event(
            event_type: str,
            *,
            content: Any = None,
            meta: Optional[Dict[str, Any]] = None,
            modality: str = "text",
        ) -> StreamEvent:
            nonlocal sequence
            sequence += 1
            return self._build_stream_event(
                event_type,
                sequence,
                request_id,
                content=content,
                meta=meta,
                modality=modality,
            )

        async def emit(event: StreamEvent) -> AsyncGenerator[StreamEvent, None]:
            processed_events = apply_event_processors(event, self._event_processors)
            for processed in processed_events:
                yield processed

        try:
            async for processed in emit(
                next_event("plan_update", content={"steps": plan_steps})
            ):
                yield processed

            async for processed in emit(
                next_event(
                    "step_start",
                    content={"step_id": "understand", "title": "解析任务"},
                )
            ):
                yield processed

            plan_steps = self._update_plan_status(plan_steps, "understand", "done")
            async for processed in emit(
                next_event(
                    "step_end",
                    content={
                        "step_id": "understand",
                        "title": "解析任务",
                        "status": "done",
                    },
                )
            ):
                yield processed

            if use_long_term_memory:
                plan_steps = self._update_plan_status(plan_steps, "memory", "running")
                async for processed in emit(
                    next_event(
                        "step_start",
                        content={"step_id": "memory", "title": "读取记忆"},
                    )
                ):
                    yield processed
                for hit in memory_hits:
                    evidence_items.append(
                        {
                            "source_id": hit["memory_id"],
                            "kind": "memory",
                            "title": f'{hit["memory_type"]}.{hit["key"]}',
                            "snippet": self._preview_text(hit["value"], 240),
                            "reason": hit["reason"],
                        }
                    )
                    async for processed in emit(next_event("memory_hit", content=hit)):
                        yield processed
                plan_steps = self._update_plan_status(plan_steps, "memory", "done")
                async for processed in emit(
                    next_event(
                        "step_end",
                        content={
                            "step_id": "memory",
                            "title": "读取记忆",
                            "status": "done",
                        },
                    )
                ):
                    yield processed

            decision = None
            if self._request_router and hasattr(self._request_router, "route"):
                with latency.measure("route_decision_ms"):
                    candidate = self._request_router.route(query)
                if isinstance(getattr(candidate, "route", None), str) and hasattr(
                    candidate, "to_meta"
                ):
                    decision = candidate
                    async for processed in emit(
                        next_event("route_decision", content=decision.to_meta())
                    ):
                        yield processed

            plan_steps = self._update_plan_status(plan_steps, "tools", "running")
            async for processed in emit(
                next_event("plan_update", content={"steps": plan_steps})
            ):
                yield processed
            async for processed in emit(
                next_event(
                    "step_start",
                    content={"step_id": "tools", "title": "调用工具"},
                )
            ):
                yield processed

            if decision and decision.route != "complex_agent":
                direct_result = await self._run_direct_path(
                    query,
                    decision,
                    use_long_term_memory=use_long_term_memory,
                    latency=latency,
                )
                direct_answer = direct_result.get("answer", "")
                response_chunks.append(direct_answer)
                for tool_run in direct_result.get("tools_used", []):
                    tool_timeline.append(tool_run)
                for source in direct_result.get("sources", []):
                    evidence_items.append(source)
                    async for processed in emit(
                        next_event("evidence_found", content=source)
                    ):
                        yield processed
                if direct_result.get("retrieval", {}).get("stages_ms"):
                    for key, value in direct_result["retrieval"]["stages_ms"].items():
                        latency.record_ms(key, value)
                async for processed in emit(
                    next_event("final_answer_delta", content=direct_answer)
                ):
                    yield processed
            else:
                with latency.measure("agent_prepare_ms"):
                    agent_input = self._prepare_input(
                        query, use_long_term_memory=use_long_term_memory
                    )
                agent_started = time.perf_counter()
                async for raw_event in self._agent_executor.astream_events(
                    agent_input,
                    version="v1",
                ):
                    event_type = raw_event.get("event")
                    data = raw_event.get("data", {}) or {}
                    run_id = raw_event.get("run_id")

                    if event_type == "on_tool_start":
                        async for processed in emit(
                            next_event(
                                "tool_start",
                                content={
                                    "tool": (
                                        data.get("name")
                                        or data.get("tool")
                                        or "unknown_tool"
                                    ),
                                    "input": (
                                        ""
                                        if data.get("input") is None
                                        else str(data.get("input"))
                                    ),
                                    "status": "running",
                                },
                                meta={
                                    "run_id": run_id,
                                    "tool": data.get("name")
                                    or data.get("tool")
                                    or "unknown_tool",
                                },
                            )
                        ):
                            yield processed
                        tool_name = (
                            data.get("name") or data.get("tool") or "unknown_tool"
                        )
                        tool_input = (
                            "" if data.get("input") is None else str(data.get("input"))
                        )
                        result = self._handle_tool_start(
                            request_id,
                            run_id,
                            data,
                            check_repeated=stop_on_repeated_action,
                        )
                        if result == "repeated":
                            warning = "\n\n⚠️ 检测到重复操作，已自动终止循环。"
                            response_chunks.append(warning)
                            async for processed in emit(
                                next_event(
                                    "warning",
                                    content=warning,
                                    meta={
                                        "reason": "repeated_action",
                                        "run_id": run_id,
                                        "tool": tool_name,
                                    },
                                )
                            ):
                                yield processed
                            break
                        continue

                    if event_type == "on_tool_end":
                        tool_result = self._handle_tool_end(request_id, run_id, data)
                        tool_meta = tool_result.get("meta", {}) or {}
                        async for processed in emit(
                            next_event(
                                "tool_end",
                                content={
                                    "tool": tool_meta.get("name"),
                                    "output": tool_result.get("tool_output_str", ""),
                                    "output_summary": self._preview_text(
                                        tool_result.get("tool_output_str", ""), 240
                                    ),
                                    "status": (
                                        "success"
                                        if not ErrorHandler.is_error_response(
                                            tool_result.get("tool_output_str", "")
                                        )
                                        else "error"
                                    ),
                                },
                                meta={
                                    "run_id": run_id,
                                    "tool": tool_meta.get("name"),
                                    "duration_sec": tool_result.get("duration"),
                                    "extracted_url": tool_result.get("extracted_url"),
                                },
                            )
                        ):
                            yield processed
                        tool_timeline.append(
                            {
                                "run_id": run_id,
                                "tool": tool_meta.get("name"),
                                "input": tool_meta.get("input", ""),
                                "output_summary": self._preview_text(
                                    tool_result.get("tool_output_str", ""), 240
                                ),
                                "duration_sec": tool_result.get("duration"),
                                "status": (
                                    "success"
                                    if not ErrorHandler.is_error_response(
                                        tool_result.get("tool_output_str", "")
                                    )
                                    else "error"
                                ),
                            }
                        )
                        for evidence in self._extract_evidence(
                            tool_meta.get("name") or "unknown_tool",
                            tool_result.get("tool_output_str", ""),
                            run_id,
                        ):
                            evidence_items.append(evidence)
                            async for processed in emit(
                                next_event("evidence_found", content=evidence)
                            ):
                                yield processed
                        if tool_result.get("extracted_url"):
                            async for processed in emit(
                                next_event(
                                    "image",
                                    content=tool_result["extracted_url"],
                                    meta={
                                        "run_id": run_id,
                                        "tool": tool_meta.get("name"),
                                    },
                                    modality="image",
                                )
                            ):
                                yield processed
                        continue

                    if event_type not in ("on_chat_model_stream", "on_llm_stream"):
                        continue

                    text = self._extract_stream_text(data)
                    if not text:
                        continue

                    if "llm_first_token_ms" not in latency.stages_ms():
                        latency.record_ms(
                            "llm_first_token_ms",
                            (time.perf_counter() - agent_started) * 1000.0,
                        )

                    if response_mode == "raw_text":
                        response_chunks.append(text)
                    else:
                        reasoning_fragments.append(text)

                    async for processed in emit(
                        next_event(
                            "raw_text_delta", content=text, meta={"run_id": run_id}
                        )
                    ):
                        yield processed

                    parse_result = self._parse_thinking_and_final_answer(
                        text,
                        thinking_buffer,
                        final_answer_started,
                        final_answer_extracted,
                        thinking_logged,
                        request_id,
                    )
                    thinking_buffer = parse_result["thinking_buffer"]
                    final_answer_started = parse_result["final_answer_started"]
                    final_answer_extracted = parse_result["final_answer_extracted"]
                    thinking_logged = parse_result["thinking_logged"]

                    if not parse_result["should_continue"]:
                        if parse_result["final_answer_text"]:
                            final_answer_text = parse_result["final_answer_text"]
                            if response_mode == "final_answer":
                                response_chunks.append(final_answer_text)
                            async for processed in emit(
                                next_event(
                                    "final_answer_delta",
                                    content=final_answer_text,
                                    meta={"run_id": run_id},
                                )
                            ):
                                yield processed
                        continue

                    if parse_result["is_final_answer_chunk"]:
                        if response_mode == "final_answer":
                            response_chunks.append(text)
                        async for processed in emit(
                            next_event(
                                "final_answer_delta",
                                content=text,
                                meta={"run_id": run_id},
                            )
                        ):
                            yield processed
                    elif parse_result["is_thinking"]:
                        async for processed in emit(
                            next_event(
                                "thinking_delta",
                                content=text,
                                meta={"run_id": run_id},
                            )
                        ):
                            yield processed
                latency.record_ms(
                    "agent_total_ms",
                    (time.perf_counter() - agent_started) * 1000.0,
                )

        except Exception as e:
            logger.error(f"[{request_id}] ❌ 统一事件流生成失败：{e}")
            fallback = "抱歉，当前模型服务暂时不可用。请检查所选模型的 API Key、Base URL 配置，或稍后重试。"
            response_chunks = [fallback]
            async for processed in emit(
                next_event(
                    "error",
                    content=fallback,
                    meta={"error_type": type(e).__name__},
                )
            ):
                yield processed
            self._save_to_memory(
                query,
                fallback,
                use_long_term_memory=use_long_term_memory,
                latency=latency,
            )
            response_saved = True
        else:
            runtime_metrics_after = self._snapshot_runtime_metrics()
            self._record_runtime_metric_deltas(
                latency, runtime_metrics_before, runtime_metrics_after
            )
            final_response = "".join(response_chunks)
            self._save_to_memory(
                query,
                final_response,
                use_long_term_memory=use_long_term_memory,
                latency=latency,
            )
            response_saved = True
            plan_steps = self._update_plan_status(plan_steps, "tools", "done")
            plan_steps = self._update_plan_status(plan_steps, "answer", "done")
            total_duration_sec = round(time.time() - request_started_at, 3)
            tool_success_count = sum(
                1 for item in tool_timeline if item.get("status") == "success"
            )
            tool_error_count = sum(
                1 for item in tool_timeline if item.get("status") == "error"
            )

            reasoning_summary = self._preview_text(
                "".join(thinking_buffer or reasoning_fragments), 300
            )
            if reasoning_summary:
                async for processed in emit(
                    next_event(
                        "reasoning_summary", content={"summary": reasoning_summary}
                    )
                ):
                    yield processed
            async for processed in emit(
                next_event(
                    "step_end",
                    content={"step_id": "tools", "title": "调用工具", "status": "done"},
                )
            ):
                yield processed
            async for processed in emit(
                next_event(
                    "step_start",
                    content={"step_id": "answer", "title": "生成答案"},
                )
            ):
                yield processed
            async for processed in emit(
                next_event(
                    "final_answer",
                    content={
                        "final_answer": final_response,
                        "sources": evidence_items,
                        "tools_used": tool_timeline,
                        "confidence": self._compute_confidence(
                            tool_timeline, memory_hits, evidence_items
                        ),
                        "reasoning_summary": reasoning_summary,
                        "memory_hits": memory_hits,
                        "total_duration_sec": total_duration_sec,
                        "tool_count": len(tool_timeline),
                        "tool_success_count": tool_success_count,
                        "tool_error_count": tool_error_count,
                        "evidence_count": len(evidence_items),
                        "memory_hit_count": len(memory_hits),
                        "request_id": request_id,
                        "latency_metrics": latency.to_payload(),
                    },
                    meta={
                        "request_id": request_id,
                        "total_duration_sec": total_duration_sec,
                        "latency_metrics": latency.to_payload(),
                    },
                )
            ):
                yield processed
            async for processed in emit(
                next_event("latency_metrics", content=latency.to_payload())
            ):
                yield processed
            async for processed in emit(
                next_event(
                    "step_end",
                    content={
                        "step_id": "answer",
                        "title": "生成答案",
                        "status": "done",
                    },
                )
            ):
                yield processed
            async for processed in emit(
                next_event("plan_update", content={"steps": plan_steps})
            ):
                yield processed

            if not thinking_logged and thinking_buffer:
                logger.info(f"[{request_id}] 🔍 Thinking: {''.join(thinking_buffer)}")
            logger.info(
                f"[{request_id}] ✅ 统一事件流完成，响应长度：{len(final_response)} 字符"
            )
        finally:
            if not response_saved and response_chunks:
                self._save_to_memory(
                    query,
                    "".join(response_chunks),
                    use_long_term_memory=use_long_term_memory,
                    latency=latency,
                )
            self._finalize_request(request_id)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class StreamingService(BaseStreamingGenerator):
    def generate_response(self, query: str) -> Generator[str, None, None]:
        logger.info(f"\n=== 处理用户查询：{query} ===")

        tool_call_failed = False
        fallback_used = False

        try:
            agent_input = self._prepare_input(query)
            response = self._agent_executor.invoke(agent_input)
            output = response.get("output", "")
            intermediate_steps = response.get("intermediate_steps", [])

            if self._fallback_service and self._fallback_service.should_use_fallback(
                output
            ):
                logger.warning(
                    "检测到工具调用可能未返回有效结果，尝试从中间步骤生成答案..."
                )
                tool_call_failed = True

                if intermediate_steps:
                    built_response = self._build_response_from_intermediate_steps(
                        query, intermediate_steps
                    )
                    if built_response:
                        output = built_response
                        logger.info("✅ 成功从中间步骤生成答案")
                    else:
                        search_result = self._fallback_service.try_web_search_fallback(
                            query
                        )
                        output = self._fallback_service.format_fallback_response(
                            query, search_result
                        )
                        fallback_used = True
                else:
                    search_result = self._fallback_service.try_web_search_fallback(
                        query
                    )
                    output = self._fallback_service.format_fallback_response(
                        query, search_result
                    )
                    fallback_used = True

            final_response = output

            yield final_response

            self._save_to_memory(query, final_response)

            if fallback_used:
                logger.info(
                    f"✅ 使用联网搜索降级 | 助手响应长度：{len(final_response)} 字符"
                )
            else:
                logger.info(
                    f"✅ 对话已存入记忆 | 助手响应长度：{len(final_response)} 字符"
                )

        except Exception as e:
            logger.error(
                f"❌ 生成响应失败：{type(e).__name__}: {str(e) or '(无错误消息)'}",
                exc_info=True,
            )

            if self._fallback_service and not tool_call_failed:
                logger.warning("检测到异常，尝试使用联网搜索降级...")
                try:
                    search_result = self._fallback_service.try_web_search_fallback(
                        query
                    )
                    fallback_response = self._fallback_service.format_fallback_response(
                        query, search_result
                    )
                    fallback_used = True

                    yield fallback_response

                    self._save_to_memory(query, fallback_response)

                    logger.info(
                        f"✅ 降级搜索成功 | 助手响应长度：{len(fallback_response)} 字符"
                    )
                    return
                except Exception as fallback_error:
                    logger.error(f"降级搜索也失败: {fallback_error}")

            default_response = "抱歉，当前模型服务暂时不可用。请检查所选模型的 API Key、Base URL 配置，或稍后重试。"
            yield default_response
            self._save_to_memory(query, default_response)

    async def generate_response_stream(self, query: str) -> AsyncGenerator[str, None]:
        """返回与历史行为兼容的原始纯文本流。"""
        adapter = PlainTextStreamAdapter()
        async for chunk in self._stream_with_adapter(
            query,
            adapter,
            stop_on_repeated_action=True,
            response_mode="raw_text",
        ):
            yield chunk

    async def generate_events(
        self, query: str, image_path: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """返回前端友好的 JSON 事件流。"""
        adapter = FrontendJsonEventAdapter()
        async for event in self._stream_with_adapter(
            query,
            adapter,
            stop_on_repeated_action=False,
            response_mode="final_answer",
            use_long_term_memory=True,
        ):
            yield event

    async def generate_sse(
        self,
        query: str,
        image_path: Optional[str] = None,
        use_long_term_memory: bool = True,
    ) -> AsyncGenerator[str, None]:
        """返回可直接写入 StreamingResponse 的 SSE 文本流。"""
        adapter = SSEEventAdapter()
        async for chunk in self._stream_with_adapter(
            query,
            adapter,
            stop_on_repeated_action=False,
            response_mode="final_answer",
            use_long_term_memory=use_long_term_memory,
        ):
            yield chunk
