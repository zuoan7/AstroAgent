"""Agent 流式主链路，负责上下文组装、路由决策、执行引擎调用、前端事件适配和记忆写入。
"""

import asyncio
import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from src.agent.governance import (
    AgentExecutionPolicy,
    GovernanceMetricsRegistry,
    RequestObservation,
)
from src.agent.frontend_event_adapter import FrontendExecutionEventAdapter
from src.agent.models.execution_event import ExecutionEvent
from src.agent.models.execution_plan import ExecutionPlan
from src.agent.models.final_response import FinalResponse
from src.agent.output_parser import extract_final_answer_text
from src.agent.policies.budget_policy import BudgetExceededError
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
from src.capabilities.selector import CapabilitySelector
from src.core.errors import ErrorHandler
from src.core.logger import logger
from src.core.mcp_protocol import extract_tool_data, is_tool_error
from src.memory.api.dto import (
    AppendMessageRequest,
    AppendToolCallRequest,
    BuildContextRequest,
)
from src.memory.application.task_state_runtime_service import TaskStateRuntimeService

MAX_ACTION_HISTORY_ENTRIES = 100
TOOL_INPUT_LOG_PREVIEW_CHARS = 300
TOOL_OUTPUT_LOG_PREVIEW_CHARS = 200
FALLBACK_TOOL_OUTPUT_PREVIEW_CHARS = 500


class BaseStreamingGenerator:
    """流式服务基类，封装上下文构建、路由执行、事件生成和记忆落库。"""
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
        execution_policy: Optional[AgentExecutionPolicy] = None,
        governance_metrics: Optional[GovernanceMetricsRegistry] = None,
        audit_logger: Optional[Any] = None,
        agent_executor_factory: Optional[Any] = None,
        execution_engine: Optional[Any] = None,
        task_state_runtime: Optional[Any] = None,
    ):
        """初始化 BaseStreamingGenerator 的依赖、配置和内部状态。"""
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
        self._execution_policy = execution_policy or AgentExecutionPolicy.from_settings()
        self._frontend_event_adapter = FrontendExecutionEventAdapter()
        self._capability_selector = CapabilitySelector()
        self._governance_metrics = governance_metrics
        self._audit_logger = audit_logger
        self._agent_executor_factory = agent_executor_factory
        # ExecutionEngine is the online execution entry; legacy orchestrator
        # injection is accepted only for old callers and is not used here.
        self._execution_engine = execution_engine
        self._task_state_runtime = task_state_runtime or TaskStateRuntimeService(memory)

    def _cleanup_action_history(self, request_id: Optional[str] = None):
        """清理指定请求的工具动作历史并执行 LRU 淘汰。"""
        if request_id and request_id in self._action_history:
            del self._action_history[request_id]
            logger.debug(f"已清理请求 {request_id} 的操作历史")

        while len(self._action_history) > MAX_ACTION_HISTORY_ENTRIES:
            oldest_key, _ = self._action_history.popitem(last=False)
            logger.debug(f"LRU淘汰最旧的操作历史: {oldest_key}")

    def _log_memory_usage(self):
        """记录工具动作历史缓存的内存状态。"""
        total_entries = len(self._action_history)
        total_actions = sum(len(v) for v in self._action_history.values())
        logger.debug(
            f"操作历史内存状态: {total_entries} 个请求, 共 {total_actions} 条动作记录"
        )

    def _build_short_term_memory_context(self, query: str) -> Dict[str, Any]:
        """调用短期记忆服务构建当前查询上下文。"""
        try:
            context = self._memory.build_context(
                BuildContextRequest(
                    session_id=self._memory_session_id(),
                    query=query or self._current_query or "",
                )
            )
            return context if isinstance(context, dict) else {}
        except Exception as e:
            logger.warning(f"build_context失败，使用空历史: {type(e).__name__}: {e}")
            return {}

    def _format_chat_history(
        self,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """把短期记忆上下文格式化为提示词中的历史对话文本。"""
        try:
            context = (
                memory_context
                if memory_context is not None
                else self._build_short_term_memory_context(self._current_query or "")
            )
            context_text = context.get("context_text", "") if context else ""
            return context_text or "无历史对话"
        except Exception as e:
            logger.warning(f"format chat_history失败，使用空历史: {type(e).__name__}: {e}")
            return "无历史对话"

    def _selected_task_state_from_context(
        self,
        memory_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """从记忆上下文中提取选中的任务状态。"""
        if not isinstance(memory_context, dict):
            return {}
        state = memory_context.get("selected_task_state")
        return state if isinstance(state, dict) else {}

    def _build_effective_query(
        self,
        query: str,
        memory_context: Optional[Dict[str, Any]],
    ) -> str:
        """结合任务状态把用户原始问题补全为有效查询。"""
        try:
            return self._task_state_runtime.build_effective_query(
                query,
                self._selected_task_state_from_context(memory_context),
            )
        except Exception as exc:
            logger.warning(
                "build effective query from task_state failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return query

    def _task_state_expected_version(
        self,
        state: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """从任务状态中解析乐观锁版本号。"""
        if not isinstance(state, dict):
            return None
        try:
            return int(state.get("version")) if state.get("version") is not None else None
        except (TypeError, ValueError):
            return None

    def _apply_task_state_turn_started(
        self,
        *,
        session_id: str,
        turn_id: str,
        original_query: str,
        effective_query: str,
        profile: Optional[Any],
        execution_decision: Optional[Any],
        execution_plan: Optional[ExecutionPlan],
        selected_task_state: Optional[Dict[str, Any]],
    ) -> Optional[Any]:
        """在请求开始时写入任务状态补丁。"""
        try:
            patch = self._task_state_runtime.build_turn_started_patch(
                effective_query or original_query,
                profile=profile,
                execution_decision=execution_decision,
                execution_plan=execution_plan,
                selected_task_state=selected_task_state,
            )
            state = self._task_state_runtime.apply_patch_with_retry(
                session_id=session_id,
                patch=patch,
                expected_version=self._task_state_expected_version(selected_task_state),
                turn_id=turn_id,
                created_by="task_state_runtime",
            )
            return state
        except Exception as exc:
            logger.warning(
                "[%s] task_state start update failed: %s: %s",
                turn_id,
                type(exc).__name__,
                exc,
            )
            return None

    def _apply_task_state_turn_completed(
        self,
        *,
        session_id: str,
        turn_id: str,
        response: Optional[FinalResponse] = None,
        profile: Optional[Any] = None,
        execution_decision: Optional[Any] = None,
        error: Optional[BaseException | str] = None,
        fallback_message: str = "",
        expected_version: Optional[int] = None,
    ) -> Optional[Any]:
        """在请求结束时写入任务状态补丁。"""
        try:
            patch = self._task_state_runtime.build_turn_completed_patch(
                response=response,
                profile=profile,
                execution_decision=execution_decision,
                error=error,
                fallback_message=fallback_message,
            )
            return self._task_state_runtime.apply_patch_with_retry(
                session_id=session_id,
                patch=patch,
                expected_version=expected_version,
                turn_id=turn_id,
                created_by="task_state_runtime",
            )
        except Exception as exc:
            logger.warning(
                "[%s] task_state completion update failed: %s: %s",
                turn_id,
                type(exc).__name__,
                exc,
            )
            return None

    def _schedule_task_state_enrichment(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_message: str,
        assistant_message: str,
        current_state: Optional[Any],
    ) -> None:
        """异步调度任务状态 LLM 增强。"""
        try:
            self._task_state_runtime.enrich_patch_with_llm_async(
                session_id=session_id,
                turn_id=turn_id,
                user_message=user_message,
                assistant_message=assistant_message,
                current_state=current_state,
            )
        except Exception as exc:
            logger.warning(
                "[%s] task_state enrichment scheduling failed: %s: %s",
                turn_id,
                type(exc).__name__,
                exc,
            )

    def _get_task_state_debug(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """读取当前任务状态调试快照。"""
        if not hasattr(self._memory, "get_task_state"):
            return None
        try:
            state = self._memory.get_task_state(session_id or self._memory_session_id())
            return state.to_dict() if hasattr(state, "to_dict") else None
        except Exception:
            return None

    @staticmethod
    def _final_response_from_stream_state(
        *,
        answer: str,
        execution_path: str,
        decision: Optional[Any],
        tool_timeline: list[Dict[str, Any]],
        fallback_used: bool,
    ) -> FinalResponse:
        """根据流式执行过程中收集的状态构造 FinalResponse。"""
        task_type = getattr(decision, "task_type", "") if decision else ""
        route = getattr(decision, "route", "") if decision else ""
        trace = []
        for item in tool_timeline or []:
            if not isinstance(item, dict):
                continue
            trace.append(
                {
                    "step_id": str(item.get("run_id") or item.get("tool") or "tool"),
                    "title": str(item.get("tool") or item.get("tool_name") or "tool"),
                    "kind": "tool",
                    "status": item.get("status", "success"),
                    "tool_name": item.get("tool") or item.get("tool_name"),
                    "tool_input": item.get("input", ""),
                    "tool_output_summary": item.get("output_summary", ""),
                    "summary": item.get("output_summary", ""),
                    "duration_sec": item.get("duration_sec"),
                }
            )
        fallback_path = (
            [{"strategy": "stream_fallback", "reason": "fallback_used"}]
            if fallback_used
            else []
        )
        return FinalResponse(
            answer=answer,
            summary=answer[:200] if len(answer) > 200 else answer,
            tools_used=list(tool_timeline or []),
            confidence=0.4,
            route=route,
            task_type=task_type or ("open_domain_reasoning" if execution_path == "react" else ""),
            execution_trace=trace,
            fallback_path=fallback_path,
        )

    def _format_user_profile(self) -> str:
        """把长期记忆画像格式化为提示词上下文。"""
        if not self._long_term_memory:
            return "暂无用户偏好信息"
        if hasattr(self._long_term_memory, "build_prompt_context"):
            return self._long_term_memory.build_prompt_context(
                self._user_id, self._current_query or ""
            )
        if hasattr(self._long_term_memory, "render_profile_prompt"):
            return self._long_term_memory.render_profile_prompt(self._user_id)
        return "暂无用户偏好信息"

    def _extract_and_update_long_term_memory(
        self, user_message: str, assistant_message: str
    ):
        """从本轮对话中抽取长期记忆并写入服务。"""
        if not self._long_term_memory:
            return

        try:
            if hasattr(self._long_term_memory, "extract_and_store_async"):
                self._long_term_memory.extract_and_store_async(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    user_id=self._user_id,
                )
                return
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
        except Exception as e:
            logger.error(f"❌ 更新长期记忆失败: {e}")

    def _schedule_long_term_memory_update(
        self, user_message: str, assistant_message: str
    ) -> None:
        """后台调度长期记忆抽取和更新。"""
        if not self._long_term_memory:
            return

        if hasattr(self._long_term_memory, "extract_and_store_async"):
            self._extract_and_update_long_term_memory(user_message, assistant_message)
            return

        def _runner() -> None:
            """在线程中执行同步长期记忆更新并记录耗时。"""
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
        """检测同一请求中是否重复执行同一工具动作。"""
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
        """从 ReAct 中间工具结果中尽量恢复可用回答。"""
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
        self,
        query: str,
        use_long_term_memory: bool = True,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """准备 legacy ReAct 调用输入，包括问题、历史和用户画像。"""
        self._current_query = query
        chat_history = self._format_chat_history(memory_context)
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
        """为流式前端推断初始计划步骤。"""
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
        """返回更新指定步骤状态后的计划步骤列表。"""
        updated: List[Dict[str, Any]] = []
        for step in steps:
            clone = dict(step)
            if clone["id"] == step_id:
                clone["status"] = status
            updated.append(clone)
        return updated

    def _explain_long_term_retrieval(
        self, query: str, use_long_term_memory: bool
    ) -> List[Dict[str, Any]]:
        """获取长期记忆命中的解释信息。"""
        if not use_long_term_memory or not self._long_term_memory:
            return []
        if hasattr(self._long_term_memory, "explain_retrieval_hits"):
            try:
                hits = self._long_term_memory.explain_retrieval_hits(
                    self._user_id, query
                )
                if isinstance(hits, list):
                    return [hit for hit in hits if isinstance(hit, dict)]
            except Exception as e:
                logger.warning(f"memory explain 失败: {type(e).__name__}: {e}")
        return []

    def _extract_evidence(
        self, tool_name: str, tool_output: str, run_id: str
    ) -> List[Dict[str, Any]]:
        """从工具输出中提取证据片段和外部引用。"""
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
        """根据工具结果、证据和记忆命中估算回答置信度。"""
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
        """把任意值转换为指定长度的预览文本。"""
        text = "" if value is None else str(value)
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _handle_tool_start(
        self, request_id: str, run_id: str, data: dict, check_repeated: bool = False
    ) -> Optional[str]:
        """处理 ReAct 工具开始事件并记录运行状态。"""
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
        """处理 ReAct 工具结束事件、写入记忆并提取图片链接。"""
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
        """写入用户和助手消息，并按需触发长期记忆更新。"""
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
        """清理当前请求的运行状态和动作历史。"""
        self._current_request_id = None
        self._cleanup_action_history(request_id)
        self._log_memory_usage()

    def _memory_session_id(self) -> str:
        """计算当前短期记忆会话 ID。"""
        session_id = getattr(self._memory, "session_id", None)
        if session_id:
            return session_id
        return f"mem_{self._user_id}"

    def _ensure_agent_executor(self) -> Any:
        """获取或创建 legacy ReAct AgentExecutor。"""
        if self._agent_executor is not None:
            return self._agent_executor
        if self._agent_executor_factory is None:
            raise ValueError("react agent executor is not configured")
        self._agent_executor = self._agent_executor_factory()
        return self._agent_executor

    def _use_unified_execution_engine(self) -> bool:
        """判断当前流式服务是否配置了统一执行引擎。"""
        return getattr(self, "_execution_engine", None) is not None

    def _preview_execution_plan_for_streaming(
        self,
        execution_decision: Any,
        decision: Any,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
        exec_context: Optional[Any] = None,
    ) -> Optional[ExecutionPlan]:
        """Return the planned-task preview used by legacy frontend events.

        主展示路径使用 `ExecutionEngine.preview_plan()`；legacy orchestrator
        不再作为在线展示 fallback。
        """
        if not self._use_unified_execution_engine():
            return None

        if hasattr(self._execution_engine, "preview_plan_context") and exec_context:
            return self._execution_engine.preview_plan_context(
                execution_decision,
                exec_context,
            )
        return self._execution_engine.preview_plan(
            execution_decision,
            decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
        )

    def _build_request_context(
        self,
        query: str,
        *,
        use_long_term_memory: bool,
        request_id: Optional[str] = None,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """构造统一 RequestContext 并写入当前查询。"""
        from src.agent.models.request_context import RequestContext

        self._current_query = query
        return RequestContext(
            query=query,
            chat_history=self._format_chat_history(memory_context),
            user_profile=(
                self._format_user_profile()
                if use_long_term_memory
                else "本轮对话已禁用长期记忆"
            ),
            request_id=request_id or uuid.uuid4().hex[:8],
            use_long_term_memory=use_long_term_memory,
        )

    def _resolve_execution_decision(
        self,
        query: str,
        legacy_decision: Optional[Any],
        *,
        use_long_term_memory: bool,
        chat_history: str = "",
        user_profile: str = "",
        precomputed_profile: Optional[Any] = None,
    ) -> tuple[Any, Optional[Any], Optional[Any]]:
        """优先走 TaskProfile/ExecutionContext -> ExecutionDecision，保留 legacy 回退。

        历史 profile/context/decision feature flags 已退场，不再切换此主流程。
        """
        from src.agent.models.execution_context import ExecutionContext
        from src.agent.models.request_context import RequestContext
        from src.agent.models.task_profile import TaskProfile

        policy = getattr(self, "_execution_policy", AgentExecutionPolicy.from_settings())

        router = getattr(self, "_request_router", None)

        if precomputed_profile is not None:
            profile = precomputed_profile
        elif router and hasattr(router, "profile"):
            profile = router.profile(query)
        elif legacy_decision and isinstance(getattr(legacy_decision, "route", None), str):
            profile = TaskProfile.from_legacy_route(
                route=legacy_decision.route,
                task_type=getattr(legacy_decision, "task_type", "open_domain_reasoning"),
                confidence=getattr(legacy_decision, "confidence", 0.0),
                matched_skills=getattr(legacy_decision, "matched_skills", []),
                capability_hints=getattr(legacy_decision, "capability_hints", None),
                reason=getattr(legacy_decision, "reason", ""),
                expected_output_schema=getattr(
                    legacy_decision, "expected_output_schema", "generic_answer_v1"
                ),
            )
        else:
            # 即使缺少 router/legacy route，也构造兼容画像并统一走 decide()。
            effective_mode = getattr(policy, "effective_mode", getattr(policy, "mode", "hybrid"))
            if effective_mode == "react":
                fallback_route = "fallback_react"
                fallback_task_type = "open_domain_reasoning"
            elif effective_mode == "planned" or policy.enable_planner:
                fallback_route = "planned_task"
                fallback_task_type = "observation_recommendation"
            elif policy.enable_react_fallback:
                fallback_route = "fallback_react"
                fallback_task_type = "open_domain_reasoning"
            else:
                fallback_route = "direct_task"
                fallback_task_type = "smalltalk"

            profile = TaskProfile.from_legacy_route(
                route=fallback_route,
                task_type=fallback_task_type,
                confidence=0.0,
                reason="compat_profile_without_router_or_legacy_decision",
            )

        context = ExecutionContext(
            profile=profile,
            request=RequestContext(
                query=query,
                chat_history=chat_history,
                user_profile=user_profile,
                use_long_term_memory=use_long_term_memory,
            ),
        )
        execution_decision = policy.decide(profile, context)
        capability_selector = getattr(self, "_capability_selector", None)
        if capability_selector is None:
            capability_selector = CapabilitySelector()
            self._capability_selector = capability_selector
        context.capability_decision = capability_selector.select(
            profile=profile,
            execution_decision=execution_decision,
            query=query,
        )
        return execution_decision, profile, context

    def _resolve_legacy_route_decision(
        self,
        query: str,
        *,
        precomputed_profile: Optional[Any] = None,
    ) -> Optional[Any]:
        """Derive legacy metadata only from an already resolved TaskProfile."""
        if precomputed_profile is not None and hasattr(
            precomputed_profile, "to_legacy_route_decision"
        ):
            return precomputed_profile.to_legacy_route_decision()
        return None

    def _build_frontend_plan_steps(
        self,
        *,
        query: str,
        use_long_term_memory: bool,
        execution_plan: ExecutionPlan,
    ) -> List[Dict[str, Any]]:
        """把 ExecutionPlan 转换为前端计划步骤列表。"""
        steps = [
            {
                "id": "understand",
                "title": "解析任务",
                "description": f"理解用户问题并构建执行路径: {self._preview_text(query, 80)}",
                "status": "done",
            }
        ]
        if use_long_term_memory:
            steps.append(
                {
                    "id": "memory",
                    "title": "读取记忆",
                    "description": "检索与当前问题相关的长期记忆与用户偏好",
                    "status": "done",
                }
            )
        steps.extend(execution_plan.to_frontend_steps())
        steps.append(
            {
                "id": "answer",
                "title": "生成答案",
                "description": "汇总执行结果并生成最终回答",
                "status": "pending",
            }
        )
        return steps

    @staticmethod
    def _infer_content_type(text: str) -> str:
        """根据文本内容推断工具输出的内容类型。"""
        stripped = (text or "").strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(stripped)
                return "application/json"
            except Exception:
                return "text/plain"
        return "text/plain"

    def _extract_stream_text(self, data: dict) -> Optional[str]:
        """从 LangChain 流式事件 chunk 中提取文本。"""
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
        """解析 ReAct 流中的思考片段和最终答案片段。"""
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
            elif extract_final_answer_text(combined_thinking):
                result["final_answer_started"] = True
                if not thinking_logged and combined_thinking:
                    result["thinking_logged"] = True
                    logger.info(f"[{request_id}] 🔍 Thinking: {combined_thinking}")
                final_answer_text = extract_final_answer_text(combined_thinking)
                if final_answer_text and not final_answer_extracted:
                    result["final_answer_extracted"] = True
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
        """注册一个流式事件处理器。"""
        self._event_processors.append(processor)

    def clear_event_processors(self):
        """清空所有流式事件处理器。"""
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
        """构造并校验内部标准流式事件。"""
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
        """读取 MCP 和 RAG 当前运行时指标。"""
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
        """计算单个运行时指标的前后差值。"""
        return round(after.get(key, 0.0) - before.get(key, 0.0), 2)

    def _record_runtime_metric_deltas(
        self,
        latency: LatencyTracker,
        before: Dict[str, Dict[str, float]],
        after: Dict[str, Dict[str, float]],
    ) -> None:
        """把 MCP/RAG 指标差值写入延迟追踪器。"""
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

    async def _run_orchestrated_path(
        self,
        query: str,
        decision: Any,
        *,
        use_long_term_memory: bool,
        latency: LatencyTracker,
        execution_plan: Optional[ExecutionPlan] = None,
        event_callback: Optional[Any] = None,
        chat_history: Optional[str] = None,
        user_profile: Optional[str] = None,
        execution_decision: Optional[Any] = None,
        exec_context: Optional[Any] = None,
    ) -> FinalResponse:
        """调用统一执行引擎完成 direct/planned/react 编排路径。"""
        with latency.measure("agent_prepare_ms"):
            if chat_history is None:
                chat_history = self._format_chat_history()
            self._current_query = query
            if user_profile is None:
                user_profile = (
                    self._format_user_profile()
                    if use_long_term_memory
                    else "本轮对话已禁用长期记忆"
                )

        if not self._use_unified_execution_engine():
            raise ValueError("execution engine is not configured")

        exec_decision = execution_decision
        if exec_decision is None or exec_context is None:
            exec_decision, _, exec_context = self._resolve_execution_decision(
                query,
                decision,
                use_long_term_memory=use_long_term_memory,
                chat_history=chat_history,
                user_profile=user_profile,
            )

        with latency.measure("agent_total_ms"):
            if hasattr(self._execution_engine, "run_context") and exec_context:
                result = await self._execution_engine.run_context(
                    exec_decision,
                    exec_context,
                    execution_plan=execution_plan,
                    event_callback=event_callback,
                )
            else:
                result = await self._execution_engine.run(
                    exec_decision,
                    decision,
                    query,
                    chat_history=chat_history,
                    user_profile=user_profile,
                    execution_plan=execution_plan,
                    event_callback=event_callback,
                    context=exec_context,
                )
        latency.set_meta("execution_engine", "unified")
        latency.set_meta("exec_mode", exec_decision.mode)

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
        """把内部事件流通过指定适配器转换为对外输出。"""
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
        """生成完整内部流式事件，串联记忆、路由、执行、审计和落库。"""
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
        memory_hits = self._explain_long_term_retrieval(query, use_long_term_memory)
        tool_timeline: List[Dict[str, Any]] = []
        evidence_items: List[Dict[str, Any]] = []
        reasoning_fragments: List[str] = []
        runtime_metrics_before = self._snapshot_runtime_metrics()
        actual_plan: Optional[ExecutionPlan] = None
        final_resp_obj: Optional[FinalResponse] = None
        decision = None
        profile = None
        execution_decision = None
        execution_path = "unknown"
        fallback_used = False
        selected_task_state: Dict[str, Any] = {}
        task_state_started = None
        task_state_completed = None
        completion_state_written = False
        task_state_debug: Optional[Dict[str, Any]] = None
        output_schema_parse_success = response_mode != "final_answer"

        def next_event(
            event_type: str,
            *,
            content: Any = None,
            meta: Optional[Dict[str, Any]] = None,
            modality: str = "text",
        ) -> StreamEvent:
            """为当前请求创建带递增序号的内部事件。"""
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
            """把内部事件交给处理器链并逐个产出。"""
            processed_events = apply_event_processors(event, self._event_processors)
            for processed in processed_events:
                yield processed

        try:
            memory_context = self._build_short_term_memory_context(query)
            selected_task_state = self._selected_task_state_from_context(memory_context)
            effective_query = self._build_effective_query(query, memory_context)
            if effective_query != query:
                latency.set_meta("effective_query_used", True)
                latency.set_meta("effective_query_preview", self._preview_text(effective_query, 240))

            request_context = self._build_request_context(
                effective_query,
                use_long_term_memory=use_long_term_memory,
                request_id=request_id,
                memory_context=memory_context,
            )

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

            precomputed_profile = None
            router = getattr(self, "_request_router", None)
            if router and hasattr(router, "profile"):
                with latency.measure("route_decision_ms"):
                    precomputed_profile = router.profile(effective_query)
                    decision = self._resolve_legacy_route_decision(
                        effective_query,
                        precomputed_profile=precomputed_profile,
                    )

            execution_decision, profile, _exec_context = self._resolve_execution_decision(
                effective_query,
                decision,
                use_long_term_memory=use_long_term_memory,
                chat_history=request_context.chat_history,
                user_profile=request_context.user_profile,
                precomputed_profile=precomputed_profile,
            )
            execution_path = execution_decision.mode
            fallback_used = False
            output_schema_parse_success = response_mode != "final_answer"
            if decision is None:
                decision = self._resolve_legacy_route_decision(
                    effective_query,
                    precomputed_profile=profile,
                )

            if decision is not None:
                execution_path = execution_decision.mode
                latency.set_meta("agent_mode", self._execution_policy.mode)
                latency.set_meta("execution_path", execution_path)
                latency.set_meta(
                    "execution_decision",
                    execution_decision.to_dict(),
                )
                latency.set_meta(
                    "feature_flags",
                    self._execution_policy.to_dict(),
                )
                async for processed in self._frontend_event_adapter.emit_execution_event(
                    ExecutionEvent(
                        type="route_decided",
                        payload=decision.to_meta(),
                        source="router",
                    ),
                    next_event_fn=next_event,
                    emit_fn=emit,
                ):
                    yield processed

            if execution_path == "planned" and decision:
                actual_plan = self._preview_execution_plan_for_streaming(
                    execution_decision,
                    decision,
                    effective_query,
                    chat_history=request_context.chat_history,
                    user_profile=request_context.user_profile,
                    exec_context=_exec_context,
                )
                if actual_plan is not None:
                    plan_steps = self._build_frontend_plan_steps(
                        query=query,
                        use_long_term_memory=use_long_term_memory,
                        execution_plan=actual_plan,
                    )
                    async for processed in self._frontend_event_adapter.emit_execution_event(
                        ExecutionEvent(
                            type="plan_built",
                            payload={"steps": plan_steps},
                            source="planned",
                        ),
                        next_event_fn=next_event,
                        emit_fn=emit,
                    ):
                        yield processed

            task_state_started = self._apply_task_state_turn_started(
                session_id=self._memory_session_id(),
                turn_id=request_id,
                original_query=query,
                effective_query=effective_query,
                profile=profile,
                execution_decision=execution_decision,
                execution_plan=actual_plan,
                selected_task_state=selected_task_state,
            )

            if execution_path != "planned":
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

            if execution_path in ("direct", "planned") and decision:
                final_resp: FinalResponse = await self._run_orchestrated_path(
                    effective_query,
                    decision,
                    use_long_term_memory=use_long_term_memory,
                    latency=latency,
                    execution_plan=actual_plan,
                    event_callback=None,
                    chat_history=request_context.chat_history,
                    user_profile=request_context.user_profile,
                    execution_decision=execution_decision,
                    exec_context=_exec_context,
                )
                final_resp_obj = final_resp
                output_schema_parse_success = True
                direct_answer = final_resp.answer
                response_chunks.append(direct_answer)
                if final_resp.execution_events:
                    async for processed in self._frontend_event_adapter.emit_response_execution_events(
                        final_resp,
                        plan_steps=plan_steps,
                        evidence_items=evidence_items,
                        tool_timeline=tool_timeline,
                        next_event_fn=next_event,
                        emit_fn=emit,
                    ):
                        yield processed
                elif execution_path == "planned":
                    for trace in final_resp.execution_trace:
                        async for processed in self._frontend_event_adapter.emit_trace_events(
                            trace,
                            plan_steps=plan_steps,
                            evidence_items=evidence_items,
                            tool_timeline=tool_timeline,
                            next_event_fn=next_event,
                            emit_fn=emit,
                            preview_text_fn=self._preview_text,
                        ):
                            yield processed
                else:
                    tool_timeline.extend(final_resp.tools_used)
                for source in final_resp.sources:
                    if source not in evidence_items:
                        evidence_items.append(source)
                        async for processed in emit(
                            next_event("evidence_found", content=source)
                        ):
                            yield processed
                async for processed in emit(
                    next_event("final_answer_delta", content=direct_answer)
                ):
                    yield processed
            else:
                if execution_path == "planned":
                    latency.set_meta("planned_path_ready", False)
                agent_input = {
                    "input": request_context.query,
                    "chat_history": request_context.chat_history,
                    "user_profile": request_context.user_profile,
                }
                agent_started = time.perf_counter()
                if execution_path == "react" and decision:
                    if not self._use_unified_execution_engine():
                        raise ValueError("execution engine is not configured")
                    _, _, exec_context = self._resolve_execution_decision(
                        effective_query,
                        decision,
                        use_long_term_memory=use_long_term_memory,
                        chat_history=agent_input.get("chat_history", ""),
                        user_profile=agent_input.get("user_profile", ""),
                    )
                    if hasattr(self._execution_engine, "astream_events_context"):
                        raw_events = self._execution_engine.astream_events_context(
                            execution_decision,
                            exec_context,
                            version="v1",
                        )
                    else:
                        raw_events = self._execution_engine.astream_events(
                            execution_decision,
                            decision,
                            effective_query,
                            chat_history=agent_input.get("chat_history", ""),
                            user_profile=agent_input.get("user_profile", ""),
                            version="v1",
                            context=exec_context,
                        )
                else:
                    raise ValueError(
                        f"unsupported streaming execution path: {execution_path!r}"
                    )

                async for raw_event in raw_events:
                    event_type = raw_event.get("event")
                    data = raw_event.get("data", {}) or {}
                    run_id = raw_event.get("run_id")
                    event_name = raw_event.get("name")

                    if event_type == "on_tool_start":
                        tool_name = (
                            data.get("name")
                            or data.get("tool")
                            or event_name
                            or "unknown_tool"
                        )
                        tool_input = (
                            "" if data.get("input") is None else str(data.get("input"))
                        )
                        tool_data = {**data, "name": tool_name, "input": tool_input}
                        async for processed in self._frontend_event_adapter.emit_execution_event(
                            ExecutionEvent(
                                type="tool_called",
                                payload={
                                    "run_id": run_id,
                                    "tool": tool_name,
                                    "input": tool_input,
                                    "status": "running",
                                },
                                source="react",
                            ),
                            next_event_fn=lambda event_type, **kwargs: next_event(
                                event_type,
                                meta={
                                    "run_id": run_id,
                                    "tool": tool_name,
                                },
                                **kwargs,
                            ),
                            emit_fn=emit,
                        ):
                            yield processed
                        result = self._handle_tool_start(
                            request_id,
                            run_id,
                            tool_data,
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
                        tool_name = (
                            tool_meta.get("name")
                            or event_name
                            or data.get("name")
                            or data.get("tool")
                            or "unknown_tool"
                        )
                        async for processed in self._frontend_event_adapter.emit_execution_event(
                            ExecutionEvent(
                                type="tool_returned",
                                payload={
                                    "run_id": run_id,
                                    "tool": tool_name,
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
                                source="react",
                            ),
                            next_event_fn=lambda event_type, **kwargs: next_event(
                                event_type,
                                meta={
                                    "run_id": run_id,
                                    "tool": tool_name,
                                    "duration_sec": tool_result.get("duration"),
                                    "extracted_url": tool_result.get("extracted_url"),
                                },
                                **kwargs,
                            ),
                            emit_fn=emit,
                        ):
                            yield processed
                        tool_timeline.append(
                            {
                                "run_id": run_id,
                                "tool": tool_name,
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
                            output_schema_parse_success = True
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
                        output_schema_parse_success = True
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

        except BudgetExceededError as e:
            logger.error(f"[{request_id}] ❌ 预算限制触发：{e}")
            fallback = f"当前请求超出执行预算，已提前停止：{e}"
            response_chunks = [fallback]
            fallback_used = True
            async for processed in emit(
                next_event(
                    "error",
                    content=fallback,
                    meta={"error_type": type(e).__name__, "fallback_strategy": "partial_answer"},
                )
            ):
                yield processed
            task_state_completed = self._apply_task_state_turn_completed(
                session_id=self._memory_session_id(),
                turn_id=request_id,
                response=final_resp_obj,
                profile=profile,
                execution_decision=execution_decision,
                error=e,
                fallback_message=fallback,
                expected_version=getattr(task_state_started, "version", None),
            )
            completion_state_written = task_state_completed is not None
            self._save_to_memory(
                query,
                fallback,
                use_long_term_memory=use_long_term_memory,
                latency=latency,
            )
            response_saved = True
        except Exception as e:
            logger.error(f"[{request_id}] ❌ 统一事件流生成失败：{e}")
            fallback = "抱歉，当前模型服务暂时不可用。请检查所选模型的 API Key、Base URL 配置，或稍后重试。"
            response_chunks = [fallback]
            fallback_used = True
            async for processed in emit(
                next_event(
                    "error",
                    content=fallback,
                    meta={"error_type": type(e).__name__},
                )
            ):
                yield processed
            task_state_completed = self._apply_task_state_turn_completed(
                session_id=self._memory_session_id(),
                turn_id=request_id,
                response=final_resp_obj,
                profile=profile,
                execution_decision=execution_decision,
                error=e,
                fallback_message=fallback,
                expected_version=getattr(task_state_started, "version", None),
            )
            completion_state_written = task_state_completed is not None
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
            if response_mode == "final_answer" and execution_path == "react":
                if not final_response.strip():
                    recovered_answer = extract_final_answer_text(
                        "".join(thinking_buffer or reasoning_fragments)
                    )
                    if recovered_answer:
                        final_response = recovered_answer
                        response_chunks = [recovered_answer]
                        output_schema_parse_success = True
                        async for processed in emit(
                            next_event("final_answer_delta", content=recovered_answer)
                        ):
                            yield processed
                output_schema_parse_success = bool(final_response.strip()) and (
                    output_schema_parse_success or final_answer_extracted
                )
            completion_response = final_resp_obj or self._final_response_from_stream_state(
                answer=final_response,
                execution_path=execution_path,
                decision=decision,
                tool_timeline=tool_timeline,
                fallback_used=fallback_used,
            )
            task_state_completed = self._apply_task_state_turn_completed(
                session_id=self._memory_session_id(),
                turn_id=request_id,
                response=completion_response,
                profile=profile,
                execution_decision=execution_decision,
                expected_version=getattr(task_state_started, "version", None),
            )
            completion_state_written = task_state_completed is not None
            task_state_debug = (
                task_state_completed.to_dict()
                if hasattr(task_state_completed, "to_dict")
                else self._get_task_state_debug()
            )
            self._save_to_memory(
                query,
                final_response,
                use_long_term_memory=use_long_term_memory,
                latency=latency,
            )
            response_saved = True
            if completion_state_written:
                self._schedule_task_state_enrichment(
                    session_id=self._memory_session_id(),
                    turn_id=request_id,
                    user_message=query,
                    assistant_message=final_response,
                    current_state=task_state_debug,
                )
            if execution_path != "planned":
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
            if execution_path != "planned":
                async for processed in emit(
                    next_event(
                        "step_end",
                        content={"step_id": "tools", "title": "调用工具", "status": "done"},
                    )
                ):
                    yield processed
            audit_metadata = (
                final_resp_obj.audit_metadata
                if final_resp_obj and final_resp_obj.audit_metadata
                else {}
            )
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
                        "task_state": task_state_debug or self._get_task_state_debug(),
                        "total_duration_sec": total_duration_sec,
                        "tool_count": len(tool_timeline),
                        "tool_success_count": tool_success_count,
                        "tool_error_count": tool_error_count,
                        "evidence_count": len(evidence_items),
                        "memory_hit_count": len(memory_hits),
                        "request_id": request_id,
                        "latency_metrics": latency.to_payload(),
                        "fallback_path": (
                            final_resp_obj.fallback_path if final_resp_obj else []
                        ),
                        "route_decision": (
                            final_resp_obj.route_decision if final_resp_obj else None
                        ),
                        "audit_metadata": audit_metadata,
                        "router_source": audit_metadata.get("router_source"),
                        "rule_confidence": audit_metadata.get("rule_confidence"),
                        "llm_confidence": audit_metadata.get("llm_confidence"),
                        "tool_necessity_action": audit_metadata.get(
                            "tool_necessity_action"
                        ),
                        "tool_necessity_reason": audit_metadata.get(
                            "tool_necessity_reason"
                        ),
                        "tool_necessity_confidence": audit_metadata.get(
                            "tool_necessity_confidence"
                        ),
                        "planner_source": audit_metadata.get("planner_source"),
                        "plan_steps_with_params": audit_metadata.get(
                            "plan_steps_with_params", []
                        ),
                        "param_builder_source": audit_metadata.get(
                            "param_builder_source"
                        ),
                        "handler_mcp_tools_used": audit_metadata.get(
                            "handler_mcp_tools_used", []
                        ),
                        "budget_usage": (
                            final_resp_obj.budget_usage if final_resp_obj else None
                        ),
                        "versions": final_resp_obj.versions if final_resp_obj else None,
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
            if self._audit_logger and getattr(self._audit_logger, "enabled", False):
                try:
                    self._audit_logger.append(
                        {
                            "request_id": request_id,
                            "query": query,
                            "effective_query": locals().get("effective_query", query),
                            "route_decision": (
                                decision.to_meta()
                                if decision and hasattr(decision, "to_meta")
                                else None
                            ),
                            "plan": (
                                actual_plan.to_dict()
                                if actual_plan is not None
                                else (
                                    final_resp_obj.execution_plan
                                    if final_resp_obj
                                    else None
                                )
                            ),
                            "step_results": (
                                final_resp_obj.execution_trace if final_resp_obj else []
                            ),
                            "final_response": (
                                final_resp_obj.to_dict()
                                if final_resp_obj
                                else {"answer": "".join(response_chunks)}
                            ),
                            "latency_profile": latency.to_payload(),
                            "fallback_path": (
                                final_resp_obj.fallback_path if final_resp_obj else []
                            ),
                            "task_state": task_state_debug or self._get_task_state_debug(),
                        }
                    )
                except Exception as audit_error:
                    logger.warning(f"[{request_id}] audit append failed: {audit_error}")
            if self._governance_metrics:
                snapshot = latency.stages_ms()
                self._governance_metrics.record(
                    RequestObservation(
                        route=getattr(decision, "route", "unknown"),
                        request_total_ms=snapshot.get("request_total_ms", 0.0),
                        agent_mode=self._execution_policy.mode,
                        execution_path=execution_path,
                        fallback_used=fallback_used,
                        output_schema_parse_success=output_schema_parse_success,
                    )
                )
            if not response_saved and response_chunks:
                final_text = "".join(response_chunks)
                if not completion_state_written:
                    completion_response = final_resp_obj or self._final_response_from_stream_state(
                        answer=final_text,
                        execution_path=execution_path,
                        decision=decision,
                        tool_timeline=tool_timeline,
                        fallback_used=fallback_used,
                    )
                    task_state_completed = self._apply_task_state_turn_completed(
                        session_id=self._memory_session_id(),
                        turn_id=request_id,
                        response=completion_response,
                        profile=profile,
                        execution_decision=execution_decision,
                        expected_version=getattr(task_state_started, "version", None),
                    )
                    completion_state_written = task_state_completed is not None
                    task_state_debug = (
                        task_state_completed.to_dict()
                        if hasattr(task_state_completed, "to_dict")
                        else self._get_task_state_debug()
                    )
                self._save_to_memory(
                    query,
                    final_text,
                    use_long_term_memory=use_long_term_memory,
                    latency=latency,
                )
                if completion_state_written:
                    self._schedule_task_state_enrichment(
                        session_id=self._memory_session_id(),
                        turn_id=request_id,
                        user_message=query,
                        assistant_message=final_text,
                        current_state=task_state_debug,
                    )
            self._finalize_request(request_id)


class _nullcontext:
    """无操作上下文管理器，用于统一可选延迟测量分支。"""
    def __enter__(self):
        """进入无操作上下文。"""
        return None

    def __exit__(self, exc_type, exc, tb):
        """退出无操作上下文且不吞掉异常。"""
        return False


class StreamingService(BaseStreamingGenerator):
    """对外流式服务门面，提供纯文本、JSON 事件和 SSE 三种输出形式。"""
    def generate_response(self, query: str) -> Generator[str, None, None]:
        """同步生成完整文本响应。"""
        logger.info(f"\n=== 处理用户查询：{query} ===")

        tool_call_failed = False
        fallback_used = False

        try:
            request_id = uuid.uuid4().hex[:8]
            self._current_request_id = request_id
            memory_context = self._build_short_term_memory_context(query)
            selected_task_state = self._selected_task_state_from_context(memory_context)
            effective_query = self._build_effective_query(query, memory_context)
            precomputed_profile = None
            decision = None
            router = getattr(self, "_request_router", None)
            if router and hasattr(router, "profile"):
                precomputed_profile = router.profile(effective_query)
                decision = self._resolve_legacy_route_decision(
                    effective_query,
                    precomputed_profile=precomputed_profile,
                )

            execution_decision, profile, exec_context = self._resolve_execution_decision(
                effective_query,
                decision,
                use_long_term_memory=True,
                chat_history=self._format_chat_history(memory_context),
                user_profile=self._format_user_profile(),
                precomputed_profile=precomputed_profile,
            )
            if decision is None:
                decision = self._resolve_legacy_route_decision(
                    effective_query,
                    precomputed_profile=profile,
                )
            execution_path = execution_decision.mode
            task_state_started = self._apply_task_state_turn_started(
                session_id=self._memory_session_id(),
                turn_id=request_id,
                original_query=query,
                effective_query=effective_query,
                profile=profile,
                execution_decision=execution_decision,
                execution_plan=None,
                selected_task_state=selected_task_state,
            )
            if decision and (
                execution_path in ("direct", "planned")
                or execution_path == "react"
            ):
                final_resp = asyncio.run(
                    self._run_orchestrated_path(
                        effective_query,
                        decision,
                        use_long_term_memory=True,
                        latency=LatencyTracker(),
                        chat_history=self._format_chat_history(memory_context),
                        user_profile=self._format_user_profile(),
                        execution_decision=execution_decision,
                        exec_context=exec_context,
                    )
                )
                output = final_resp.answer
                intermediate_steps = []
            else:
                final_resp = None
                agent_input = self._prepare_input(effective_query, memory_context=memory_context)
                agent_executor = self._ensure_agent_executor()
                response = agent_executor.invoke(agent_input)
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

            completion_response = final_resp or self._final_response_from_stream_state(
                answer=final_response,
                execution_path=execution_path,
                decision=decision,
                tool_timeline=[],
                fallback_used=fallback_used,
            )
            task_state_completed = self._apply_task_state_turn_completed(
                session_id=self._memory_session_id(),
                turn_id=request_id,
                response=completion_response,
                profile=profile,
                execution_decision=execution_decision,
                expected_version=getattr(task_state_started, "version", None),
            )
            self._save_to_memory(query, final_response)
            if task_state_completed is not None:
                self._schedule_task_state_enrichment(
                    session_id=self._memory_session_id(),
                    turn_id=request_id,
                    user_message=query,
                    assistant_message=final_response,
                    current_state=task_state_completed,
                )

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

                    task_state_completed = self._apply_task_state_turn_completed(
                        session_id=self._memory_session_id(),
                        turn_id=self._current_request_id or uuid.uuid4().hex[:8],
                        response=locals().get("final_resp"),
                        profile=locals().get("profile"),
                        execution_decision=locals().get("execution_decision"),
                        error=e,
                        fallback_message=fallback_response,
                        expected_version=getattr(locals().get("task_state_started"), "version", None),
                    )
                    self._save_to_memory(query, fallback_response)
                    if task_state_completed is not None:
                        self._schedule_task_state_enrichment(
                            session_id=self._memory_session_id(),
                            turn_id=self._current_request_id or "",
                            user_message=query,
                            assistant_message=fallback_response,
                            current_state=task_state_completed,
                        )

                    logger.info(
                        f"✅ 降级搜索成功 | 助手响应长度：{len(fallback_response)} 字符"
                    )
                    return
                except Exception as fallback_error:
                    logger.error(f"降级搜索也失败: {fallback_error}")

            default_response = "抱歉，当前模型服务暂时不可用。请检查所选模型的 API Key、Base URL 配置，或稍后重试。"
            yield default_response
            self._apply_task_state_turn_completed(
                session_id=self._memory_session_id(),
                turn_id=self._current_request_id or uuid.uuid4().hex[:8],
                response=locals().get("final_resp"),
                profile=locals().get("profile"),
                execution_decision=locals().get("execution_decision"),
                error=e,
                fallback_message=default_response,
                expected_version=getattr(locals().get("task_state_started"), "version", None),
            )
            self._save_to_memory(query, default_response)

    async def generate_response_stream(self, query: str) -> AsyncGenerator[str, None]:
        """返回清洗后的纯文本答案流。"""
        adapter = PlainTextStreamAdapter()
        async for chunk in self._stream_with_adapter(
            query,
            adapter,
            stop_on_repeated_action=True,
            response_mode="final_answer",
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
