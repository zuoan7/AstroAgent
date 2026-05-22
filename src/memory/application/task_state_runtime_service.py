from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Dict, Iterable, Optional

from src.agent.prompts import get_prompt_renderer
from src.core.config import settings
from src.core.logger import logger
from src.memory.domain.events import MemoryEventType
from src.memory.domain.task_state import TaskState, TaskStateConflictError


ALLOWED_ENRICHMENT_FIELDS = {
    "current_goal",
    "active_constraints",
    "open_questions",
    "assumptions",
    "next_action",
    "confidence",
}

FOLLOW_UP_CUES = (
    "继续",
    "下一步",
    "接着",
    "然后呢",
    "刚才",
    "上面",
    "它",
    "这个",
    "那个",
    "这一步",
    "后续",
)


class TaskStateRuntimeService:
    """Build and apply deterministic task-state patches for request turns."""

    def __init__(self, memory: Any, llm: Optional[Any] = None) -> None:
        self._memory = memory
        self._llm = llm

    def build_effective_query(
        self,
        query: str,
        selected_task_state: Optional[Dict[str, Any] | TaskState],
    ) -> str:
        """Augment ambiguous follow-up queries with the current task projection."""

        state = self._state_to_dict(selected_task_state)
        if not self._is_follow_up_query(query) or not self._has_meaningful_state(state):
            return query

        context_parts: list[str] = []
        if state.get("current_goal"):
            context_parts.append(f"当前任务目标: {state['current_goal']}")
        if state.get("next_action"):
            context_parts.append(f"下一步动作: {state['next_action']}")
        if state.get("active_constraints"):
            context_parts.append(
                "约束: " + "; ".join(str(x) for x in state["active_constraints"][:5])
            )
        if state.get("pending_steps"):
            context_parts.append(
                "待办步骤: " + "; ".join(str(x) for x in state["pending_steps"][:5])
            )

        if not context_parts:
            return query
        return f"{query}\n\n内部任务上下文：{'；'.join(context_parts)}"

    def build_turn_started_patch(
        self,
        query: str,
        *,
        profile: Optional[Any] = None,
        execution_decision: Optional[Any] = None,
        execution_plan: Optional[Any] = None,
        selected_task_state: Optional[Dict[str, Any] | TaskState] = None,
    ) -> Dict[str, Any]:
        if self._is_smalltalk(profile=profile):
            return {}

        state = self._state_to_dict(selected_task_state)
        pending_steps = self._pending_steps_from_plan(execution_plan)
        if not pending_steps:
            pending_steps = self._pending_steps_from_profile(
                profile=profile,
                execution_decision=execution_decision,
            )

        goal = self._derive_goal(query, profile=profile, selected_task_state=state)
        next_action = self._derive_start_next_action(
            pending_steps,
            profile=profile,
            execution_decision=execution_decision,
        )

        patch: Dict[str, Any] = {
            "status": "running",
            "current_goal": goal,
            "pending_steps": pending_steps,
            "next_action": next_action,
        }
        confidence = self._confidence_from_profile(profile)
        if confidence is not None:
            patch["confidence"] = confidence
        return patch

    def build_turn_completed_patch(
        self,
        *,
        response: Optional[Any] = None,
        profile: Optional[Any] = None,
        execution_decision: Optional[Any] = None,
        error: Optional[BaseException | str] = None,
        fallback_message: str = "",
    ) -> Dict[str, Any]:
        if self._is_smalltalk(response=response, profile=profile):
            return {}

        trace = list(getattr(response, "execution_trace", []) or [])
        completed_steps = self._completed_steps_from_trace(trace)
        failed_steps, blockers = self._failed_steps_from_trace(trace)

        if not completed_steps:
            completed_steps = self._completed_steps_from_tools(
                getattr(response, "tools_used", []) if response else []
            )
        if not completed_steps and response is not None and getattr(response, "answer", ""):
            completed_steps = ["生成回答"]

        fallback_blockers = self._blockers_from_fallback_path(
            getattr(response, "fallback_path", []) if response else []
        )
        blockers.extend(item for item in fallback_blockers if item not in blockers)

        if error is not None:
            blockers.insert(0, self._truncate(f"{type(error).__name__}: {error}", 180))
        elif fallback_message and response is None:
            blockers.insert(0, self._truncate(fallback_message, 180))

        task_type = str(
            getattr(response, "task_type", "")
            or getattr(profile, "task_type", "")
            or ""
        )
        clarification_prompt = self._clarification_prompt(response=response, profile=profile)
        if task_type == "clarification" or clarification_prompt:
            questions = self._questions_from_clarification(clarification_prompt)
            return {
                "status": "awaiting_user",
                "completed_steps": completed_steps,
                "pending_steps": ["等待用户补充信息"],
                "open_questions": questions,
                "blockers": [],
                "next_action": "等待用户补充信息",
                "confidence": self._confidence_from_response(response, profile),
            }

        if blockers:
            pending = failed_steps or ["处理阻塞后重试"]
            return {
                "status": "blocked",
                "completed_steps": completed_steps,
                "pending_steps": pending,
                "open_questions": [],
                "blockers": blockers[:8],
                "next_action": f"解决阻塞后继续: {pending[0]}",
                "confidence": self._confidence_from_response(response, profile),
            }

        pending_steps = self._pending_steps_from_trace(trace)
        next_action = (
            pending_steps[0]
            if pending_steps
            else self._completed_next_action(response=response, profile=profile)
        )
        return {
            "status": "completed",
            "completed_steps": completed_steps,
            "pending_steps": pending_steps,
            "open_questions": [],
            "blockers": [],
            "next_action": next_action,
            "confidence": self._confidence_from_response(response, profile),
        }

    def apply_patch_with_retry(
        self,
        *,
        session_id: str,
        patch: Dict[str, Any],
        tenant_id: Optional[str] = None,
        expected_version: Optional[int] = None,
        turn_id: Optional[str] = None,
        created_by: str = "task_state_runtime",
    ) -> Optional[TaskState]:
        if not patch or not self._enabled() or not hasattr(self._memory, "update_task_state"):
            return None

        try:
            return self._memory.update_task_state(
                session_id,
                patch,
                tenant_id=tenant_id,
                expected_version=expected_version,
                created_by=created_by,
                turn_id=turn_id,
            )
        except TaskStateConflictError:
            if expected_version is None or not hasattr(self._memory, "get_task_state"):
                raise
            latest = self._memory.get_task_state(session_id, tenant_id=tenant_id)
            return self._memory.update_task_state(
                session_id,
                patch,
                tenant_id=tenant_id,
                expected_version=latest.version,
                created_by=created_by,
                turn_id=turn_id,
            )

    def enrich_patch_with_llm_async(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_message: str,
        assistant_message: str,
        current_state: Optional[Dict[str, Any] | TaskState],
        tenant_id: Optional[str] = None,
    ) -> None:
        if not self._llm_enrichment_enabled():
            return
        if not self._has_llm_available():
            return

        state_snapshot = self._state_to_dict(current_state)

        def _runner() -> None:
            started = time.perf_counter()
            try:
                patch = self._extract_enrichment_patch(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    current_state=state_snapshot,
                )
                if not patch:
                    return
                if not self._is_latest_task_state_turn(session_id, turn_id):
                    logger.info(
                        "discard stale task_state enrichment: session=%s turn=%s",
                        session_id,
                        turn_id,
                    )
                    return
                latest = (
                    self._memory.get_task_state(session_id, tenant_id=tenant_id)
                    if hasattr(self._memory, "get_task_state")
                    else None
                )
                expected_version = getattr(latest, "version", None)
                self.apply_patch_with_retry(
                    session_id=session_id,
                    patch=patch,
                    tenant_id=tenant_id,
                    expected_version=expected_version,
                    turn_id=turn_id,
                    created_by="task_state_llm_enrichment",
                )
            except Exception as exc:
                logger.warning(
                    "task_state LLM enrichment failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            finally:
                logger.debug(
                    "task_state enrichment finished: session=%s turn=%s ms=%.2f",
                    session_id,
                    turn_id,
                    (time.perf_counter() - started) * 1000.0,
                )

        threading.Thread(target=_runner, daemon=True).start()

    def _extract_enrichment_patch(
        self,
        *,
        user_message: str,
        assistant_message: str,
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        llm = self._resolve_llm()
        if llm is None:
            return {}

        prompt = self._build_enrichment_prompt(
            user_message=user_message,
            assistant_message=assistant_message,
            current_state=current_state,
        )
        result = llm.invoke(prompt)
        raw = getattr(result, "content", None) or str(result)
        payload = self._extract_json(raw)
        if not isinstance(payload, dict):
            return {}

        patch = {
            key: value
            for key, value in payload.items()
            if key in ALLOWED_ENRICHMENT_FIELDS
        }
        return self._sanitize_enrichment_patch(patch)

    def _build_enrichment_prompt(
        self,
        *,
        user_message: str,
        assistant_message: str,
        current_state: Dict[str, Any],
    ) -> str:
        return get_prompt_renderer().render(
            "memory.task_state_enrichment",
            {
                "current_state_json": json.dumps(
                    current_state,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "user_message": user_message,
                "assistant_message": assistant_message[:1200],
            },
        )

    def _sanitize_enrichment_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for key, value in patch.items():
            if key in {"active_constraints", "open_questions", "assumptions"}:
                clean[key] = self._clean_list(value, max_items=12)
            elif key in {"current_goal", "next_action"}:
                text = self._truncate(str(value or "").strip(), 240)
                if text:
                    clean[key] = text
            elif key == "confidence":
                confidence = self._coerce_confidence(value)
                if confidence is not None:
                    clean[key] = confidence
        return clean

    def _is_latest_task_state_turn(self, session_id: str, turn_id: str) -> bool:
        event_store = getattr(self._memory, "event_store", None)
        if event_store is None:
            return True
        events = event_store.list_by_session(
            session_id,
            event_type=MemoryEventType.TASK_STATE_UPDATED.value,
            limit=1,
            descending=True,
        )
        if not events:
            return True
        return events[-1].turn_id == turn_id

    def _resolve_llm(self) -> Optional[Any]:
        if self._llm is not None:
            return self._llm
        if not self._has_llm_available():
            return None
        try:
            from src.core.llm_factory import build_chat_model

            model_name = (
                getattr(settings, "MEMORY_TASK_STATE_EXTRACT_MODEL_NAME", "") or
                getattr(settings, "SMALL_MODEL_NAME", "")
            )
            self._llm = build_chat_model(
                provider=getattr(settings, "SMALL_MODEL_PROVIDER", None),
                model=model_name,
                temperature=0.0,
                request_timeout=float(
                    getattr(settings, "MEMORY_TASK_STATE_EXTRACT_TIMEOUT_SECONDS", 3.0)
                ),
                streaming=False,
                max_retries=0,
            )
            return self._llm
        except Exception as exc:
            logger.warning(
                "task_state enrichment model unavailable: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None

    def _has_llm_available(self) -> bool:
        if self._llm is not None:
            return True
        return bool(getattr(settings, "DASHSCOPE_API_KEY", None))

    def _enabled(self) -> bool:
        return bool(getattr(settings, "MEMORY_TASK_STATE_ENABLED", True))

    def _llm_enrichment_enabled(self) -> bool:
        return self._enabled() and bool(
            getattr(settings, "MEMORY_TASK_STATE_LLM_ENRICH_ENABLED", True)
        )

    @staticmethod
    def _extract_json(raw: str) -> Any:
        text = (raw or "").strip()
        if not text:
            return None
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        elif not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _state_to_dict(state: Optional[Dict[str, Any] | TaskState]) -> Dict[str, Any]:
        if state is None:
            return {}
        if hasattr(state, "to_dict"):
            try:
                return dict(state.to_dict())
            except Exception:
                return {}
        if isinstance(state, dict):
            return dict(state)
        return {}

    @staticmethod
    def _has_meaningful_state(state: Dict[str, Any]) -> bool:
        return bool(
            state.get("current_goal")
            or state.get("next_action")
            or state.get("active_constraints")
            or state.get("pending_steps")
        )

    @staticmethod
    def _is_follow_up_query(query: str) -> bool:
        text = (query or "").strip()
        if not text:
            return False
        if any(cue in text for cue in FOLLOW_UP_CUES):
            return True
        return len(text) <= 32 and bool(re.search(r"^(那|再|还|按|用).{0,24}[吗呢？?]?$", text))

    def _derive_goal(
        self,
        query: str,
        *,
        profile: Optional[Any],
        selected_task_state: Dict[str, Any],
    ) -> str:
        if self._is_follow_up_query(query) and selected_task_state.get("current_goal"):
            return self._truncate(str(selected_task_state["current_goal"]), 180)
        text = (query or "").strip()
        if text:
            return self._truncate(text, 180)
        task_type = getattr(profile, "task_type", "") if profile else ""
        return task_type or "处理当前任务"

    @staticmethod
    def _pending_steps_from_plan(execution_plan: Optional[Any]) -> list[str]:
        if execution_plan is None:
            return []
        if hasattr(execution_plan, "steps"):
            raw_steps = getattr(execution_plan, "steps", []) or []
            return [
                str(getattr(step, "title", "") or getattr(step, "skill", "") or getattr(step, "id", "")).strip()
                for step in raw_steps
                if str(getattr(step, "title", "") or getattr(step, "skill", "") or getattr(step, "id", "")).strip()
            ][:12]
        if isinstance(execution_plan, dict):
            raw_steps = execution_plan.get("steps") or []
            return [
                str(step.get("title") or step.get("skill") or step.get("id") or "").strip()
                for step in raw_steps
                if isinstance(step, dict)
                and str(step.get("title") or step.get("skill") or step.get("id") or "").strip()
            ][:12]
        return []

    @staticmethod
    def _pending_steps_from_profile(
        *,
        profile: Optional[Any],
        execution_decision: Optional[Any],
    ) -> list[str]:
        mode = getattr(execution_decision, "mode", "") if execution_decision else ""
        skills = (
            list(getattr(profile, "capability_hints", []) or [])
            if profile
            else []
        )
        task_type = getattr(profile, "task_type", "") if profile else ""
        if task_type == "clarification":
            return ["向用户澄清关键信息"]
        if skills:
            return [f"调用 {skill}" for skill in skills[:8]]
        if mode == "react":
            return ["逐步推理并按需调用工具"]
        if mode == "planned":
            return ["生成计划", "执行计划", "汇总答案"]
        return ["生成回答"]

    @staticmethod
    def _derive_start_next_action(
        pending_steps: list[str],
        *,
        profile: Optional[Any],
        execution_decision: Optional[Any],
    ) -> str:
        clarification = getattr(profile, "clarification_prompt", "") if profile else ""
        if clarification:
            return "向用户澄清关键信息"
        return pending_steps[0] if pending_steps else "执行当前任务"

    @staticmethod
    def _completed_steps_from_trace(trace: Iterable[Dict[str, Any]]) -> list[str]:
        completed: list[str] = []
        for item in trace:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).lower()
            if status in {"success", "succeeded", "completed", "done"}:
                label = str(
                    item.get("title")
                    or item.get("skill")
                    or item.get("tool_name")
                    or item.get("step_id")
                    or ""
                ).strip()
                if label and label not in completed:
                    completed.append(label)
        return completed[:20]

    @staticmethod
    def _pending_steps_from_trace(trace: Iterable[Dict[str, Any]]) -> list[str]:
        pending: list[str] = []
        for item in trace:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).lower()
            if status in {"pending", "running"}:
                label = str(
                    item.get("title") or item.get("skill") or item.get("step_id") or ""
                ).strip()
                if label and label not in pending:
                    pending.append(label)
        return pending[:12]

    def _failed_steps_from_trace(
        self, trace: Iterable[Dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        failed: list[str] = []
        blockers: list[str] = []
        for item in trace:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).lower()
            if status not in {"error", "failed", "failure", "timeout"}:
                continue
            label = str(item.get("title") or item.get("skill") or item.get("step_id") or "").strip()
            if label and label not in failed:
                failed.append(label)
            detail = str(item.get("error") or item.get("summary") or label or "步骤失败").strip()
            if detail:
                blockers.append(self._truncate(detail, 180))
        return failed[:12], blockers[:8]

    @staticmethod
    def _completed_steps_from_tools(tools_used: Iterable[Dict[str, Any]]) -> list[str]:
        completed: list[str] = []
        for item in tools_used or []:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "success")).lower()
            if status not in {"success", "succeeded", "completed", "done"}:
                continue
            label = str(item.get("tool") or item.get("tool_name") or item.get("name") or "").strip()
            if label and label not in completed:
                completed.append(label)
        return completed[:20]

    def _blockers_from_fallback_path(self, fallback_path: Iterable[Dict[str, Any]]) -> list[str]:
        blockers: list[str] = []
        for item in fallback_path or []:
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason") or item.get("strategy") or "").strip()
            metadata = item.get("metadata") or {}
            halt_reason = ""
            if isinstance(metadata, dict):
                halt_reason = str(metadata.get("halt_reason") or metadata.get("error") or "").strip()
            text = halt_reason or reason
            if text:
                blockers.append(self._truncate(text, 180))
        return blockers[:8]

    @staticmethod
    def _clarification_prompt(*, response: Optional[Any], profile: Optional[Any]) -> str:
        if response is not None and getattr(response, "task_type", "") == "clarification":
            return str(getattr(response, "answer", "") or "")
        return str(getattr(profile, "clarification_prompt", "") or "") if profile else ""

    @staticmethod
    def _questions_from_clarification(prompt: str) -> list[str]:
        text = (prompt or "").strip()
        if not text:
            return ["请补充目标、时间、地点或器材参数"]
        pieces = [piece.strip() for piece in re.split(r"[？?\n]", text) if piece.strip()]
        if not pieces:
            return [text[:180]]
        return [piece[:180] for piece in pieces[:5]]

    @staticmethod
    def _completed_next_action(*, response: Optional[Any], profile: Optional[Any]) -> str:
        summary = str(getattr(response, "summary", "") or "").strip() if response else ""
        if summary:
            return "等待用户确认是否继续细化: " + summary[:80]
        task_type = getattr(profile, "task_type", "") if profile else ""
        if task_type:
            return f"等待用户确认是否继续处理 {task_type}"
        return "等待用户确认是否继续"

    @staticmethod
    def _confidence_from_profile(profile: Optional[Any]) -> Optional[float]:
        if profile is None:
            return None
        return TaskStateRuntimeService._coerce_confidence(
            getattr(profile, "confidence", None)
        )

    @staticmethod
    def _confidence_from_response(response: Optional[Any], profile: Optional[Any]) -> Optional[float]:
        if response is not None:
            confidence = TaskStateRuntimeService._coerce_confidence(
                getattr(response, "confidence", None)
            )
            if confidence is not None:
                return confidence
        return TaskStateRuntimeService._confidence_from_profile(profile)

    @staticmethod
    def _coerce_confidence(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def _clean_list(value: Any, *, max_items: int) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            cleaned.append(text[:180])
            seen.add(text)
            if len(cleaned) >= max_items:
                break
        return cleaned

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        text = text or ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."

    @staticmethod
    def _is_smalltalk(*, response: Optional[Any] = None, profile: Optional[Any] = None) -> bool:
        return (
            getattr(response, "task_type", "") == "smalltalk"
            or getattr(profile, "task_type", "") == "smalltalk"
        )
