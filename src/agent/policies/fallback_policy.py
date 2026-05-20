from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from src.core.config import settings


@dataclass(frozen=True)
class FallbackDecision:
    strategy: str
    reason: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class FallbackPolicy:
    PLAN_REPAIR_ERROR_HINTS = (
        "missing required",
        "required parameter",
        "missing parameter",
        "invalid dependency",
        "depends_on",
        "unsupported node kind",
        "unsupported step kind",
        "workflowgraph",
        "no executable dag nodes",
        "dependency resolution stalled",
        "不存在",
        "循环依赖",
        "没有任何节点",
    )

    def __init__(self) -> None:
        self.version = str(
            getattr(settings, "FALLBACK_POLICY_VERSION", "fallback_v2")
        )

    def decide_for_execution(
        self,
        *,
        outcome: Any,
        plan: Any,
        cache_hit: Optional[Dict[str, Any]] = None,
    ) -> Optional[FallbackDecision]:
        if cache_hit:
            return FallbackDecision(
                strategy="cached_answer",
                reason="reused_cached_final_response",
                metadata={"cache_keys": list(cache_hit)},
            )
        if getattr(outcome, "halted", False):
            halt_reason = getattr(outcome, "halt_reason", "") or "executor_halted"
            required_failures = [
                step.step_id
                for step in getattr(outcome, "step_results", [])
                if step.required and step.status in {"error", "skipped"}
            ]
            if required_failures:
                required_failure_messages = [
                    getattr(step, "error", "") or ""
                    for step in getattr(outcome, "step_results", [])
                    if step.required and step.status in {"error", "skipped"}
                ]
                if self._is_plan_repairable(
                    [halt_reason, *required_failure_messages]
                ):
                    return FallbackDecision(
                        strategy="plan_repair",
                        reason="repairable_plan_failure",
                        metadata={
                            "halt_reason": halt_reason,
                            "required_failed_steps": required_failures,
                            "task_type": getattr(plan, "task_type", ""),
                            "recovery_mode": "plan_repair",
                            "executed": False,
                        },
                    )
                return FallbackDecision(
                    strategy="react_fallback",
                    reason="required_step_failed",
                    metadata={
                        "halt_reason": halt_reason,
                        "required_failed_steps": required_failures,
                        "task_type": getattr(plan, "task_type", ""),
                        "recovery_mode": "react",
                        "executed": False,
                    },
                )
            if any(
                not step.required and step.status in {"error", "skipped"}
                for step in getattr(outcome, "step_results", [])
            ):
                return FallbackDecision(
                    strategy="partial_answer",
                    reason="optional_steps_failed_but_required_path_completed",
                    metadata={
                        "halt_reason": halt_reason,
                        "recovery_mode": "partial_answer",
                        "executed": True,
                    },
                )
            if self._is_plan_repairable([halt_reason]):
                return FallbackDecision(
                    strategy="plan_repair",
                    reason="repairable_plan_failure",
                    metadata={
                        "halt_reason": halt_reason,
                        "task_type": getattr(plan, "task_type", ""),
                        "recovery_mode": "plan_repair",
                        "executed": False,
                    },
                )
            return FallbackDecision(
                strategy="react_fallback",
                reason="required_step_failed",
                metadata={
                    "halt_reason": halt_reason,
                    "task_type": getattr(plan, "task_type", ""),
                    "recovery_mode": "react",
                    "executed": False,
                },
            )
        required_errors = [
            step.step_id
            for step in getattr(outcome, "step_results", [])
            if step.required and step.status in {"error", "skipped"}
        ]
        if required_errors:
            required_error_messages = [
                getattr(step, "error", "") or ""
                for step in getattr(outcome, "step_results", [])
                if step.required and step.status in {"error", "skipped"}
            ]
            if self._is_plan_repairable(required_error_messages):
                return FallbackDecision(
                    strategy="plan_repair",
                    reason="repairable_plan_failure",
                    metadata={
                        "required_failed_steps": required_errors,
                        "task_type": getattr(plan, "task_type", ""),
                        "recovery_mode": "plan_repair",
                        "executed": False,
                    },
                )
            return FallbackDecision(
                strategy="react_fallback",
                reason="required_step_failed",
                metadata={
                    "required_failed_steps": required_errors,
                    "task_type": getattr(plan, "task_type", ""),
                    "recovery_mode": "react",
                    "executed": False,
                },
            )

        optional_errors = [
            step.step_id
            for step in getattr(outcome, "step_results", [])
            if not step.required and step.status in {"error", "skipped"}
        ]
        if optional_errors:
            return FallbackDecision(
                strategy="partial_answer",
                reason="optional_step_failed",
                metadata={
                    "optional_failed_steps": optional_errors,
                    "recovery_mode": "partial_answer",
                    "executed": True,
                },
            )
        return None

    def _is_plan_repairable(self, messages: Iterable[str]) -> bool:
        for message in messages:
            text = str(message or "").strip().lower()
            if not text:
                continue
            if any(hint in text for hint in self.PLAN_REPAIR_ERROR_HINTS):
                return True
        return False

    def classify_tool_failure(
        self,
        *,
        error_message: str,
        retries_attempted: int,
        alternate_skills: Optional[Iterable[str]] = None,
    ) -> FallbackDecision:
        alternates = list(alternate_skills or [])
        if retries_attempted > 1:
            return FallbackDecision(
                strategy="tool_retry",
                reason="tool_failed_after_retry_attempts",
                metadata={"retries_attempted": retries_attempted},
            )
        if alternates:
            return FallbackDecision(
                strategy="alternate_tool",
                reason="primary_tool_failed_alternate_available",
                metadata={"alternate_skills": alternates},
            )
        return FallbackDecision(
            strategy="web_fallback",
            reason=error_message or "tool_failure_without_recovery",
            metadata={},
        )
