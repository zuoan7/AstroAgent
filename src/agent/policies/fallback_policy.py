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
            if any(not step.required and step.status == "error" for step in getattr(outcome, "step_results", [])):
                return FallbackDecision(
                    strategy="partial_answer",
                    reason="optional_steps_failed_but_required_path_completed",
                    metadata={"halt_reason": halt_reason},
                )
            return FallbackDecision(
                strategy="react_fallback",
                reason="required_step_failed",
                metadata={
                    "halt_reason": halt_reason,
                    "task_type": getattr(plan, "task_type", ""),
                },
            )
        optional_errors = [
            step.step_id
            for step in getattr(outcome, "step_results", [])
            if not step.required and step.status == "error"
        ]
        if optional_errors:
            return FallbackDecision(
                strategy="partial_answer",
                reason="optional_step_failed",
                metadata={"optional_failed_steps": optional_errors},
            )
        return None

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
