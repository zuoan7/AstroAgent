from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict

from src.core.config import settings


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class RequestBudget:
    max_llm_calls: int
    max_tool_calls: int
    max_total_time_ms: int
    max_parallelism: int
    max_context_chars: int
    policy_version: str = "budget_v1"

    @classmethod
    def from_settings(cls) -> "RequestBudget":
        return cls(
            max_llm_calls=int(getattr(settings, "AGENT_MAX_LLM_CALLS", 4)),
            max_tool_calls=int(getattr(settings, "AGENT_MAX_TOOL_CALLS", 6)),
            max_total_time_ms=int(getattr(settings, "AGENT_MAX_TOTAL_TIME_MS", 60000)),
            max_parallelism=int(getattr(settings, "AGENT_MAX_PARALLELISM", 2)),
            max_context_chars=int(getattr(settings, "AGENT_MAX_CONTEXT_CHARS", 6000)),
            policy_version=str(getattr(settings, "BUDGET_POLICY_VERSION", "budget_v1")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RequestBudgetTracker:
    def __init__(self, budget: RequestBudget | None = None) -> None:
        self.budget = budget or RequestBudget.from_settings()
        self.started_at = time.perf_counter()
        self.llm_calls = 0
        self.tool_calls = 0
        self.context_chars = 0
        self.max_parallelism_seen = 1

    def ensure_time_budget(self) -> None:
        if self.elapsed_ms() > self.budget.max_total_time_ms:
            raise BudgetExceededError(
                f"total time budget exceeded: {self.elapsed_ms()} ms > {self.budget.max_total_time_ms} ms"
            )

    def register_llm_call(self) -> None:
        self.ensure_time_budget()
        self.llm_calls += 1
        if self.llm_calls > self.budget.max_llm_calls:
            raise BudgetExceededError(
                f"llm call budget exceeded: {self.llm_calls} > {self.budget.max_llm_calls}"
            )

    def register_tool_call(self) -> None:
        self.ensure_time_budget()
        self.tool_calls += 1
        if self.tool_calls > self.budget.max_tool_calls:
            raise BudgetExceededError(
                f"tool call budget exceeded: {self.tool_calls} > {self.budget.max_tool_calls}"
            )

    def register_parallelism(self, parallelism: int) -> None:
        self.max_parallelism_seen = max(self.max_parallelism_seen, max(parallelism, 1))
        if parallelism > self.budget.max_parallelism:
            raise BudgetExceededError(
                f"parallelism budget exceeded: {parallelism} > {self.budget.max_parallelism}"
            )

    def register_context_chars(self, chars: int) -> None:
        self.context_chars += max(chars, 0)
        if self.context_chars > self.budget.max_context_chars:
            raise BudgetExceededError(
                f"context budget exceeded: {self.context_chars} > {self.budget.max_context_chars}"
            )

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000.0, 2)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "budget": self.budget.to_dict(),
            "usage": {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "context_chars": self.context_chars,
                "max_parallelism_seen": self.max_parallelism_seen,
                "elapsed_ms": self.elapsed_ms(),
            },
        }
