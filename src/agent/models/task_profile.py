"""
TaskProfile — 中性任务画像，Phase 1 引入。

当前阶段：已成为 RequestRouter 内部主分类结果。
收敛计划：后续继续向 UnifiedExecutionEngine 主输入收口，旧 RouteDecision 仅保留兼容输出。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


LEGACY_ROUTE_MAP: Dict[str, str] = {
    "direct_task": "direct_task",
    "planned_task": "planned_task",
    "fallback_react": "fallback_react",
}


@dataclass
class TaskProfile:
    task_type: str
    complexity: str          # low | medium | high
    openness: str            # low | medium | high
    tool_need: str           # none | single | multi
    matched_skills: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    expected_output_schema: str = "generic_answer_v1"
    legacy_route: str = "fallback_react"
    router_source: str = "rule"
    rule_confidence: Optional[float] = None
    llm_confidence: Optional[float] = None
    tool_necessity_action: str = ""
    tool_necessity_reason: str = ""
    tool_necessity_confidence: Optional[float] = None
    answer_hint: str = ""
    clarification_prompt: str = ""
    tool_necessity_missing_params: List[str] = field(default_factory=list)
    tool_necessity_allowed_skill_hints: List[str] = field(default_factory=list)
    tool_necessity_forbidden_skill_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "task_type": self.task_type,
            "complexity": self.complexity,
            "openness": self.openness,
            "tool_need": self.tool_need,
            "matched_skills": list(self.matched_skills),
            "confidence": self.confidence,
            "reason": self.reason,
            "expected_output_schema": self.expected_output_schema,
            "legacy_route": self.legacy_route,
            "router_source": self.router_source,
            "rule_confidence": self.rule_confidence,
            "llm_confidence": self.llm_confidence,
            "tool_necessity_action": self.tool_necessity_action,
            "tool_necessity_reason": self.tool_necessity_reason,
            "tool_necessity_confidence": self.tool_necessity_confidence,
            "answer_hint": self.answer_hint,
            "clarification_prompt": self.clarification_prompt,
            "tool_necessity_missing_params": list(self.tool_necessity_missing_params),
            "tool_necessity_allowed_skill_hints": list(
                self.tool_necessity_allowed_skill_hints
            ),
            "tool_necessity_forbidden_skill_hints": list(
                self.tool_necessity_forbidden_skill_hints
            ),
        }

    def to_legacy_route_decision(self) -> "RouteDecision":
        """转换为旧 RouteDecision，供旧执行链路继续消费。"""
        from src.agent.request_router import RouteDecision

        return RouteDecision(
            route=self.legacy_route,
            task_type=self.task_type,
            confidence=self.confidence,
            reason=self.reason,
            matched_skills=list(self.matched_skills),
            expected_output_schema=self.expected_output_schema,
            router_source=self.router_source,
            rule_confidence=self.rule_confidence,
            llm_confidence=self.llm_confidence,
            tool_necessity_action=self.tool_necessity_action,
            tool_necessity_reason=self.tool_necessity_reason,
            tool_necessity_confidence=self.tool_necessity_confidence,
            answer_hint=self.answer_hint,
            clarification_prompt=self.clarification_prompt,
            tool_necessity_missing_params=list(self.tool_necessity_missing_params),
            tool_necessity_allowed_skill_hints=list(
                self.tool_necessity_allowed_skill_hints
            ),
            tool_necessity_forbidden_skill_hints=list(
                self.tool_necessity_forbidden_skill_hints
            ),
        )

    @classmethod
    def from_legacy_route(
        cls,
        route: str,
        task_type: str,
        confidence: float,
        matched_skills: Optional[List[str]] = None,
        reason: str = "",
        expected_output_schema: str = "generic_answer_v1",
        router_source: str = "rule",
        rule_confidence: Optional[float] = None,
        llm_confidence: Optional[float] = None,
        tool_necessity_action: str = "",
        tool_necessity_reason: str = "",
        tool_necessity_confidence: Optional[float] = None,
        answer_hint: str = "",
        clarification_prompt: str = "",
        tool_necessity_missing_params: Optional[List[str]] = None,
        tool_necessity_allowed_skill_hints: Optional[List[str]] = None,
        tool_necessity_forbidden_skill_hints: Optional[List[str]] = None,
    ) -> "TaskProfile":
        """从旧 RouteDecision 字段推断 TaskProfile，保证双向可转换。"""
        skills = list(matched_skills or [])

        if route == "direct_task" and task_type == "smalltalk":
            complexity, openness, tool_need = "low", "low", "none"
        elif route == "direct_task" and task_type == "simple_qa":
            complexity, openness, tool_need = "low", "low", "none"
        elif route == "direct_task" and task_type in {
            "clarification",
            "direct_answer_no_tool",
        }:
            complexity, openness, tool_need = "low", "low", "none"
        elif route == "direct_task" and task_type == "single_tool_lookup":
            complexity, openness, tool_need = "low", "low", "single"
        elif route == "planned_task":
            skill_count = len(skills)
            complexity = "high" if skill_count >= 2 else "medium"
            openness = "low"
            tool_need = "multi" if skill_count >= 2 else "single"
        elif route == "fallback_react":
            complexity, openness, tool_need = "high", "high", "none"
        else:
            complexity, openness, tool_need = "medium", "medium", "single"

        return cls(
            task_type=task_type,
            complexity=complexity,
            openness=openness,
            tool_need=tool_need,
            matched_skills=skills,
            confidence=confidence,
            reason=reason,
            expected_output_schema=expected_output_schema,
            legacy_route=LEGACY_ROUTE_MAP.get(route, route),
            router_source=router_source,
            rule_confidence=confidence if rule_confidence is None and router_source == "rule" else rule_confidence,
            llm_confidence=llm_confidence,
            tool_necessity_action=tool_necessity_action,
            tool_necessity_reason=tool_necessity_reason,
            tool_necessity_confidence=tool_necessity_confidence,
            answer_hint=answer_hint,
            clarification_prompt=clarification_prompt,
            tool_necessity_missing_params=list(tool_necessity_missing_params or []),
            tool_necessity_allowed_skill_hints=list(
                tool_necessity_allowed_skill_hints or []
            ),
            tool_necessity_forbidden_skill_hints=list(
                tool_necessity_forbidden_skill_hints or []
            ),
        )
