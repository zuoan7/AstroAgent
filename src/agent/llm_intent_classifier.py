from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from src.agent.prompts import get_prompt_renderer
from src.skills import registry


VALID_ROUTES = {"direct_task", "planned_task", "fallback_react"}
VALID_TASK_TYPES = {
    "smalltalk",
    "simple_qa",
    "clarification",
    "direct_answer_no_tool",
    "single_tool_lookup",
    "observation_recommendation",
    "celestial_event_analysis",
    "deep_sky_guidance",
    "astrophotography_advice",
    "open_domain_reasoning",
}


@dataclass(frozen=True)
class LLMIntentResult:
    requires_tool: bool
    route: str
    task_type: str
    skills: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    should_clarify: bool = False
    param_hints: dict[str, Any] = field(default_factory=dict)


class LLMIntentClassifier:
    """Strict JSON intent classifier used only as low-confidence router fallback."""

    def __init__(
        self,
        llm: Any,
        *,
        skill_specs: Optional[Iterable[Any]] = None,
        min_accept_confidence: float = 0.55,
    ) -> None:
        self._llm = llm
        self._skill_specs = list(skill_specs or registry.get_skill_specs())
        self._allowed_skills = {spec.skill_name for spec in self._skill_specs}
        self._min_accept_confidence = float(min_accept_confidence)

    def classify(
        self,
        query: str,
        *,
        rule_profile: Any = None,
    ) -> Optional[LLMIntentResult]:
        prompt = self._build_prompt(query, rule_profile=rule_profile)
        raw = self._invoke(prompt)
        payload = self._extract_json(raw)
        if not isinstance(payload, dict):
            return None
        return self._parse_payload(payload)

    def _invoke(self, prompt: str) -> str:
        result = self._llm.invoke(prompt)
        return getattr(result, "content", None) or str(result)

    def _build_prompt(self, query: str, *, rule_profile: Any = None) -> str:
        skills_text = "\n".join(
            f"- {spec.skill_name}: {spec.summary}" for spec in self._skill_specs
        )
        rule_summary = ""
        if rule_profile is not None and hasattr(rule_profile, "to_dict"):
            try:
                rule_summary = json.dumps(
                    rule_profile.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            except Exception:
                rule_summary = ""

        return get_prompt_renderer().render(
            "router.intent_classifier",
            {
                "valid_routes": sorted(VALID_ROUTES),
                "valid_task_types": sorted(VALID_TASK_TYPES),
                "skills_text": skills_text,
                "rule_summary": rule_summary or "无",
                "query": query,
            },
        )

    def _extract_json(self, raw: str) -> Any:
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

    def _parse_payload(self, payload: dict[str, Any]) -> Optional[LLMIntentResult]:
        route = str(payload.get("route") or "").strip()
        task_type = str(payload.get("task_type") or "").strip()
        if route not in VALID_ROUTES or task_type not in VALID_TASK_TYPES:
            return None

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        if not 0.0 <= confidence <= 1.0 or confidence < self._min_accept_confidence:
            return None

        raw_skills = payload.get("skills") or []
        if not isinstance(raw_skills, list):
            return None
        skills = []
        for item in raw_skills:
            skill = str(item).strip()
            if skill not in self._allowed_skills:
                return None
            if skill not in skills:
                skills.append(skill)

        requires_tool = bool(payload.get("requires_tool", bool(skills)))
        if requires_tool and not skills:
            return None
        if not requires_tool and skills:
            return None

        param_hints = payload.get("param_hints") or {}
        if not isinstance(param_hints, dict):
            param_hints = {}

        return LLMIntentResult(
            requires_tool=requires_tool,
            route=route,
            task_type=task_type,
            skills=skills,
            confidence=confidence,
            reason=str(payload.get("reason") or "").strip(),
            should_clarify=bool(payload.get("should_clarify", False)),
            param_hints=param_hints,
        )
