from __future__ import annotations

from types import SimpleNamespace

from src.agent.llm_intent_classifier import LLMIntentClassifier, LLMIntentResult
from src.agent.request_router import RequestRouter


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.content)


class _FakeClassifier:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[str, object]] = []

    def classify(self, query: str, *, rule_profile=None):
        self.calls.append((query, rule_profile))
        return self.result


def test_llm_intent_classifier_accepts_fenced_json_and_validates_registry_skills():
    llm = _FakeLLM(
        """```json
        {
          "requires_tool": true,
          "route": "direct_task",
          "task_type": "single_tool_lookup",
          "skills": ["weather-lookup"],
          "confidence": 0.86,
          "reason": "询问今晚云量",
          "should_clarify": false,
          "param_hints": {"location": "北京", "date": "今晚"}
        }
        ```"""
    )
    result = LLMIntentClassifier(llm).classify("今晚云会不会影响观测？")

    assert result is not None
    assert result.requires_tool is True
    assert result.skills == ["weather-lookup"]
    assert result.param_hints == {"location": "北京", "date": "今晚"}
    assert "weather-lookup" in llm.prompts[0]


def test_llm_intent_classifier_rejects_unknown_skill():
    llm = _FakeLLM(
        '{"requires_tool": true, "route": "direct_task", '
        '"task_type": "single_tool_lookup", "skills": ["made-up-tool"], '
        '"confidence": 0.9, "reason": "bad"}'
    )

    assert LLMIntentClassifier(llm).classify("今晚那颗亮星在哪？") is None


def test_llm_intent_classifier_rejects_low_confidence():
    llm = _FakeLLM(
        '{"requires_tool": true, "route": "direct_task", '
        '"task_type": "single_tool_lookup", "skills": ["weather-lookup"], '
        '"confidence": 0.2, "reason": "uncertain"}'
    )

    assert LLMIntentClassifier(llm).classify("今晚薄云如何？") is None


def test_router_uses_llm_fallback_for_low_confidence_astronomy_boundary():
    classifier = _FakeClassifier(
        LLMIntentResult(
            requires_tool=True,
            route="planned_task",
            task_type="observation_recommendation",
            skills=["observation-planner"],
            confidence=0.84,
            reason="隐含的今晚观测建议",
        )
    )
    router = RequestRouter(
        llm_intent_classifier=classifier,
        enable_llm_fallback=True,
    )

    profile = router.profile("今晚亮星适合追踪吗？")

    assert classifier.calls
    assert profile.legacy_route == "planned_task"
    assert profile.task_type == "observation_recommendation"
    assert profile.matched_skills == ["observation-planner"]
    assert profile.reason.startswith("llm_intent_fallback:")


def test_router_keeps_high_confidence_rule_result_without_llm_call():
    classifier = _FakeClassifier(
        LLMIntentResult(
            requires_tool=True,
            route="planned_task",
            task_type="observation_recommendation",
            skills=["observation-planner"],
            confidence=0.9,
            reason="should not be used",
        )
    )
    router = RequestRouter(
        llm_intent_classifier=classifier,
        enable_llm_fallback=True,
    )

    profile = router.profile("北京今晚云多吗？")

    assert classifier.calls == []
    assert profile.legacy_route == "direct_task"
    assert profile.matched_skills == ["weather-lookup"]


def test_router_keeps_stable_knowledge_simple_qa_without_llm_call():
    classifier = _FakeClassifier(
        LLMIntentResult(
            requires_tool=True,
            route="direct_task",
            task_type="single_tool_lookup",
            skills=["celestial-position-calculator"],
            confidence=0.9,
            reason="should not be used",
        )
    )
    router = RequestRouter(
        llm_intent_classifier=classifier,
        enable_llm_fallback=True,
    )

    profile = router.profile("赤经是什么？")

    assert classifier.calls == []
    assert profile.task_type == "simple_qa"
    assert profile.matched_skills == []


def test_router_ignores_invalid_llm_fallback_result():
    classifier = _FakeClassifier(None)
    router = RequestRouter(
        llm_intent_classifier=classifier,
        enable_llm_fallback=True,
    )

    profile = router.profile("今晚亮星适合追踪吗？")

    assert classifier.calls
    assert profile.task_type == "simple_qa"
    assert profile.matched_skills == []


def test_router_normalizes_llm_direct_tool_result_for_direct_executor_contract():
    classifier = _FakeClassifier(
        LLMIntentResult(
            requires_tool=True,
            route="direct_task",
            task_type="deep_sky_guidance",
            skills=["deep-sky-observing-guide"],
            confidence=0.88,
            reason="单个深空目标查询",
        )
    )
    router = RequestRouter(
        llm_intent_classifier=classifier,
        enable_llm_fallback=True,
    )

    profile = router.profile("今晚亮星适合追踪吗？")

    assert profile.legacy_route == "direct_task"
    assert profile.task_type == "single_tool_lookup"
    assert profile.matched_skills == ["deep-sky-observing-guide"]
