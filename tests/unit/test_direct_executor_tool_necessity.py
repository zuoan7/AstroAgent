from __future__ import annotations

import pytest

from src.agent.execution.direct_executor import DirectExecutor
from src.agent.request_router import RouteDecision


class _FailingSkillManager:
    def call_skill(self, name, **params):
        raise AssertionError("skill should not be called")


class _FailingRAG:
    def retrieve(self, query, fast_mode=False):
        raise AssertionError("RAG should not be called")


class _FakeLLM:
    def invoke(self, prompt):
        raise AssertionError("LLM should not be called when answer_hint is present")


class _FakeSynthesizer:
    pass


def _executor() -> DirectExecutor:
    return DirectExecutor(
        skill_manager=_FailingSkillManager(),
        rag_retriever=_FailingRAG(),
        llm=_FakeLLM(),
        synthesizer=_FakeSynthesizer(),
    )


@pytest.mark.asyncio
async def test_clarification_path_does_not_call_rag_or_skill():
    decision = RouteDecision(
        route="direct_task",
        task_type="clarification",
        confidence=0.9,
        reason="tool_necessity_gate:test",
        clarification_prompt="请补充焦距和是否跟踪。",
        tool_necessity_action="clarify",
        tool_necessity_reason="missing_params",
        tool_necessity_confidence=0.9,
        tool_necessity_missing_params=["focal_length"],
    )

    response = await _executor().run(decision, "帮我算曝光多久")

    assert response.answer == "请补充焦距和是否跟踪。"
    assert response.tools_used == []
    assert response.sources == []
    assert response.audit_metadata["tool_necessity_action"] == "clarify"


@pytest.mark.asyncio
async def test_direct_answer_no_tool_path_does_not_call_rag_or_skill():
    decision = RouteDecision(
        route="direct_task",
        task_type="direct_answer_no_tool",
        confidence=0.9,
        reason="tool_necessity_gate:test",
        answer_hint="湿度高时镜头可能结露，可以使用除露带。",
        tool_necessity_action="answer_without_tool",
        tool_necessity_reason="stable_experience",
        tool_necessity_confidence=0.9,
    )

    response = await _executor().run(decision, "湿度很高会不会镜头起雾？")

    assert "结露" in response.answer
    assert response.tools_used == []
    assert response.sources == []
    assert response.audit_metadata["tool_necessity_action"] == "answer_without_tool"
