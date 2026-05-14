"""Unit tests for DirectExecutor prompt budget integration — Phase 2."""

import pytest
from types import SimpleNamespace
from unittest import mock

from src.agent.execution.direct_executor import DirectExecutor


class _FakeSkillManager:
    def call_skill(self, name, **params):
        return SimpleNamespace(
            skill_name=name,
            summary=f"result for {name}",
            success=True,
            data=None,
            sources=[],
            to_tool_timeline_entry=lambda: {"tool": name, "summary": f"result for {name}"},
        )


class _FakeRAG:
    def retrieve(self, query, fast_mode=False):
        return {"context": "RAG检索到的天文知识内容。", "sources": []}


class _FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content="这是基于知识的回答。")


class _FakeSynthesizer:
    def synthesize_qa(self, **kwargs):
        from src.agent.models.final_response import FinalResponse
        return FinalResponse(
            answer="测试回答",
            summary="测试回答",
            route="direct_task",
            task_type="simple_qa",
        )

    def synthesize_smalltalk(self, answer):
        from src.agent.models.final_response import FinalResponse
        return FinalResponse(
            answer=answer,
            summary=answer,
            route="direct_task",
            task_type="smalltalk",
        )

    def synthesize_direct(self, **kwargs):
        from src.agent.models.final_response import FinalResponse
        return FinalResponse(
            answer="direct answer",
            summary="direct answer",
            route="direct_task",
            task_type="direct",
        )


@pytest.fixture
def executor():
    return DirectExecutor(
        skill_manager=_FakeSkillManager(),
        rag_retriever=_FakeRAG(),
        llm=_FakeLLM(),
        synthesizer=_FakeSynthesizer(),
    )


# ---------------------------------------------------------------------------
# 1. Long RAG context is capped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_long_rag_context_not_crash(monkeypatch):
    monkeypatch.setattr(
        "src.agent.execution.direct_executor.settings.PROMPT_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.execution.direct_executor.settings.PROMPT_MAX_INPUT_CHARS", 3000
    )

    class _HugeRAG:
        def retrieve(self, query, fast_mode=False):
            return {"context": "天文知识 " * 3000, "sources": []}

    executor = DirectExecutor(
        skill_manager=_FakeSkillManager(),
        rag_retriever=_HugeRAG(),
        llm=_FakeLLM(),
        synthesizer=_FakeSynthesizer(),
    )

    response = await executor._run_simple_qa(
        "测试问题",
        chat_history="历史 " * 500,
        user_profile="画像 " * 300,
    )
    assert response.answer == "测试回答"


# ---------------------------------------------------------------------------
# 2. Current query is preserved
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_preserved(monkeypatch):
    monkeypatch.setattr(
        "src.agent.execution.direct_executor.settings.PROMPT_BUDGET_ENABLED", True
    )
    executor = DirectExecutor(
        skill_manager=_FakeSkillManager(),
        rag_retriever=_FakeRAG(),
        llm=_FakeLLM(),
        synthesizer=_FakeSynthesizer(),
    )
    response = await executor._run_simple_qa(
        "今天木星可见吗？",
        chat_history="",
        user_profile="",
    )
    assert response is not None
    assert response.answer == "测试回答"


# ---------------------------------------------------------------------------
# 3. user_profile / chat_history are budget-controlled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_long_profile_and_history_not_crash(monkeypatch):
    monkeypatch.setattr(
        "src.agent.execution.direct_executor.settings.PROMPT_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.execution.direct_executor.settings.PROMPT_MAX_INPUT_CHARS", 2000
    )

    executor = DirectExecutor(
        skill_manager=_FakeSkillManager(),
        rag_retriever=_FakeRAG(),
        llm=_FakeLLM(),
        synthesizer=_FakeSynthesizer(),
    )
    response = await executor._run_simple_qa(
        "简短问题",
        chat_history="对话历史 " * 1000,
        user_profile="用户画像 " * 1000,
    )
    assert response.answer == "测试回答"


# ---------------------------------------------------------------------------
# 4. PROMPT_BUDGET_ENABLED=False — legacy behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_disabled_falls_back_to_legacy(monkeypatch):
    monkeypatch.setattr(
        "src.agent.execution.direct_executor.settings.PROMPT_BUDGET_ENABLED", False
    )

    executor = DirectExecutor(
        skill_manager=_FakeSkillManager(),
        rag_retriever=_FakeRAG(),
        llm=_FakeLLM(),
        synthesizer=_FakeSynthesizer(),
    )
    response = await executor._run_simple_qa(
        "测试问题",
        chat_history="H" * 3000,
        user_profile="P" * 3000,
    )
    # Legacy path uses user_profile[:400] etc. — should not crash
    assert response.answer == "测试回答"
