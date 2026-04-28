"""Phase 4 ExecutionEngine 测试

目标：验证 DirectExecutor / PlannedExecutor / ReactExecutor / ExecutionEngine 结构，
      以及三种模式都可通过 ExecutionEngine 被调用，旧路径仍可工作。

当前状态：ExecutionEngine 已是默认主路径，TaskOrchestrator 为兼容回退门面。
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.execution.engine import ExecutionEngine
from src.agent.execution.direct_executor import DirectExecutor
from src.agent.execution.planned_executor import PlannedExecutor
from src.agent.execution.react_executor import ReactExecutor
from src.agent.models.execution_context import ExecutionContext
from src.agent.models.execution_decision import ExecutionDecision
from src.agent.models.final_response import FinalResponse
from src.agent.models.request_context import RequestContext
from src.agent.models.task_profile import TaskProfile
from src.agent.request_router import RouteDecision


# ─────────────────────────────────────────────────────────────────
# 辅助工厂
# ─────────────────────────────────────────────────────────────────

def _make_final_response(answer: str = "测试答案") -> FinalResponse:
    return FinalResponse(
        answer=answer,
        summary=answer,
        route="direct_task",
        task_type="smalltalk",
    )


def _route_decision(route: str, task_type: str, matched_skills=None) -> RouteDecision:
    return RouteDecision(
        route=route,
        task_type=task_type,
        confidence=0.9,
        reason="test",
        matched_skills=matched_skills or [],
        expected_output_schema="generic_answer_v1",
    )


def _exec_decision(mode: str) -> ExecutionDecision:
    return ExecutionDecision(mode=mode, reason="test")


def _exec_context(task_type: str, *, legacy_route: str = "direct_task") -> ExecutionContext:
    return ExecutionContext(
        profile=TaskProfile(
            task_type=task_type,
            complexity="low",
            openness="low",
            tool_need="none",
            legacy_route=legacy_route,
            confidence=0.9,
            reason="test",
        ),
        request=RequestContext(
            query="测试",
            chat_history="",
            user_profile="",
            use_long_term_memory=True,
        ),
    )


def _mock_synthesizer(answer: str = "合成答案") -> MagicMock:
    synth = MagicMock()
    fr = _make_final_response(answer)
    synth.synthesize_smalltalk.return_value = fr
    synth.synthesize_direct.return_value = fr
    synth.synthesize_qa.return_value = fr
    synth.synthesize.return_value = fr
    return synth


def _make_engine(*, react_executor=None) -> ExecutionEngine:
    skill_mgr = MagicMock()
    rag = MagicMock()
    llm = MagicMock()
    synth = _mock_synthesizer()
    from src.agent.response_synthesizer import ResponseSynthesizer
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine._direct = DirectExecutor(
        skill_manager=skill_mgr, rag_retriever=rag, llm=llm, synthesizer=synth
    )
    engine._planned = PlannedExecutor(
        skill_manager=skill_mgr, llm=llm, synthesizer=synth
    )
    engine._react = react_executor or ReactExecutor()
    return engine


# ─────────────────────────────────────────────────────────────────


class TestExecutionEngineStructure:
    """验证 ExecutionEngine 的基本结构。"""

    def test_has_direct_executor(self):
        engine = _make_engine()
        assert isinstance(engine.direct, DirectExecutor)

    def test_has_planned_executor(self):
        engine = _make_engine()
        assert isinstance(engine.planned, PlannedExecutor)

    def test_has_react_executor(self):
        engine = _make_engine()
        assert isinstance(engine.react, ReactExecutor)

    def test_full_construction(self):
        skill_mgr = MagicMock()
        rag = MagicMock()
        llm = MagicMock()
        engine = ExecutionEngine(
            skill_manager=skill_mgr,
            rag_retriever=rag,
            llm=llm,
        )
        assert isinstance(engine, ExecutionEngine)

    def test_invalid_mode_raises(self):
        engine = _make_engine()
        decision = ExecutionDecision(mode="direct", reason="r")
        bad_decision = ExecutionDecision.__new__(ExecutionDecision)
        object.__setattr__(bad_decision, "mode", "unknown")
        object.__setattr__(bad_decision, "reason", "r")
        object.__setattr__(bad_decision, "fallback_modes", [])
        object.__setattr__(bad_decision, "legacy_execution_path", "unknown")
        rd = _route_decision("direct_task", "smalltalk")
        with pytest.raises((ValueError, NotImplementedError)):
            asyncio.get_event_loop().run_until_complete(
                engine.run(bad_decision, rd, "query")
            )


class TestDirectExecutorViaEngine:
    """通过 ExecutionEngine 调用 direct 模式。"""

    def test_direct_smalltalk(self):
        engine = _make_engine()
        ed = _exec_decision("direct")
        rd = _route_decision("direct_task", "smalltalk")
        result = asyncio.get_event_loop().run_until_complete(
            engine.run(ed, rd, "你好", context=_exec_context("smalltalk"))
        )
        assert isinstance(result, FinalResponse)
        event_types = [event["type"] for event in result.execution_events]
        assert event_types[:3] == ["task_profile", "route_decided", "execution_decision"]
        assert "answer_ready" in event_types
        engine.direct._synthesizer.synthesize_smalltalk.assert_called_once()

    def test_direct_simple_qa(self):
        synth = _mock_synthesizer("qa答案")
        rag = MagicMock()
        rag.retrieve.return_value = {"context": "天文知识"}
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="天文答案")

        skill_mgr = MagicMock()
        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine._direct = DirectExecutor(skill_manager=skill_mgr, rag_retriever=rag, llm=llm, synthesizer=synth)
        engine._planned = PlannedExecutor(skill_manager=skill_mgr, llm=llm, synthesizer=synth)
        engine._react = ReactExecutor()

        ed = _exec_decision("direct")
        rd = _route_decision("direct_task", "simple_qa")
        result = asyncio.get_event_loop().run_until_complete(
            engine.run(ed, rd, "什么是黑洞")
        )
        assert isinstance(result, FinalResponse)


class TestPlannedExecutorViaEngine:
    """通过 ExecutionEngine 调用 planned 模式。"""

    def test_planned_returns_final_response(self):
        synth = _mock_synthesizer("观测建议")
        skill_mgr = MagicMock()
        llm = MagicMock()

        from src.agent.models.execution_plan import ExecutionPlan, PlanStep
        from src.agent.models.skill_result import SkillResult

        plan = ExecutionPlan(
            task_type="observation_recommendation",
            output_schema="observation_answer_v1",
            steps=[
                PlanStep(id="s1", kind="tool", title="天气", skill="weather-lookup")
            ],
        )
        mock_planner = MagicMock()
        mock_planner.plan.return_value = plan

        skill_result = SkillResult(
            skill_name="weather-lookup",
            success=True,
            summary="北京今天晴",
            data={"weather": "晴"},
        )
        from src.agent.executor import ExecutionOutcome
        outcome = ExecutionOutcome(plan=plan, skill_results=[skill_result], step_results=[])

        async def mock_wf_execute(*args, **kwargs):
            return outcome

        mock_wf_executor = MagicMock()
        mock_wf_executor.execute = mock_wf_execute

        mock_fallback = MagicMock()
        mock_fallback.version = "fallback_v2"
        mock_fallback.decide_for_execution.return_value = None

        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine._direct = DirectExecutor(skill_manager=skill_mgr, rag_retriever=MagicMock(), llm=llm, synthesizer=synth)
        engine._planned = PlannedExecutor(
            skill_manager=skill_mgr, llm=llm, synthesizer=synth,
            planner=mock_planner, workflow_executor=mock_wf_executor, fallback_policy=mock_fallback,
        )
        engine._react = ReactExecutor()

        ed = _exec_decision("planned")
        rd = _route_decision("planned_task", "observation_recommendation", ["weather-lookup"])
        result = asyncio.get_event_loop().run_until_complete(
            engine.run(
                ed,
                rd,
                "北京今晚观测条件",
                context=_exec_context(
                    "observation_recommendation",
                    legacy_route="planned_task",
                ),
            )
        )
        assert isinstance(result, FinalResponse)
        event_types = [event["type"] for event in result.execution_events]
        assert "plan_created" in event_types
        assert "answer_ready" in event_types
        synth.synthesize.assert_called_once()


class TestReactExecutor:
    """ReactExecutor 独立测试。"""

    def test_react_executor_construction(self):
        mock_agent = MagicMock()
        re = ReactExecutor(agent_executor=mock_agent)
        assert re.ensure_executor() is mock_agent

    def test_react_executor_factory(self):
        mock_agent = MagicMock()
        factory = MagicMock(return_value=mock_agent)
        re = ReactExecutor(agent_executor_factory=factory)
        result = re.ensure_executor()
        assert result is mock_agent
        factory.assert_called_once()

    def test_react_executor_no_config_raises(self):
        re = ReactExecutor()
        with pytest.raises(ValueError, match="react agent executor is not configured"):
            re.ensure_executor()

    def test_astream_events_delegates(self):
        events = [{"event": "on_tool_start", "data": {}}]

        async def mock_astream(*args, **kwargs):
            for e in events:
                yield e

        mock_agent = MagicMock()
        mock_agent.astream_events = mock_astream
        re = ReactExecutor(agent_executor=mock_agent)

        collected = []
        async def collect():
            async for e in re.astream_events({"input": "你好"}):
                collected.append(e)

        asyncio.get_event_loop().run_until_complete(collect())
        assert len(collected) == 1
        assert collected[0]["event"] == "on_tool_start"

    def test_react_run_uses_invoke_when_available(self):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Final Answer: 统一 react 答案"}
        engine = _make_engine(react_executor=ReactExecutor(agent_executor=mock_agent))
        ed = _exec_decision("react")
        rd = _route_decision("fallback_react", "open_domain_reasoning")
        result = asyncio.get_event_loop().run_until_complete(
            engine.run(
                ed,
                rd,
                "帮我写小说",
                context=_exec_context(
                    "open_domain_reasoning",
                    legacy_route="fallback_react",
                ),
            )
        )
        assert isinstance(result, FinalResponse)
        assert result.answer == "统一 react 答案"
        assert result.route == "fallback_react"
        event_types = [event["type"] for event in result.execution_events]
        assert event_types[:3] == ["task_profile", "route_decided", "execution_decision"]
        assert "answer_ready" in event_types

    def test_react_run_falls_back_to_stream_aggregation(self):
        class _AgentStub:
            async def astream_events(self, agent_input, version="v1"):
                yield {
                    "event": "on_tool_start",
                    "data": {"name": "weather-lookup", "input": "北京天气"},
                    "run_id": "react-tool-1",
                }
                yield {
                    "event": "on_tool_end",
                    "data": {"name": "weather-lookup", "output": "晴朗"},
                    "run_id": "react-tool-1",
                }
                yield {
                    "event": "on_llm_stream",
                    "data": {"chunk": MagicMock(content="Thought: 正在推理\n")},
                    "run_id": "react-1",
                }
                yield {
                    "event": "on_llm_stream",
                    "data": {"chunk": MagicMock(content="Final Answer: 流式 react 答案")},
                    "run_id": "react-1",
                }

        engine = _make_engine(react_executor=ReactExecutor(agent_executor=_AgentStub()))
        ed = _exec_decision("react")
        rd = _route_decision("fallback_react", "open_domain_reasoning")
        result = asyncio.get_event_loop().run_until_complete(
            engine.run(
                ed,
                rd,
                "帮我写小说",
                context=_exec_context(
                    "open_domain_reasoning",
                    legacy_route="fallback_react",
                ),
            )
        )
        assert isinstance(result, FinalResponse)
        assert result.answer == "流式 react 答案"
        event_types = [event["type"] for event in result.execution_events]
        assert "tool_called" in event_types
        assert "tool_result" in event_types
        assert "answer_ready" in event_types


class TestDirectExecutorUnit:
    """DirectExecutor 单元测试。"""

    def _make_direct(self) -> DirectExecutor:
        skill_mgr = MagicMock()
        rag = MagicMock()
        rag.retrieve.return_value = {"context": ""}
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="答案")
        synth = _mock_synthesizer()
        return DirectExecutor(skill_manager=skill_mgr, rag_retriever=rag, llm=llm, synthesizer=synth)

    def test_unsupported_task_type_raises(self):
        direct = self._make_direct()
        rd = _route_decision("direct_task", "unknown_task_type")
        with pytest.raises(ValueError, match="unsupported direct task type"):
            asyncio.get_event_loop().run_until_complete(
                direct.run(rd, "query")
            )

    def test_smalltalk_reply_greet(self):
        direct = self._make_direct()
        reply = direct._smalltalk_reply("你好")
        assert "天文" in reply or "查询" in reply

    def test_smalltalk_reply_thanks(self):
        direct = self._make_direct()
        reply = direct._smalltalk_reply("谢谢")
        assert "不客气" in reply


class TestLegacyPathUnchanged:
    """验证旧路径 (TaskOrchestrator) 在新增 ExecutionEngine 后仍完整可用。"""

    def test_task_orchestrator_direct_still_works(self):
        from src.agent.task_orchestrator import TaskOrchestrator

        synth = _mock_synthesizer()
        planner = MagicMock()
        executor = MagicMock()
        fallback = MagicMock()
        fallback.version = "fallback_v2"

        orch = TaskOrchestrator(
            skill_manager=MagicMock(),
            rag_retriever=MagicMock(),
            llm=MagicMock(),
            response_synthesizer=synth,
            planner=planner,
            executor=executor,
            fallback_policy=fallback,
        )
        rd = _route_decision("direct_task", "smalltalk")

        result = asyncio.get_event_loop().run_until_complete(
            orch.run(rd, "你好", chat_history="", user_profile="")
        )
        assert isinstance(result, FinalResponse)
        synth.synthesize_smalltalk.assert_called_once()

    def test_task_orchestrator_unsupported_route_raises(self):
        from src.agent.task_orchestrator import TaskOrchestrator

        orch = TaskOrchestrator(
            skill_manager=MagicMock(),
            rag_retriever=MagicMock(),
            llm=MagicMock(),
        )
        rd = _route_decision("fallback_react", "open_domain_reasoning")
        with pytest.raises(ValueError, match="unsupported orchestrated route"):
            asyncio.get_event_loop().run_until_complete(
                orch.run(rd, "query", chat_history="", user_profile="")
            )
