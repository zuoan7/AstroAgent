"""
重构阶段 0 基线测试
目标：建立关键路径行为快照，为后续重构提供可回滚基准。
本文件不测试任何新功能，只记录现有行为。
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()
sys.modules.pop("src.agent.streaming_service", None)

from src.agent.governance import AgentExecutionPolicy
from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.models.final_response import FinalResponse
from src.agent.models.skill_result import SkillResult
from src.agent.request_router import RequestRouter, RouteDecision
from src.agent.task_orchestrator import TaskOrchestrator
from src.agent.streaming_service import StreamingService
from src.core.config import settings


# ─── 共用 Stub ───────────────────────────────────────────────────────────────

class _MemoryStub:
    session_id = "baseline_session"

    def build_context(self, request):
        return {"context_text": ""}

    def append_message(self, request):
        pass

    def append_tool_call(self, request):
        pass


class _RagStub:
    def retrieve(self, query, fast_mode=True):
        return {"context": ""}


class _LLMStub:
    def invoke(self, prompt):
        return SimpleNamespace(content="stubbed_answer")


class _SkillManagerStub:
    def call_skill(self, name, **params):
        return SkillResult(
            skill_name=name,
            success=True,
            data={"params": params},
            summary=f"{name} result",
            sources=[{"source_id": name, "kind": "tool_output", "title": name}],
        )


# ─── Task 1: RequestRouter 路由分支覆盖 ──────────────────────────────────────

class TestRequestRouterBaseline:
    """覆盖 RequestRouter 关键路由分支，记录现有行为。"""

    def setup_method(self):
        self.router = RequestRouter()

    def test_smalltalk_routes_to_direct_task(self):
        decision = self.router.route("你好")
        assert decision.route == "direct_task"
        assert decision.task_type == "smalltalk"
        assert decision.confidence >= 0.9

    def test_smalltalk_thanks_routes_to_direct_task(self):
        decision = self.router.route("谢谢")
        assert decision.route == "direct_task"
        assert decision.task_type == "smalltalk"

    def test_simple_qa_routes_to_direct_task(self):
        decision = self.router.route("黑洞是什么")
        assert decision.route == "direct_task"
        assert decision.task_type == "simple_qa"

    def test_simple_qa_explain_hint(self):
        decision = self.router.route("请解释恒星的生命周期原理")
        assert decision.route == "direct_task"
        assert decision.task_type == "simple_qa"

    def test_single_skill_simple_routes_to_direct_task(self):
        decision = self.router.route("帮我查一下北京今天天气怎么样")
        assert decision.route == "direct_task"
        assert decision.task_type == "single_tool_lookup"
        assert "weather-lookup" in decision.matched_skills

    def test_multi_skill_routes_to_planned_task(self):
        decision = self.router.route("帮我看下北京今晚观测条件，同时查查有没有天象活动")
        assert decision.route == "planned_task"
        assert len(decision.matched_skills) >= 1

    def test_complex_single_skill_routes_to_planned_task(self):
        # 复杂问题（有分析/步骤词），即使只匹配一个 skill 也应升级为 planned_task
        decision = self.router.route("请分析本周北京的天气对深空摄影有哪些影响，并给出详细方案")
        assert decision.route == "planned_task"

    def test_open_ended_routes_to_fallback_react(self):
        decision = self.router.route("帮我写一篇关于宇宙的科幻小说")
        assert decision.route == "fallback_react"
        assert decision.task_type == "open_domain_reasoning"

    def test_unclassified_routes_to_fallback_react(self):
        # 短语，没有任何命中 hint/skill
        decision = self.router.route("哈哈哈")
        assert decision.route == "fallback_react"

    def test_route_decision_has_expected_fields(self):
        decision = self.router.route("今晚能看流星雨吗")
        meta = decision.to_meta()
        assert "route" in meta
        assert "task_type" in meta
        assert "route_confidence" in meta
        assert "route_reason" in meta
        assert "matched_skills" in meta
        assert "expected_output_schema" in meta

    def test_is_direct_task_property(self):
        decision = self.router.route("你好")
        assert decision.is_direct_task is True
        assert decision.is_planned_task is False
        assert decision.is_fallback_react is False

    def test_is_planned_task_property(self):
        decision = self.router.route("帮我看下北京今晚观测条件，同时查查有没有天象活动")
        assert decision.is_planned_task is True

    def test_is_fallback_react_property(self):
        decision = self.router.route("帮我写一篇关于宇宙的科幻小说")
        assert decision.is_fallback_react is True


# ─── Task 2: AgentExecutionPolicy.choose_path 覆盖 ───────────────────────────

class TestAgentExecutionPolicyBaseline:
    """覆盖 choose_path 关键分支。"""

    def test_direct_task_route_returns_direct(self):
        policy = AgentExecutionPolicy(mode="hybrid", enable_react_fallback=True)
        assert policy.choose_path("direct_task") == "direct"

    def test_planned_task_route_returns_planned(self):
        policy = AgentExecutionPolicy(mode="hybrid", enable_react_fallback=True)
        assert policy.choose_path("planned_task") == "planned"

    def test_fallback_react_with_flag_true_returns_react(self):
        policy = AgentExecutionPolicy(mode="hybrid", enable_react_fallback=True)
        assert policy.choose_path("fallback_react") == "react"

    def test_fallback_react_with_flag_false_and_planner_enabled_returns_planned(self):
        policy = AgentExecutionPolicy(mode="hybrid", enable_planner=True, enable_react_fallback=False)
        assert policy.choose_path("fallback_react") == "planned"

    def test_fallback_react_with_flag_false_no_planner_returns_direct(self):
        policy = AgentExecutionPolicy(mode="hybrid", enable_planner=False, enable_react_fallback=False)
        assert policy.choose_path("fallback_react") == "direct"

    def test_react_mode_always_returns_react(self):
        policy = AgentExecutionPolicy(mode="react")
        for route in ("direct_task", "planned_task", "fallback_react", None):
            assert policy.choose_path(route) == "react"

    def test_planned_mode_returns_planned_for_none_route(self):
        policy = AgentExecutionPolicy(mode="planned", enable_react_fallback=False)
        assert policy.choose_path(None) == "planned"

    def test_hybrid_mode_none_route_with_react_fallback(self):
        policy = AgentExecutionPolicy(mode="hybrid", enable_react_fallback=True)
        assert policy.choose_path(None) == "react"

    def test_hybrid_mode_none_route_no_react_fallback(self):
        policy = AgentExecutionPolicy(mode="hybrid", enable_react_fallback=False)
        assert policy.choose_path(None) == "direct"

    def test_from_settings_reads_defaults(self):
        policy = AgentExecutionPolicy.from_settings()
        assert policy.mode in ("react", "hybrid", "planned")
        assert isinstance(policy.enable_react_fallback, bool)


# ─── Task 3: TaskOrchestrator.run 覆盖 ───────────────────────────────────────

class TestTaskOrchestratorBaseline:
    """覆盖 direct_task 和 planned_task 两条路径。"""

    def setup_method(self):
        self.orchestrator = TaskOrchestrator(
            skill_manager=_SkillManagerStub(),
            rag_retriever=_RagStub(),
            llm=_LLMStub(),
        )

    @pytest.mark.asyncio
    async def test_direct_task_smalltalk_returns_final_response(self):
        decision = RouteDecision(
            route="direct_task",
            task_type="smalltalk",
            confidence=0.98,
            reason="matched_smalltalk_pattern",
        )
        result = await self.orchestrator.run(
            decision, "你好", chat_history="", user_profile=""
        )
        assert isinstance(result, FinalResponse)
        assert result.answer
        assert result.route == "direct_task"
        assert result.task_type == "smalltalk"

    @pytest.mark.asyncio
    async def test_direct_task_simple_qa_returns_final_response(self):
        decision = RouteDecision(
            route="direct_task",
            task_type="simple_qa",
            confidence=0.8,
            reason="matched_simple_qa_hint",
        )
        result = await self.orchestrator.run(
            decision, "黑洞是什么", chat_history="", user_profile=""
        )
        assert isinstance(result, FinalResponse)
        assert result.answer
        assert result.task_type == "simple_qa"

    @pytest.mark.asyncio
    async def test_direct_task_single_tool_returns_final_response(self):
        decision = RouteDecision(
            route="direct_task",
            task_type="single_tool_lookup",
            confidence=0.9,
            reason="matched_single_skill",
            matched_skills=["weather-lookup"],
        )
        result = await self.orchestrator.run(
            decision, "北京今天天气", chat_history="", user_profile=""
        )
        assert isinstance(result, FinalResponse)
        assert result.answer
        assert result.task_type == "single_tool_lookup"

    @pytest.mark.asyncio
    async def test_planned_task_returns_final_response(self):
        plan = ExecutionPlan(
            task_type="observation_recommendation",
            output_schema="observation_answer_v1",
            steps=[
                PlanStep(
                    id="weather_context",
                    kind="tool",
                    title="查询天气",
                    skill="weather-lookup",
                ),
                PlanStep(
                    id="observation_plan",
                    kind="tool",
                    title="生成计划",
                    skill="observation-planner",
                ),
            ],
        )
        decision = RouteDecision(
            route="planned_task",
            task_type="observation_recommendation",
            confidence=0.82,
            reason="matched_multiple_skills",
            matched_skills=["weather-lookup", "observation-planner"],
            expected_output_schema="observation_answer_v1",
        )
        result = await self.orchestrator.run(
            decision,
            "今晚北京适合观测吗",
            chat_history="",
            user_profile="",
            execution_plan=plan,
        )
        assert isinstance(result, FinalResponse)
        assert result.answer
        assert result.route == "planned_task"
        assert result.task_type == "observation_recommendation"
        # execution_trace 必须存在且包含步骤
        assert isinstance(result.execution_trace, list)
        assert len(result.execution_trace) >= 1

    @pytest.mark.asyncio
    async def test_unsupported_route_raises(self):
        decision = RouteDecision(
            route="fallback_react",
            task_type="open_domain_reasoning",
            confidence=0.45,
            reason="fallback",
        )
        with pytest.raises(ValueError, match="unsupported orchestrated route"):
            await self.orchestrator.run(
                decision, "随便", chat_history="", user_profile=""
            )


# ─── Task 4: StreamingService 最小集成测试 ─────────────────────────────────────

class TestStreamingServiceBaseline:
    """验证 route_decision 事件与 planned 路径的 plan_update 事件。"""

    def _make_service(self, decision: RouteDecision, plan: ExecutionPlan) -> StreamingService:
        async def fake_run(dec, query, **kwargs):
            return FinalResponse(
                answer="今晚适合观测猎户座。",
                summary="今晚适合观测猎户座。",
                tools_used=[],
                sources=[],
                confidence=0.88,
                route=dec.route,
                task_type=dec.task_type,
                execution_plan=plan.to_dict(),
                execution_trace=[
                    {
                        "step_id": "weather_context",
                        "title": "查询天气",
                        "status": "success",
                        "skill": "weather-lookup",
                        "summary": "天气良好",
                        "latency_ms": 10.0,
                        "sources": [],
                    }
                ],
            )

        return StreamingService(
            agent_executor=None,
            memory=_MemoryStub(),
            user_id="test_user",
            request_router=SimpleNamespace(route=lambda q: decision),
            task_orchestrator=SimpleNamespace(
                build_execution_plan=lambda *a, **kw: plan,
                run=fake_run,
            ),
        )

    @pytest.mark.asyncio
    async def test_streaming_emits_route_decision_event(self):
        decision = RouteDecision(
            route="direct_task",
            task_type="smalltalk",
            confidence=0.98,
            reason="matched_smalltalk_pattern",
        )
        plan = ExecutionPlan(
            task_type="smalltalk",
            output_schema="chat_answer_v1",
            steps=[],
        )
        service = self._make_service(decision, plan)

        events = []
        async for event in service.generate_events("你好"):
            events.append(event)

        route_events = [e for e in events if e.get("type") == "route_decision"]
        assert len(route_events) >= 1
        rd = route_events[0]
        assert rd["route"] == "direct_task"
        assert rd["task_type"] == "smalltalk"

    @pytest.mark.asyncio
    async def test_planned_path_emits_plan_update(self):
        plan = ExecutionPlan(
            task_type="observation_recommendation",
            output_schema="observation_answer_v1",
            steps=[
                PlanStep(
                    id="weather_context",
                    kind="tool",
                    title="查询天气",
                    skill="weather-lookup",
                ),
            ],
        )
        decision = RouteDecision(
            route="planned_task",
            task_type="observation_recommendation",
            confidence=0.82,
            reason="matched_multiple_skills",
            matched_skills=["weather-lookup"],
            expected_output_schema="observation_answer_v1",
        )
        service = self._make_service(decision, plan)

        events = []
        async for event in service.generate_events("今晚北京适合观测吗"):
            events.append(event)

        plan_updates = [e for e in events if e.get("type") == "plan_update"]
        assert len(plan_updates) >= 1

        # planned 路径应有来自 execution_trace 的 step_start 事件
        step_start_ids = [
            e.get("step_id") for e in events if e.get("type") == "step_start"
        ]
        assert "weather_context" in step_start_ids


# ─── Task 5: 最小行为基线快照 ───────────────────────────────────────────────────

class TestMinimalBehaviorBaseline:
    """记录关键数据结构的最小字段约束，作为重构不可破坏的基线。"""

    def test_route_decision_to_meta_has_required_keys(self):
        d = RouteDecision(
            route="direct_task",
            task_type="smalltalk",
            confidence=0.98,
            reason="test",
            matched_skills=["skill-a"],
            expected_output_schema="chat_answer_v1",
        )
        meta = d.to_meta()
        required_keys = {
            "route",
            "task_type",
            "route_confidence",
            "route_reason",
            "matched_skills",
            "expected_output_schema",
        }
        assert required_keys.issubset(set(meta.keys()))
        assert meta["route"] == "direct_task"
        assert meta["task_type"] == "smalltalk"
        assert meta["route_confidence"] == 0.98
        assert meta["matched_skills"] == ["skill-a"]

    def test_final_response_to_dict_has_required_keys(self):
        resp = FinalResponse(
            answer="test answer",
            summary="test",
            route="direct_task",
            task_type="smalltalk",
            confidence=0.98,
        )
        d = resp.to_dict()
        required_keys = {"answer", "summary", "sources", "tools_used", "confidence", "route", "task_type"}
        assert required_keys.issubset(set(d.keys()))
        assert d["answer"] == "test answer"
        assert d["route"] == "direct_task"

    def test_execution_trace_step_has_required_keys(self):
        from src.agent.executor import StepExecutionResult
        step = StepExecutionResult(
            step_id="s1",
            title="step 1",
            kind="tool",
            status="success",
            skill="weather-lookup",
            latency_ms=42.0,
        )
        d = step.to_dict()
        required_keys = {"step_id", "title", "kind", "status", "skill", "latency_ms"}
        assert required_keys.issubset(set(d.keys()))
        assert d["step_id"] == "s1"
        assert d["status"] == "success"
        assert d["latency_ms"] == 42.0

    def test_final_response_execution_trace_is_list(self):
        resp = FinalResponse(
            answer="x",
            summary="x",
            execution_trace=[
                {"step_id": "s1", "status": "success"},
                {"step_id": "s2", "status": "error"},
            ],
        )
        assert isinstance(resp.execution_trace, list)
        assert len(resp.execution_trace) == 2

    def test_route_decision_schema_default(self):
        d = RouteDecision(
            route="direct_task",
            task_type="simple_qa",
            confidence=0.8,
            reason="test",
        )
        assert d.expected_output_schema == "generic_answer_v1"

    def test_route_decision_schema_from_task_type_mapping(self):
        from src.agent.request_router import TASK_TYPE_TO_OUTPUT_SCHEMA
        router = RequestRouter()
        d = router.route("你好")
        assert d.expected_output_schema == TASK_TYPE_TO_OUTPUT_SCHEMA.get("smalltalk", "generic_answer_v1")


# ─── Task 6: feature flags 默认值验证 ─────────────────────────────────────────

class TestFeatureFlagsDefaults:
    """验证 DAG 重构配置位的默认值与兼容语义。"""

    def test_enable_task_profile_default_false(self):
        # Deprecated config: TaskProfile 已固定为 Router 主输出
        assert settings.ENABLE_TASK_PROFILE is False

    def test_enable_execution_context_default_false(self):
        # Deprecated config: ExecutionContext 已固定进入主链路
        assert settings.ENABLE_EXECUTION_CONTEXT is False

    def test_enable_execution_decision_default_false(self):
        # Deprecated config: ExecutionDecision 已固定为 Policy 主输出
        assert settings.ENABLE_EXECUTION_DECISION is False

    def test_enable_unified_execution_engine_default_true(self):
        # Compatibility flag: True 为默认主路径，False 回退 legacy TaskOrchestrator
        assert settings.ENABLE_UNIFIED_EXECUTION_ENGINE is True

    def test_enable_workflow_graph_default_true(self):
        # Compatibility flag: True 优先 plan_graph，False 回退 legacy plan()->graph
        assert settings.ENABLE_WORKFLOW_GRAPH is True

    def test_enable_unified_execution_trace_default_true(self):
        # Deprecated config: 仅保留历史配置位，不再切换主路径
        assert settings.ENABLE_UNIFIED_EXECUTION_TRACE is True

    def test_enable_unified_execution_events_default_true(self):
        # Deprecated config: 仅保留历史配置位，不再切换主路径
        assert settings.ENABLE_UNIFIED_EXECUTION_EVENTS is True

    def test_existing_flags_unaffected(self):
        # 确保原有 flag 默认值未被改动
        assert settings.ENABLE_STRUCTURED_SKILL_RESULT is False
        assert settings.ENABLE_PLANNER is False
        assert settings.ENABLE_REACT_FALLBACK is True
        assert settings.AGENT_MODE == "hybrid"
