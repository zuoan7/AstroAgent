"""
Phase 2 ExecutionContext 测试
目标：验证 RequestContext / ExecutionContext / ExecutionResult 构造与适配行为。

当前状态：三个模型已稳定，但主路径仍通过散参调用 TaskOrchestrator.run()。
          from_legacy_params() / from_final_response() 仅被 _run_orchestrated_path()
          旁路记录块调用，不参与主执行逻辑。
收敛计划：待 UnifiedExecutionEngine 实现后，ExecutionContext 成为主路径统一输入，
          ExecutionResult 成为统一输出，legacy adapter 方法降为可选。
"""
from __future__ import annotations

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.models.request_context import RequestContext
from src.agent.models.execution_context import ExecutionContext
from src.agent.models.execution_result import ExecutionResult
from src.agent.models.final_response import FinalResponse
from src.agent.models.task_profile import TaskProfile


class TestRequestContextConstruction:
    """验证 RequestContext 构造与字段访问。"""

    def test_minimal_construction(self):
        ctx = RequestContext(query="你好")
        assert ctx.query == "你好"
        assert ctx.chat_history == ""
        assert ctx.user_profile == ""
        assert ctx.use_long_term_memory is True
        assert len(ctx.request_id) == 8

    def test_request_id_auto_generated(self):
        a = RequestContext(query="q1")
        b = RequestContext(query="q2")
        assert a.request_id != b.request_id

    def test_explicit_request_id(self):
        ctx = RequestContext(query="q", request_id="abc12345")
        assert ctx.request_id == "abc12345"

    def test_from_legacy_params(self):
        ctx = RequestContext.from_legacy_params(
            "帮我查天气",
            chat_history="上一轮对话",
            user_profile="北京用户",
            request_id="req00001",
        )
        assert ctx.query == "帮我查天气"
        assert ctx.chat_history == "上一轮对话"
        assert ctx.user_profile == "北京用户"
        assert ctx.request_id == "req00001"

    def test_to_dict_has_all_fields(self):
        ctx = RequestContext(query="q", chat_history="h", user_profile="p")
        d = ctx.to_dict()
        assert {"query", "chat_history", "user_profile", "request_id", "use_long_term_memory"}.issubset(d.keys())
        assert d["query"] == "q"

    def test_use_long_term_memory_default_true(self):
        ctx = RequestContext(query="q")
        assert ctx.use_long_term_memory is True

    def test_use_long_term_memory_can_be_false(self):
        ctx = RequestContext(query="q", use_long_term_memory=False)
        assert ctx.use_long_term_memory is False


class TestExecutionContextConstruction:
    """验证 ExecutionContext 的构造、便捷属性与 from_legacy_params 工厂。"""

    def _make_profile(self, route="direct_task", task_type="smalltalk") -> TaskProfile:
        return TaskProfile.from_legacy_route(
            route=route, task_type=task_type, confidence=0.9
        )

    def _make_request(self, query="你好") -> RequestContext:
        return RequestContext(query=query, chat_history="ch", user_profile="up")

    def test_basic_construction(self):
        ctx = ExecutionContext(
            profile=self._make_profile(),
            request=self._make_request(),
        )
        assert ctx.query == "你好"
        assert ctx.task_type == "smalltalk"
        assert ctx.legacy_route == "direct_task"
        assert ctx.chat_history == "ch"
        assert ctx.user_profile == "up"

    def test_request_id_delegated_to_request(self):
        req = self._make_request()
        ctx = ExecutionContext(profile=self._make_profile(), request=req)
        assert ctx.request_id == req.request_id

    def test_to_dict_structure(self):
        ctx = ExecutionContext(
            profile=self._make_profile(),
            request=self._make_request(),
        )
        d = ctx.to_dict()
        assert "profile" in d
        assert "request" in d
        assert "extra_meta" in d
        assert d["profile"]["task_type"] == "smalltalk"
        assert d["request"]["query"] == "你好"

    def test_from_legacy_params_direct_task(self):
        ctx = ExecutionContext.from_legacy_params(
            route="direct_task",
            task_type="smalltalk",
            confidence=0.98,
            query="你好",
            chat_history="历史",
            user_profile="偏好",
        )
        assert ctx.query == "你好"
        assert ctx.task_type == "smalltalk"
        assert ctx.legacy_route == "direct_task"
        assert ctx.chat_history == "历史"
        assert ctx.profile.complexity == "low"
        assert ctx.profile.tool_need == "none"

    def test_from_legacy_params_planned_task(self):
        ctx = ExecutionContext.from_legacy_params(
            route="planned_task",
            task_type="observation_recommendation",
            confidence=0.82,
            query="帮我看北京今晚观测条件",
            matched_skills=["weather-lookup", "observation-planner"],
        )
        assert ctx.legacy_route == "planned_task"
        assert ctx.profile.tool_need == "multi"
        assert ctx.profile.complexity == "high"

    def test_from_legacy_params_fallback_react(self):
        ctx = ExecutionContext.from_legacy_params(
            route="fallback_react",
            task_type="open_domain_reasoning",
            confidence=0.45,
            query="帮我写科幻小说",
        )
        assert ctx.legacy_route == "fallback_react"
        assert ctx.profile.openness == "high"
        assert ctx.profile.tool_need == "none"

    def test_extra_meta_empty_by_default(self):
        ctx = ExecutionContext(
            profile=self._make_profile(),
            request=self._make_request(),
        )
        assert ctx.extra_meta == {}

    def test_extra_meta_can_be_set(self):
        ctx = ExecutionContext(
            profile=self._make_profile(),
            request=self._make_request(),
            extra_meta={"debug": True},
        )
        assert ctx.extra_meta["debug"] is True


class TestExecutionResultConstruction:
    """验证 ExecutionResult 的构造与便捷属性。"""

    def _make_final_response(self) -> FinalResponse:
        return FinalResponse(
            answer="北京今天晴",
            summary="天气查询结果",
            sources=[{"title": "高德天气", "snippet": "晴"}],
            tools_used=[{"tool": "weather-lookup", "status": "success"}],
            execution_trace=[{"step_id": "s1", "status": "success"}],
            route="direct_task",
            task_type="single_tool_lookup",
        )

    def test_from_final_response(self):
        fr = self._make_final_response()
        er = ExecutionResult.from_final_response(fr, execution_path="direct_task")
        assert er.answer == "北京今天晴"
        assert er.execution_path == "direct_task"
        assert len(er.execution_trace) == 1
        assert er.sources == fr.sources
        assert er.tools_used == fr.tools_used

    def test_execution_trace_copied_from_final_response(self):
        fr = self._make_final_response()
        er = ExecutionResult.from_final_response(fr)
        assert er.execution_trace == fr.execution_trace

    def test_default_execution_path_unknown(self):
        fr = self._make_final_response()
        er = ExecutionResult.from_final_response(fr)
        assert er.execution_path == "unknown"

    def test_raw_artifacts_empty_by_default(self):
        fr = self._make_final_response()
        er = ExecutionResult.from_final_response(fr)
        assert er.raw_artifacts == {}

    def test_raw_artifacts_can_be_set(self):
        fr = self._make_final_response()
        er = ExecutionResult.from_final_response(fr, raw_artifacts={"img": "url"})
        assert er.raw_artifacts["img"] == "url"

    def test_to_dict_structure(self):
        fr = self._make_final_response()
        er = ExecutionResult.from_final_response(fr, execution_path="direct_task")
        d = er.to_dict()
        assert {"response", "execution_trace", "raw_artifacts", "execution_path"}.issubset(d.keys())
        assert d["execution_path"] == "direct_task"
        assert d["response"]["answer"] == "北京今天晴"

    def test_direct_construction(self):
        fr = self._make_final_response()
        er = ExecutionResult(
            response=fr,
            execution_trace=[{"step_id": "x"}],
            execution_path="planned_task",
        )
        assert er.answer == "北京今天晴"
        assert len(er.execution_trace) == 1
        assert er.execution_path == "planned_task"


class TestPhase2Integration:
    """验证三个模型的联合构造（集成场景）。"""

    def test_full_context_chain_direct_task(self):
        """模拟直接任务的完整上下文链构建。"""
        ctx = ExecutionContext.from_legacy_params(
            route="direct_task",
            task_type="single_tool_lookup",
            confidence=0.9,
            query="帮我查北京天气",
            chat_history="之前聊过观测",
            user_profile="北京，爱好天文",
            matched_skills=["weather-lookup"],
        )
        fr = FinalResponse(
            answer="北京今天晴转多云",
            summary="天气信息",
            execution_trace=[],
        )
        er = ExecutionResult.from_final_response(fr, execution_path=ctx.legacy_route)

        assert ctx.task_type == "single_tool_lookup"
        assert ctx.profile.tool_need == "single"
        assert er.execution_path == "direct_task"
        assert er.answer == "北京今天晴转多云"

    def test_context_and_result_consistent_route(self):
        """ExecutionContext.legacy_route 与 ExecutionResult.execution_path 应一致。"""
        ctx = ExecutionContext.from_legacy_params(
            route="planned_task",
            task_type="observation_recommendation",
            confidence=0.82,
            query="帮我看北京今晚观测",
            matched_skills=["weather-lookup", "observation-planner"],
        )
        fr = FinalResponse(answer="适合观测", summary="obs", execution_trace=[])
        er = ExecutionResult.from_final_response(fr, execution_path=ctx.legacy_route)

        assert ctx.legacy_route == er.execution_path == "planned_task"
