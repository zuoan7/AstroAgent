from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.agent.execution.engine import ExecutionEngine
from src.agent.execution.react_executor import ReactExecutor
from src.agent.execution.react_trace_adapter import ReactToolTraceAdapter
from src.agent.models.execution_context import ExecutionContext
from src.agent.models.execution_decision import ExecutionDecision
from src.agent.models.final_response import FinalResponse
from src.agent.request_router import RouteDecision


def _route_decision(
    route: str = "fallback_react",
    task_type: str = "open_domain_reasoning",
) -> RouteDecision:
    return RouteDecision(
        route=route,
        task_type=task_type,
        confidence=0.8,
        reason="test",
    )


def test_react_trace_adapter_maps_skill_atomic_and_rag_tools():
    adapter = ReactToolTraceAdapter()

    weather = adapter.build_entry(
        step_id="r1",
        tool_name="WeatherLookup",
        tool_input={"city": "北京"},
        tool_output="北京晴",
    )
    search = adapter.build_entry(
        step_id="r2",
        tool_name="web_search",
        tool_input={"query": "JWST latest"},
        tool_output="搜索结果",
    )
    rag = adapter.build_entry(
        step_id="r3",
        tool_name="RAGRetrieve",
        tool_input="天球",
        tool_output="本地知识",
    )

    assert weather.logical_skill == "weather-lookup"
    assert weather.capability_kind == "skill"
    assert weather.capability_name == "weather-lookup"
    assert weather.expected_mcp_tools == ["get_weather"]
    assert weather.mcp_tools_used == []

    assert search.logical_skill == "web_search"
    assert search.capability_kind == "tool"
    assert search.capability_name == "web_search"
    assert search.expected_mcp_tools == ["web_search"]

    assert rag.logical_skill == "RAGRetrieve"
    assert rag.capability_kind == ""
    assert rag.expected_mcp_tools == []


class _InvokeExecutor:
    def invoke(self, agent_input):
        return {
            "output": "Final Answer: 北京今晚天气晴。",
            "intermediate_steps": [
                (
                    SimpleNamespace(
                        tool="WeatherLookup",
                        tool_input={"city": "北京", "extensions": "all"},
                    ),
                    "北京今晚晴，适合观测。",
                )
            ],
        }


def test_react_executor_invoke_builds_structured_tool_trace():
    executor = ReactExecutor(agent_executor=_InvokeExecutor())

    response = asyncio.run(
        executor.run_context(
            ExecutionContext.from_legacy_decision(
                _route_decision(),
                "北京今晚天气怎么样",
            )
        )
    )

    assert response.answer == "北京今晚天气晴。"
    assert response.execution_trace[0]["logical_skill"] == "weather-lookup"
    assert response.execution_trace[0]["capability_kind"] == "skill"
    assert response.execution_trace[0]["capability_name"] == "weather-lookup"
    assert response.execution_trace[0]["expected_mcp_tools"] == ["get_weather"]
    assert response.tools_used[0]["logical_skill"] == "weather-lookup"
    assert response.audit_metadata["react_tools_used"] == ["weather-lookup"]
    assert response.audit_metadata["react_expected_mcp_tools"] == ["get_weather"]
    assert [event["type"] for event in response.execution_events] == [
        "tool_called",
        "tool_result",
        "answer_ready",
    ]
    assert response.execution_events[0]["payload"]["capability_kind"] == "skill"


def test_execution_engine_react_fallback_merges_react_audit_and_mismatch():
    planned_response = FinalResponse(
        answer="planned failed",
        summary="planned failed",
        route="planned_task",
        task_type="observation_recommendation",
        execution_trace=[
            {
                "step_id": "weather",
                "kind": "tool",
                "status": "error",
                "expected_mcp_tools": ["get_weather"],
            }
        ],
        fallback_path=[
            {
                "strategy": "react_fallback",
                "reason": "required_step_failed",
                "metadata": {"required_failed_steps": ["weather"]},
            }
        ],
        audit_metadata={"expected_mcp_tools": ["get_weather"]},
    )
    react_response = FinalResponse(
        answer="react recovered",
        summary="react recovered",
        route="planned_task",
        task_type="observation_recommendation",
        tools_used=[
            {
                "tool": "web_search",
                "logical_skill": "web_search",
                "expected_mcp_tools": ["web_search"],
            }
        ],
        execution_events=[
            {
                "type": "answer_ready",
                "payload": {"answer": "react recovered"},
                "source": "react",
            }
        ],
        audit_metadata={
            "react_tools_used": ["web_search"],
            "react_mcp_tools_used": ["web_search"],
            "react_expected_mcp_tools": ["web_search"],
            "react_trace_count": 1,
        },
    )

    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine._planned = MagicMock()
    engine._planned.run_context = AsyncMock(return_value=planned_response)
    engine._react = MagicMock()
    engine._react.run_context = AsyncMock(return_value=react_response)

    result = asyncio.run(
        engine.run(
            ExecutionDecision(mode="planned", reason="test"),
            _route_decision("planned_task", "observation_recommendation"),
            "今晚北京观测条件",
        )
    )

    assert result.answer == "react recovered"
    assert result.execution_trace == planned_response.execution_trace
    assert result.tools_used == react_response.tools_used
    assert result.audit_metadata["react_tools_used"] == ["web_search"]
    assert result.audit_metadata["react_expected_mcp_tools"] == ["web_search"]
    assert result.audit_metadata["react_tool_mismatch"] == ["web_search"]
    assert result.audit_metadata["recovery"]["metadata"]["recovery_mode"] == "react"
