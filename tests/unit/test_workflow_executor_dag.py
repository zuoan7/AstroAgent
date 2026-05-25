from __future__ import annotations

import asyncio
import time

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.execution.workflow_executor import WorkflowExecutor
from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.skills.result import SkillResult
from src.agent.models.workflow_graph import WorkflowGraph
from src.agent.policies.budget_policy import RequestBudget, RequestBudgetTracker
from src.tools.results import ToolResult


class _CapabilityKitStub:
    def __init__(
        self,
        *,
        delays: dict[str, float] | None = None,
        fail_first: set[str] | None = None,
        always_fail: set[str] | None = None,
    ) -> None:
        self.delays = delays or {}
        self.fail_first = fail_first or set()
        self.always_fail = always_fail or set()
        self.calls: dict[str, int] = {}
        self.tool_calls: list[tuple[str, dict]] = []

    def call_skill(self, skill_name: str, **kwargs):
        self.calls[skill_name] = self.calls.get(skill_name, 0) + 1
        delay = self.delays.get(skill_name, 0)
        if delay:
            time.sleep(delay)
        if skill_name in self.always_fail:
            return SkillResult.from_error(
                skill_name=skill_name,
                error_code="TEST_FAILURE",
                error_message=f"{skill_name} failed",
            )
        if skill_name in self.fail_first and self.calls[skill_name] == 1:
            return SkillResult.from_error(
                skill_name=skill_name,
                error_code="TEST_RETRY",
                error_message=f"{skill_name} retry me",
            )
        return SkillResult(
            skill_name=skill_name,
            success=True,
            data={"skill": skill_name, "params": kwargs},
            summary=f"{skill_name} ok",
            sources=[
                {
                    "source_id": f"src:{skill_name}",
                    "kind": "tool_output",
                    "title": skill_name,
                    "snippet": f"{skill_name} evidence",
                    "tool": skill_name,
                }
            ],
        )

    def call_tool(self, tool_name: str, **kwargs):
        self.tool_calls.append((tool_name, kwargs))
        return ToolResult(
            ok=True,
            tool_name=tool_name,
            data={"tool": tool_name},
        )


def _params(skill_name: str, query: str) -> dict:
    return {"query": query, "skill": skill_name}


def _budget(max_parallelism: int = 2) -> RequestBudgetTracker:
    return RequestBudgetTracker(
        RequestBudget(
            max_llm_calls=4,
            max_tool_calls=10,
            max_total_time_ms=10_000,
            max_parallelism=max_parallelism,
            max_context_chars=6_000,
        )
    )


def test_workflow_executor_runs_ready_dag_nodes_concurrently():
    plan = ExecutionPlan(
        task_type="observation_recommendation",
        output_schema="observation_answer_v1",
        steps=[
            PlanStep(id="root", kind="tool", skill="root-tool"),
            PlanStep(
                id="left",
                kind="tool",
                skill="left-tool",
                depends_on=["root"],
                evidence_key="left_evidence",
            ),
            PlanStep(
                id="right",
                kind="tool",
                skill="right-tool",
                depends_on=["root"],
                evidence_key="right_evidence",
            ),
            PlanStep(
                id="sink",
                kind="tool",
                skill="sink-tool",
                depends_on=["left", "right"],
            ),
        ],
    )
    graph = WorkflowGraph.from_execution_plan(plan)
    manager = _CapabilityKitStub(delays={"left-tool": 0.12, "right-tool": 0.12})
    executor = WorkflowExecutor(capability_kit=manager)

    started = time.perf_counter()
    outcome = asyncio.run(
        executor.execute(
            graph,
            plan,
            query="dag smoke",
            param_builder=_params,
            budget_tracker=_budget(max_parallelism=2),
        )
    )
    elapsed = time.perf_counter() - started

    assert not outcome.halted
    assert [step.step_id for step in outcome.step_results] == [
        "root",
        "left",
        "right",
        "sink",
    ]
    assert elapsed < 0.22
    assert "left_evidence" in outcome.evidence_by_key
    assert "right_evidence" in outcome.evidence_by_key
    assert outcome.evidence_items


def test_workflow_executor_retries_failed_step_before_success():
    plan = ExecutionPlan(
        task_type="celestial_event_analysis",
        output_schema="event_answer_v1",
        steps=[
            PlanStep(
                id="event",
                kind="tool",
                skill="event-tool",
                retry_policy=1,
                evidence_key="event_evidence",
            )
        ],
    )
    manager = _CapabilityKitStub(fail_first={"event-tool"})
    executor = WorkflowExecutor(capability_kit=manager)

    outcome = asyncio.run(
        executor.execute(
            WorkflowGraph.from_execution_plan(plan),
            plan,
            query="retry",
            param_builder=_params,
            budget_tracker=_budget(),
        )
    )

    assert not outcome.halted
    assert outcome.step_results[0].attempts == 2
    assert outcome.step_results[0].status == "success"
    assert manager.calls["event-tool"] == 2


def test_workflow_executor_uses_capability_name_when_skill_field_is_absent():
    plan = ExecutionPlan(
        task_type="single_tool_lookup",
        output_schema="tool_answer_v1",
        steps=[
            PlanStep(
                id="capability_step",
                kind="tool",
                capability_kind="skill",
                capability_name="weather-lookup",
            )
        ],
    )
    manager = _CapabilityKitStub()
    executor = WorkflowExecutor(capability_kit=manager)

    outcome = asyncio.run(
        executor.execute(
            WorkflowGraph.from_execution_plan(plan),
            plan,
            query="北京天气",
            param_builder=_params,
            budget_tracker=_budget(),
        )
    )

    assert not outcome.halted
    assert manager.calls["weather-lookup"] == 1
    assert outcome.step_results[0].capability_name == "weather-lookup"
    assert outcome.step_results[0].skill is None


def test_workflow_executor_does_not_infer_executable_from_node_skill():
    graph = WorkflowGraph(
        nodes=[
            # `skill` is a display/serialization field; executable identity
            # must come from capability_kind/capability_name.
            WorkflowGraph.node_from_plan_step(
                PlanStep(
                    id="legacy_display_only",
                    kind="tool",
                    skill=None,
                )
            )
        ],
        output_schema="tool_answer_v1",
    )
    graph.nodes[0].skill = "weather-lookup"
    graph.nodes[0].capability_kind = ""
    graph.nodes[0].capability_name = ""
    plan = ExecutionPlan(
        task_type="single_tool_lookup",
        output_schema="tool_answer_v1",
        steps=[
            PlanStep(
                id="legacy_display_only",
                kind="tool",
                skill=None,
            )
        ],
    )
    manager = _CapabilityKitStub()
    executor = WorkflowExecutor(capability_kit=manager)

    outcome = asyncio.run(
        executor.execute(
            graph,
            plan,
            query="北京天气",
            param_builder=_params,
            budget_tracker=_budget(),
        )
    )

    assert outcome.halted
    assert manager.calls == {}
    assert outcome.step_results[0].status == "error"


def test_workflow_executor_runs_tool_only_plan_step():
    plan = ExecutionPlan(
        task_type="single_tool_lookup",
        output_schema="tool_answer_v1",
        steps=[
            PlanStep(
                id="weather_tool",
                kind="tool",
                skill=None,
                capability_kind="tool",
                capability_name="get_weather",
                allowed_tools=["get_weather"],
                params={"city": "北京", "extensions": "all"},
            )
        ],
    )
    manager = _CapabilityKitStub()
    executor = WorkflowExecutor(capability_kit=manager)

    outcome = asyncio.run(
        executor.execute(
            WorkflowGraph.from_execution_plan(plan),
            plan,
            query="北京今晚天气怎么样",
            param_builder=_params,
            budget_tracker=_budget(),
        )
    )

    assert not outcome.halted
    assert manager.tool_calls == [
        ("get_weather", {"city": "北京", "extensions": "all"})
    ]
    assert outcome.step_results[0].capability_kind == "tool"
    assert outcome.step_results[0].capability_name == "get_weather"
    assert outcome.step_results[0].expected_mcp_tools == ["get_weather"]


def test_workflow_executor_runs_mixed_skill_and_tool_plan():
    plan = ExecutionPlan(
        task_type="observation_recommendation",
        output_schema="generic_answer_v1",
        steps=[
            PlanStep(
                id="skill_context",
                kind="tool",
                skill="weather-lookup",
                capability_kind="skill",
                capability_name="weather-lookup",
                params={"city": "北京", "extensions": "all"},
                parallel_group="mixed",
            ),
            PlanStep(
                id="search_context",
                kind="tool",
                skill=None,
                capability_kind="tool",
                capability_name="web_search",
                allowed_tools=["web_search"],
                params={"query": "JWST latest", "max_results": 5},
                parallel_group="mixed",
            ),
        ],
    )
    manager = _CapabilityKitStub()
    executor = WorkflowExecutor(capability_kit=manager)

    outcome = asyncio.run(
        executor.execute(
            WorkflowGraph.from_execution_plan(plan),
            plan,
            query="mixed",
            param_builder=_params,
            budget_tracker=_budget(),
        )
    )

    assert not outcome.halted
    assert manager.calls["weather-lookup"] == 1
    assert manager.tool_calls == [
        ("web_search", {"query": "JWST latest", "max_results": 5})
    ]
    assert {step.step_id: step.capability_kind for step in outcome.step_results} == {
        "skill_context": "skill",
        "search_context": "tool",
    }


def test_optional_failure_does_not_block_dependent_continue_strategy():
    plan = ExecutionPlan(
        task_type="observation_recommendation",
        output_schema="observation_answer_v1",
        steps=[
            PlanStep(
                id="optional_weather",
                kind="tool",
                skill="weather-tool",
                required=False,
                fallback_strategy="continue",
            ),
            PlanStep(
                id="answer_context",
                kind="tool",
                skill="planner-tool",
                depends_on=["optional_weather"],
            ),
        ],
    )
    manager = _CapabilityKitStub(always_fail={"weather-tool"})
    executor = WorkflowExecutor(capability_kit=manager)

    outcome = asyncio.run(
        executor.execute(
            WorkflowGraph.from_execution_plan(plan),
            plan,
            query="optional",
            param_builder=_params,
            budget_tracker=_budget(),
        )
    )

    assert not outcome.halted
    assert [step.status for step in outcome.step_results] == ["error", "success"]
    assert outcome.evidence_by_key["optional_weather"]["status"] == "error"


def test_skip_dependents_failure_strategy_skips_downstream_branch():
    plan = ExecutionPlan(
        task_type="observation_recommendation",
        output_schema="observation_answer_v1",
        steps=[
            PlanStep(
                id="branch_root",
                kind="tool",
                skill="bad-tool",
                fallback_strategy="skip_dependents",
                parallel_group="roots",
            ),
            PlanStep(
                id="independent",
                kind="tool",
                skill="independent-tool",
                parallel_group="roots",
            ),
            PlanStep(
                id="downstream",
                kind="tool",
                skill="downstream-tool",
                depends_on=["branch_root"],
            ),
        ],
    )
    manager = _CapabilityKitStub(always_fail={"bad-tool"})
    executor = WorkflowExecutor(capability_kit=manager)

    outcome = asyncio.run(
        executor.execute(
            WorkflowGraph.from_execution_plan(plan),
            plan,
            query="skip",
            param_builder=_params,
            budget_tracker=_budget(),
        )
    )

    statuses = {step.step_id: step.status for step in outcome.step_results}
    assert statuses["branch_root"] == "error"
    assert statuses["downstream"] == "skipped"
    assert statuses["independent"] == "success"
    assert "downstream" in outcome.skipped_step_ids
