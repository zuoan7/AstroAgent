"""Phase 5 WorkflowGraph 测试

目标：验证 WorkflowNode / WorkflowEdge / WorkflowGraph 的构造、序列化、
      拓扑排序、验证，以及从 ExecutionPlan 的线性转换正确性。
"""
from __future__ import annotations

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.models.workflow_graph import WorkflowEdge, WorkflowGraph, WorkflowNode
from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.planner import Planner
from src.agent.request_router import RouteDecision


# ─────────────────────────────────────────────────────────────────
# 辅助工厂
# ─────────────────────────────────────────────────────────────────

def _simple_plan(task_type: str = "observation_recommendation") -> ExecutionPlan:
    return ExecutionPlan(
        task_type=task_type,
        output_schema="observation_answer_v1",
        steps=[
            PlanStep(id="s1", kind="tool", title="天气", skill="weather-lookup", timeout_ms=8000),
            PlanStep(id="s2", kind="tool", title="观测", skill="observation-planner", timeout_ms=12000),
        ],
    )


def _parallel_plan() -> ExecutionPlan:
    return ExecutionPlan(
        task_type="astrophotography_advice",
        output_schema="photo_answer_v1",
        steps=[
            PlanStep(id="p1", kind="tool", title="摄影参数", skill="astrophotography-calculator",
                     parallel_group="img", timeout_ms=15000),
            PlanStep(id="p2", kind="tool", title="摄影天气", skill="weather-lookup",
                     parallel_group="img", required=False, timeout_ms=8000),
            PlanStep(id="p3", kind="tool", title="后续步骤", skill="observation-planner"),
        ],
    )


def _route_decision(
    task_type: str = "observation_recommendation",
    matched_skills=None,
    output_schema: str = "observation_answer_v1",
) -> RouteDecision:
    return RouteDecision(
        route="planned_task",
        task_type=task_type,
        confidence=0.9,
        reason="test",
        matched_skills=matched_skills or [],
        expected_output_schema=output_schema,
    )


# ─────────────────────────────────────────────────────────────────


class TestWorkflowNodeConstruction:
    def test_defaults(self):
        n = WorkflowNode(id="n1")
        assert n.kind == "tool"
        assert n.depends_on == []
        assert n.optional is False
        assert n.output_key is None

    def test_to_dict_keys(self):
        n = WorkflowNode(id="x", title="测试", skill="weather-lookup")
        d = n.to_dict()
        assert set(d.keys()) == {"id", "title", "kind", "skill", "inputs",
                                  "depends_on", "timeout_ms", "optional", "output_key"}

    def test_optional_flag(self):
        n = WorkflowNode(id="opt", optional=True)
        assert n.to_dict()["optional"] is True


class TestWorkflowEdgeConstruction:
    def test_basic(self):
        e = WorkflowEdge(source="a", target="b", label="dep")
        assert e.to_dict() == {"source": "a", "target": "b", "label": "dep"}

    def test_empty_label(self):
        e = WorkflowEdge(source="a", target="b")
        assert e.label == ""


class TestWorkflowGraphBasic:
    def _chain_graph(self) -> WorkflowGraph:
        n1 = WorkflowNode(id="n1", title="步骤1")
        n2 = WorkflowNode(id="n2", title="步骤2", depends_on=["n1"])
        n3 = WorkflowNode(id="n3", title="步骤3", depends_on=["n2"])
        edges = [WorkflowEdge("n1", "n2"), WorkflowEdge("n2", "n3")]
        return WorkflowGraph(nodes=[n1, n2, n3], edges=edges)

    def test_node_lookup(self):
        g = self._chain_graph()
        assert g.node("n2").title == "步骤2"
        assert g.node("missing") is None

    def test_node_ids(self):
        g = self._chain_graph()
        assert set(g.node_ids()) == {"n1", "n2", "n3"}

    def test_roots(self):
        g = self._chain_graph()
        roots = g.roots()
        assert len(roots) == 1
        assert roots[0].id == "n1"

    def test_successors_predecessors(self):
        g = self._chain_graph()
        assert [n.id for n in g.successors("n1")] == ["n2"]
        assert [n.id for n in g.predecessors("n3")] == ["n2"]
        assert g.predecessors("n1") == []

    def test_topological_order(self):
        g = self._chain_graph()
        order = g.topological_order()
        assert [n.id for n in order] == ["n1", "n2", "n3"]

    def test_validate_passes(self):
        g = self._chain_graph()
        assert g.validate() == []

    def test_to_dict_structure(self):
        g = self._chain_graph()
        d = g.to_dict()
        assert "nodes" in d and "edges" in d
        assert len(d["nodes"]) == 3
        assert len(d["edges"]) == 2

    def test_debug_dump_contains_ids(self):
        g = self._chain_graph()
        dump = g.debug_dump()
        assert "n1" in dump and "n2" in dump and "n3" in dump


class TestWorkflowGraphValidation:
    def test_empty_graph_invalid(self):
        g = WorkflowGraph()
        errors = g.validate()
        assert any("节点" in e for e in errors)

    def test_missing_dependency_detected(self):
        n = WorkflowNode(id="n1", depends_on=["nonexistent"])
        g = WorkflowGraph(nodes=[n])
        errors = g.validate()
        assert any("nonexistent" in e for e in errors)

    def test_cycle_detected(self):
        n1 = WorkflowNode(id="a", depends_on=["b"])
        n2 = WorkflowNode(id="b", depends_on=["a"])
        edges = [WorkflowEdge("a", "b"), WorkflowEdge("b", "a")]
        g = WorkflowGraph(nodes=[n1, n2], edges=edges)
        errors = g.validate()
        assert any("循环" in e for e in errors)

    def test_invalid_edge_source(self):
        n = WorkflowNode(id="n1")
        g = WorkflowGraph(nodes=[n], edges=[WorkflowEdge("ghost", "n1")])
        errors = g.validate()
        assert any("ghost" in e for e in errors)


class TestTopologicalOrder:
    def test_parallel_nodes(self):
        n1 = WorkflowNode(id="a")
        n2 = WorkflowNode(id="b")
        g = WorkflowGraph(nodes=[n1, n2], edges=[])
        order = g.topological_order()
        assert len(order) == 2
        assert {n.id for n in order} == {"a", "b"}

    def test_diamond_shape(self):
        # root -> (left, right) -> sink
        root = WorkflowNode(id="root")
        left = WorkflowNode(id="left", depends_on=["root"])
        right = WorkflowNode(id="right", depends_on=["root"])
        sink = WorkflowNode(id="sink", depends_on=["left", "right"])
        edges = [
            WorkflowEdge("root", "left"),
            WorkflowEdge("root", "right"),
            WorkflowEdge("left", "sink"),
            WorkflowEdge("right", "sink"),
        ]
        g = WorkflowGraph(nodes=[root, left, right, sink], edges=edges)
        order = g.topological_order()
        ids = [n.id for n in order]
        assert ids[0] == "root"
        assert ids[-1] == "sink"
        assert set(ids[1:3]) == {"left", "right"}
        assert g.validate() == []


class TestFromExecutionPlan:
    def test_simple_linear_chain(self):
        plan = _simple_plan()
        g = WorkflowGraph.from_execution_plan(plan)
        assert len(g.nodes) == 2
        assert g.output_schema == "observation_answer_v1"

    def test_chain_dependency_order(self):
        plan = _simple_plan()
        g = WorkflowGraph.from_execution_plan(plan)
        n2 = g.node("s2")
        assert "s1" in n2.depends_on
        assert g.node("s1").depends_on == []

    def test_chain_edges(self):
        plan = _simple_plan()
        g = WorkflowGraph.from_execution_plan(plan)
        assert len(g.edges) == 1
        assert g.edges[0].source == "s1"
        assert g.edges[0].target == "s2"

    def test_parallel_group_no_internal_edges(self):
        plan = _parallel_plan()
        g = WorkflowGraph.from_execution_plan(plan)
        # p1 和 p2 同属 parallel_group="img"，互不依赖
        p1 = g.node("p1")
        p2 = g.node("p2")
        assert "p2" not in p1.depends_on
        assert "p1" not in p2.depends_on

    def test_parallel_group_both_depend_on_nothing(self):
        plan = _parallel_plan()
        g = WorkflowGraph.from_execution_plan(plan)
        p1 = g.node("p1")
        p2 = g.node("p2")
        assert p1.depends_on == []
        assert p2.depends_on == []

    def test_post_parallel_depends_on_group(self):
        plan = _parallel_plan()
        g = WorkflowGraph.from_execution_plan(plan)
        p3 = g.node("p3")
        # p3 应依赖 p1 和 p2（整个并行组）
        assert set(p3.depends_on) == {"p1", "p2"}

    def test_optional_flag_propagated(self):
        plan = _parallel_plan()
        g = WorkflowGraph.from_execution_plan(plan)
        p2 = g.node("p2")
        assert p2.optional is True
        p1 = g.node("p1")
        assert p1.optional is False

    def test_metadata_populated(self):
        plan = _simple_plan("observation_recommendation")
        g = WorkflowGraph.from_execution_plan(plan)
        assert g.metadata.get("task_type") == "observation_recommendation"
        assert "planner_version" in g.metadata

    def test_topological_valid_after_convert(self):
        plan = _parallel_plan()
        g = WorkflowGraph.from_execution_plan(plan)
        errors = g.validate()
        assert errors == [], errors

    def test_timeout_propagated(self):
        plan = _simple_plan()
        g = WorkflowGraph.from_execution_plan(plan)
        assert g.node("s1").timeout_ms == 8000
        assert g.node("s2").timeout_ms == 12000

    def test_skill_propagated(self):
        plan = _simple_plan()
        g = WorkflowGraph.from_execution_plan(plan)
        assert g.node("s1").skill == "weather-lookup"

    def test_empty_plan_creates_empty_graph(self):
        plan = ExecutionPlan(task_type="t", output_schema="s", steps=[])
        g = WorkflowGraph.from_execution_plan(plan)
        assert g.nodes == []
        assert g.edges == []

    def test_single_step_no_edges(self):
        plan = ExecutionPlan(
            task_type="simple",
            output_schema="s",
            steps=[PlanStep(id="only", kind="tool", skill="weather-lookup")],
        )
        g = WorkflowGraph.from_execution_plan(plan)
        assert len(g.nodes) == 1
        assert g.edges == []
        assert g.node("only").depends_on == []

    def test_node_from_plan_step_preserves_key_semantics(self):
        step = PlanStep(
            id="weather_context",
            kind="tool",
            title="查询天气条件",
            skill="weather-lookup",
            params={"city": "北京"},
            required=False,
            timeout_ms=8000,
        )
        node = WorkflowGraph.node_from_plan_step(step, depends_on=["prep"])
        assert node.id == "weather_context"
        assert node.title == "查询天气条件"
        assert node.kind == "tool"
        assert node.skill == "weather-lookup"
        assert node.inputs == {"city": "北京"}
        assert node.depends_on == ["prep"]
        assert node.timeout_ms == 8000
        assert node.optional is True
        assert node.output_key == "weather_context"


class TestPlannerGraphPlanning:
    def test_plan_graph_returns_valid_workflow_graph(self):
        planner = Planner()
        graph = planner.plan_graph(
            query="北京今晚适合观测什么",
            route_decision=_route_decision(
                task_type="observation_recommendation",
                matched_skills=["weather-lookup", "observation-planner"],
            ),
        )
        assert isinstance(graph, WorkflowGraph)
        assert graph.validate() == []
        assert graph.output_schema == "observation_answer_v1"
        assert graph.metadata.get("task_type") == "observation_recommendation"
        assert graph.node("weather_context") is not None
        assert graph.node("observation_plan") is not None

    def test_legacy_plan_remains_available(self):
        planner = Planner()
        plan = planner.plan(
            query="北京今晚适合观测什么",
            route_decision=_route_decision(
                task_type="observation_recommendation",
                matched_skills=["weather-lookup", "observation-planner"],
            ),
        )
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) >= 1

    def test_execution_plan_from_workflow_graph_keeps_compatibility(self):
        planner = Planner()
        graph = planner.plan_graph(
            query="今晚用双筒看什么",
            route_decision=_route_decision(
                task_type="observation_recommendation",
                matched_skills=["weather-lookup", "observation-planner"],
            ),
        )
        plan = ExecutionPlan.from_workflow_graph(graph)
        assert isinstance(plan, ExecutionPlan)
        assert plan.task_type == "observation_recommendation"
        assert plan.output_schema == "observation_answer_v1"
        assert [step.id for step in plan.steps] == [node.id for node in graph.topological_order()]
