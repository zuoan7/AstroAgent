"""WorkflowGraph — DAG 化执行图模型（Phase 5 引入）。

将 ExecutionPlan 的线性步骤列表升级为显式 DAG 结构，
支持 depends_on 依赖声明、并行组推导与拓扑排序。

当前状态：Planner.plan_graph() 与 PlannedExecutor 已优先使用 WorkflowGraph；
          ExecutionPlan 保留为兼容表示与旧序列化格式。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkflowNode:
    """DAG 中的单个执行节点，对应原 PlanStep。"""

    id: str
    title: str = ""
    kind: str = "tool"
    skill: Optional[str] = None
    capability_kind: str = ""
    capability_name: str = ""
    operation: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    timeout_ms: Optional[int] = None
    optional: bool = False
    output_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "skill": self.skill,
            "capability_kind": self.capability_kind,
            "capability_name": self.capability_name,
            "operation": self.operation,
            "allowed_tools": list(self.allowed_tools),
            "inputs": dict(self.inputs),
            "depends_on": list(self.depends_on),
            "timeout_ms": self.timeout_ms,
            "optional": self.optional,
            "output_key": self.output_key,
        }


@dataclass
class WorkflowEdge:
    """DAG 中的有向边，从 source 到 target 表示执行依赖。"""

    source: str
    target: str
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target, "label": self.label}


@dataclass
class WorkflowGraph:
    """有向无环图（DAG）执行模型。

    nodes: 节点列表（顺序不代表执行顺序，以 depends_on 为准）
    edges: 显式依赖边（由 depends_on 生成，也可直接构造）
    output_schema: 期望输出 schema（来自 ExecutionPlan）
    metadata: 透传扩展字段
    """

    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)
    output_schema: str = "generic_answer_v1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── 查询 ──────────────────────────────────────────────────────────

    def node(self, node_id: str) -> Optional[WorkflowNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def roots(self) -> List[WorkflowNode]:
        """没有入边的节点（初始可执行节点）。"""
        has_incoming = {e.target for e in self.edges}
        return [n for n in self.nodes if n.id not in has_incoming]

    def successors(self, node_id: str) -> List[WorkflowNode]:
        target_ids = {e.target for e in self.edges if e.source == node_id}
        return [n for n in self.nodes if n.id in target_ids]

    def predecessors(self, node_id: str) -> List[WorkflowNode]:
        source_ids = {e.source for e in self.edges if e.target == node_id}
        return [n for n in self.nodes if n.id in source_ids]

    def dependency_map(self) -> Dict[str, List[str]]:
        """Return normalized node dependencies from depends_on plus explicit edges."""
        deps: Dict[str, set[str]] = {n.id: set(n.depends_on) for n in self.nodes}
        for edge in self.edges:
            deps.setdefault(edge.target, set()).add(edge.source)
            deps.setdefault(edge.source, set())
        return {node_id: sorted(values) for node_id, values in deps.items()}

    def successor_map(self) -> Dict[str, List[str]]:
        """Return normalized successors derived from the normalized dependency map."""
        successors: Dict[str, set[str]] = {n.id: set() for n in self.nodes}
        for target, deps in self.dependency_map().items():
            successors.setdefault(target, set())
            for dep in deps:
                successors.setdefault(dep, set()).add(target)
        return {node_id: sorted(values) for node_id, values in successors.items()}

    # ── 拓扑排序 ──────────────────────────────────────────────────────

    def topological_order(self) -> List[WorkflowNode]:
        """Kahn 算法拓扑排序，返回合法执行顺序；存在环则抛 ValueError。"""
        dependency_map = self.dependency_map()
        in_degree: Dict[str, int] = {n.id: len(dependency_map.get(n.id, [])) for n in self.nodes}
        successors = self.successor_map()

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[str] = []

        while queue:
            queue.sort()  # 保证确定性
            nid = queue.pop(0)
            order.append(nid)
            for succ_id in successors.get(nid, []):
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)

        if len(order) != len(self.nodes):
            raise ValueError(
                f"WorkflowGraph 存在循环依赖，无法拓扑排序: {set(in_degree) - set(order)}"
            )

        node_map = {n.id: n for n in self.nodes}
        return [node_map[nid] for nid in order]

    def topological_generations(self) -> List[List[WorkflowNode]]:
        """Return executable DAG layers; nodes in the same layer are independent."""
        dependency_map = self.dependency_map()
        successors = self.successor_map()
        in_degree: Dict[str, int] = {n.id: len(dependency_map.get(n.id, [])) for n in self.nodes}
        node_map = {n.id: n for n in self.nodes}
        ready = sorted([nid for nid, deg in in_degree.items() if deg == 0])
        generations: List[List[WorkflowNode]] = []
        visited: List[str] = []

        while ready:
            current = ready
            ready = []
            generations.append([node_map[nid] for nid in current])
            visited.extend(current)
            for nid in current:
                for succ_id in successors.get(nid, []):
                    in_degree[succ_id] -= 1
                    if in_degree[succ_id] == 0:
                        ready.append(succ_id)
            ready.sort()

        if len(visited) != len(self.nodes):
            raise ValueError(
                f"WorkflowGraph 存在循环依赖，无法拓扑分层: {set(in_degree) - set(visited)}"
            )
        return generations

    # ── 验证 ──────────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """基础校验，返回错误列表（空表示合法）。"""
        errors: List[str] = []
        ids = set(self.node_ids())

        if not self.nodes:
            errors.append("WorkflowGraph 没有任何节点")

        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in ids:
                    errors.append(f"节点 {node.id!r} 的 depends_on 引用了不存在的节点 {dep!r}")

        for edge in self.edges:
            if edge.source not in ids:
                errors.append(f"边 {edge.source!r} -> {edge.target!r} 的 source 不存在")
            if edge.target not in ids:
                errors.append(f"边 {edge.source!r} -> {edge.target!r} 的 target 不存在")

        try:
            self.topological_order()
        except ValueError as exc:
            errors.append(str(exc))

        return errors

    # ── 序列化 ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_schema": self.output_schema,
            "metadata": dict(self.metadata),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def debug_dump(self) -> str:
        """人类可读的图结构摘要，用于日志调试。"""
        lines = [f"WorkflowGraph(output_schema={self.output_schema!r})"]
        for node in self.nodes:
            deps = f" <- {node.depends_on}" if node.depends_on else ""
            optional_tag = " [optional]" if node.optional else ""
            lines.append(f"  [{node.id}] {node.title or node.skill or node.kind}{deps}{optional_tag}")
        for edge in self.edges:
            label = f" ({edge.label})" if edge.label else ""
            lines.append(f"  {edge.source} --> {edge.target}{label}")
        return "\n".join(lines)

    # ── 工厂方法：从 ExecutionPlan 线性转换 ──────────────────────────

    @staticmethod
    def node_from_plan_step(
        step: Any,
        *,
        depends_on: Optional[List[str]] = None,
    ) -> WorkflowNode:
        """将单个 PlanStep 映射为 WorkflowNode，保留关键语义字段。"""
        return WorkflowNode(
            id=step.id,
            title=step.title,
            kind=step.kind,
            skill=step.skill,
            capability_kind=getattr(step, "capability_kind", "") or (
                "skill" if getattr(step, "skill", None) else ""
            ),
            capability_name=getattr(step, "capability_name", "") or (
                getattr(step, "skill", None) or ""
            ),
            operation=getattr(step, "operation", None),
            allowed_tools=list(getattr(step, "allowed_tools", []) or []),
            inputs=dict(step.params or {}),
            depends_on=list(getattr(step, "depends_on", []) or depends_on or []),
            timeout_ms=step.timeout_ms,
            optional=not step.required,
            output_key=step.id,
        )

    @classmethod
    def from_execution_plan(cls, plan: "ExecutionPlan") -> "WorkflowGraph":  # type: ignore[name-defined]
        """将 ExecutionPlan 线性步骤列表转为 WorkflowGraph。

        转换规则：
        1. 每个 PlanStep -> 一个 WorkflowNode
        2. 同 parallel_group 内的步骤互不依赖（并行）
        3. 不同 group / 无 group 的步骤按顺序形成链式依赖
        4. 无 parallel_group 的步骤视为独占组，顺序依赖前一个组的所有尾节点
        """
        from src.agent.models.execution_plan import ExecutionPlan  # noqa: F811

        nodes: List[WorkflowNode] = []
        edges: List[WorkflowEdge] = []

        # 按 parallel_group 分段，group=None 的步骤各自独立一段
        segments: List[List] = []
        for step in plan.steps:
            if step.parallel_group is None:
                segments.append([step])
            elif segments and segments[-1] and segments[-1][0].parallel_group == step.parallel_group:
                segments[-1].append(step)
            else:
                segments.append([step])

        prev_segment_ids: List[str] = []

        for segment in segments:
            seg_ids: List[str] = []
            for step in segment:
                step_depends_on = list(getattr(step, "depends_on", []) or prev_segment_ids)
                node = cls.node_from_plan_step(
                    step,
                    depends_on=step_depends_on,
                )
                nodes.append(node)
                seg_ids.append(step.id)

                for prev_id in step_depends_on:
                    edges.append(WorkflowEdge(source=prev_id, target=step.id))

            prev_segment_ids = seg_ids

        return cls(
            nodes=nodes,
            edges=edges,
            output_schema=plan.output_schema,
            metadata={
                "_compat_plan": plan.to_dict(),
                "task_type": plan.task_type,
                "planner_type": plan.planner_type,
                "rationale": plan.rationale,
                "planner_version": plan.planner_version,
                "schema_version": plan.schema_version,
                "budget_policy_version": plan.budget_policy_version,
            },
        )
