# Agent Mode and DAG Refactor

更新时间：2026-05-20

本文说明本轮 agent 主模式与 planned 执行链路的变更。目标是把原先的 plan-and-solve 过渡实现收敛为真正的 DAG 执行模式，同时保留外部协议和旧入口兼容。

## 模式语义

Agent 现在区分两层概念：

- 策略模式：`AGENT_MODE=auto|react|planned`
- 执行路径：`ExecutionDecision.mode=direct|planned|react`

`hybrid` 仍可配置，但只作为 `auto` 的兼容别名。`AgentExecutionPolicy.to_dict()` 会同时输出原始 `mode` 和 `effective_mode`，便于审计旧配置。

默认策略仍是自动分流：

- `direct`：闲聊、稳定知识、无需工具或单工具低复杂度查询
- `planned`：多工具、观测计划、需要可审计步骤和证据聚合的任务
- `react`：开放式问题，或结构化 planned 路径失败后的兜底策略

## Planned DAG

planned 主路径现在是：

```text
Planner.plan_graph()
  -> WorkflowGraph
  -> WorkflowExecutor.execute()
  -> EvidenceAggregator
  -> ResponseSynthesizer
```

`ExecutionPlan` 仍保留为前端展示、旧序列化和兼容输入输出视图，但不再驱动 planned 主执行语义。

DAG 支持：

- 显式依赖：`PlanStep.depends_on`
- 兼容并行组：`PlanStep.parallel_group`
- 图层调度：`WorkflowGraph.topological_generations()`
- 有界并发：`WorkflowExecutor` 按 `AGENT_MAX_PARALLELISM` 分批执行 ready 节点
- 步骤重试：`PlanStep.retry_policy`
- 步骤级失败策略：`PlanStep.fallback_strategy`

## 失败策略

支持的 `fallback_strategy`：

- `halt`：必需步骤默认策略；失败后 planned 路径停止，并记录 `react_fallback`
- `continue`：可选步骤默认策略；失败后继续执行可运行节点，最终给出 partial answer 元数据
- `skip_dependents`：当前步骤失败后跳过下游依赖分支，但允许无关分支继续
- `react_fallback`：必需步骤失败后停止 planned 路径，并明确标记需要 ReAct 兜底

依赖处理规则：

- 依赖成功：下游节点可执行
- 可选依赖失败且策略为 `continue`：下游节点可继续执行
- 必需依赖失败、依赖被跳过、或策略为 `skip_dependents`：下游节点标记为 `skipped`

## 证据聚合

`WorkflowExecutor` 会为每个 DAG 节点汇总证据：

- `ExecutionOutcome.evidence_by_key`
- `ExecutionOutcome.evidence_items`
- `FinalResponse.structured_payload["dag_evidence"]`
- `FinalResponse.audit_metadata["dag_evidence"]`
- `FinalResponse.audit_metadata["dag_evidence_keys"]`

证据 key 优先级：

1. `PlanStep.evidence_key`
2. `WorkflowNode.output_key`
3. `step_id`

答案合成器会把聚合证据注入最终 synthesis prompt。启用确定性工具合成时，证据仍会进入 `structured_payload` 和审计元数据。

## 兼容层变化

本轮清理了低风险兼容分支：

- `ENABLE_WORKFLOW_GRAPH` 不再切换 planned 主路径，只保留为历史配置兼容字段
- `PlannedExecutor` 固定优先使用 `Planner.plan_graph()`
- `WorkflowExecutor` 取代 planned 主路径中的线性执行

仍保留的兼容层：

- `RequestRouter.route()` / `RouteDecision`
- `AgentExecutionPolicy.choose_path()`
- `TaskOrchestrator`
- `ExecutionPlan`
- `StepExecutor`
- 旧前端事件名：`route_decision`、`plan_update`、`step_start`、`step_end`、`final_answer`

## 规则整理

`ToolNecessityGate` 中重复的禁用工具集合已收敛为常量，观测计划关键词去除了重复项。本轮没有大幅改写路由规则，以避免影响 benchmark 行为；后续适合把规则继续拆成“安全/澄清/实时数据/稳定知识”四类独立 matcher。

## 验证范围

本轮新增和更新的重点测试：

- `tests/unit/test_workflow_executor_dag.py`
- `tests/unit/test_workflow_graph_phase5.py`
- `tests/unit/test_dag_agent_phase8.py`
- `tests/unit/test_execution_trace_phase7.py`
- `tests/unit/test_latency_optimization.py`
- `tests/unit/test_tool_necessity_gate.py`
- `tests/unit/test_tool_necessity_router_integration.py`

建议 smoke：

```bash
pytest tests/unit/test_workflow_executor_dag.py tests/unit/test_workflow_graph_phase5.py -q
pytest tests/unit/test_dag_agent_phase8.py tests/unit/test_execution_trace_phase7.py -q
python scripts/evaluation/evaluate_tool_routing_static.py --suite smoke --limit 12
```
