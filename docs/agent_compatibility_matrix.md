# Agent Compatibility Matrix

更新时间：2026-04-29

本文档基于当前源码与测试引用情况整理，目标是明确 DAG Agent 重构过程中仍保留的兼容层、它们的替代方案、保留原因与删除条件。本文档不表示“已批准删除”，只有在源码引用、外部依赖和测试基线都收口后，兼容层才可进入删除阶段。

权威性约定：

- 主路径与兼容层判断以 `src/agent/`、`src/core/config.py`、`tests/unit/` 当前引用为准
- 如与旧设计文档冲突，以当前源码和本文档为准
- `rg` 搜索结果显示，这些对象当前仍被生产代码、兼容回退或测试基线引用，因此本阶段没有“立即可删除”对象

## 主路径摘要

当前内部主路径已经收口为：

- `RequestRouter.profile(query) -> TaskProfile`
- `AgentExecutionPolicy.decide(profile, context) -> ExecutionDecision`
- `ExecutionEngine.run(decision, legacy_decision, ...)`
- planned 路径：`Planner.plan_graph() -> WorkflowGraph -> WorkflowExecutor.execute()`
- 前端兼容事件：`FrontendExecutionEventAdapter`

## Compatibility Matrix

| 对象 | 当前状态 | 新替代方案 | 当前保留原因 | 是否仍被测试覆盖 | 删除条件 | 建议删除阶段 |
| --- | --- | --- | --- | --- | --- | --- |
| `RequestRouter.route(query)` | `deprecated compatibility` | `RequestRouter.profile(query)` | 外部兼容 API；无 `profile()` 的旧调用方；旧 stream/event 元信息；`TaskOrchestrator` 兼容链路 | 是。`test_task_profile_phase1.py`、`test_refactor_baseline.py`、`test_latency_optimization.py` | 不再有外部调用方依赖 `RouteDecision`；`StreamingService` 不再需要 legacy route 元信息；`TaskOrchestrator` 退场 | `TaskOrchestrator` 清理完成后再评估 |
| `RouteDecision` | `compatibility` | `TaskProfile` + `ExecutionDecision` | legacy route 元信息载体；旧执行器签名；旧事件输出；测试基线 | 是。大量路由、执行引擎、基线测试仍直接构造 | `ExecutionEngine` / executors / stream metadata 不再消费它；外部 API 完成迁移 | 与 `route()` 同步评估 |
| `AgentExecutionPolicy.choose_path(route)` | `deprecated compatibility` | `AgentExecutionPolicy.decide(profile, context)` | 外部调用方仍可能需要 `direct/planned/react` 字符串；老测试仍锁定旧返回值 | 是。`test_execution_decision_phase3.py`、`test_refactor_baseline.py`、`test_agent_governance_phase0.py` | 无生产代码和外部调用依赖旧字符串接口；兼容测试退场 | `ExecutionDecision` 外部消费稳定后 |
| `TaskOrchestrator` | `deprecated compatibility` | `ExecutionEngine` | `ENABLE_UNIFIED_EXECUTION_ENGINE=False` 时回退；engine 未注入时回退；外部 direct/planned 旧入口 | 是。`test_refactor_baseline.py`、`test_execution_engine_phase4.py`、`test_dag_agent_phase8.py`、`test_enterprise_governance_stage4.py` | 统一执行引擎不可回退需求消失；外部旧入口迁移完成；compat 测试退场 | 统一引擎不再需要 fallback 时 |
| `TaskOrchestrator.build_execution_plan()` | `deprecated compatibility` | `ExecutionEngine.preview_plan()` / `Planner.plan_graph()` | legacy planned 展示；legacy orchestrator planned 执行 | 是。`test_execution_engine_phase4.py`、`test_latency_optimization.py` | `StreamingService` 不再存在 legacy orchestrator planned 展示回退；`TaskOrchestrator` 退场 | 与 `TaskOrchestrator` 同步 |
| `Planner.plan()` | `deprecated compatibility` | `Planner.plan_graph()` | 旧调用方仍需要 `ExecutionPlan` 视图；graph fallback 分支仍保留 | 是。`test_workflow_graph_phase5.py`、`test_workflow_executor_phase6.py`、`test_planner_executor_stage3.py` | 不再需要 `ExecutionPlan` 兼容输出；graph fallback 删除；旧调用方迁移完成 | `ExecutionPlan` 降为纯输出 DTO 后再评估 |
| `ExecutionPlan` | `compatibility representation` | `WorkflowGraph` | 旧序列化；展示层 plan 视图；`FinalResponse.execution_plan`；graph/plan 双向兼容；legacy orchestrator | 是。workflow/planner/streaming/engine 多组测试 | 前端/API/审计不再输出 plan 兼容结构；legacy orchestrator 与 plan fallback 删除 | 最后批次，晚于 `Planner.plan()` |
| `StepExecutor` | `deprecated compatibility` | `WorkflowExecutor` | `TaskOrchestrator` 的 legacy planned 执行；历史线性执行测试 | 是。`test_planner_executor_stage3.py`、`test_dag_agent_phase8.py` | `TaskOrchestrator` planned 路径删除；不再需要线性 `ExecutionPlan` 执行 | 与 `TaskOrchestrator` 同步 |
| DAG 重构 flags | 混合：`compatibility flag` / `deprecated config` | 见下方 flags 细分表 | 配置系统兼容；灰度回退；历史观测位 | 是。`test_refactor_baseline.py`、`test_dag_agent_phase8.py` | 见 flags 细分表 | 分对象评估，不一起删除 |
| 旧前端事件适配逻辑 | `compatibility output` | 内部主事件协议 `ExecutionEvent`；适配器为 `FrontendExecutionEventAdapter` | 前端协议仍要求 `route_decision` / `plan_update` / `step_start` / `step_end` / `final_answer` 等旧事件名；不要求前端同步升级 | 是。`test_execution_trace_phase7.py`、`test_latency_optimization.py`、`test_refactor_baseline.py` | 前端/调用方完成新事件协议迁移；不再要求旧事件名稳定输出 | 前端协议升级后 |

## DAG Flags Matrix

| Flag | 当前状态 | 新替代方案 | 当前保留原因 | 是否仍被测试覆盖 | 删除条件 | 建议删除阶段 |
| --- | --- | --- | --- | --- | --- | --- |
| `ENABLE_TASK_PROFILE` | `deprecated config` | `RequestRouter.profile()` 固定主路径 | 配置系统兼容；历史语义保留 | 是。`test_dag_agent_phase8.py` | 无配置/测试再读取；可接受移除旧 config 位 | 与 router compat 清理解耦，可更早删 |
| `ENABLE_EXECUTION_CONTEXT` | `deprecated config` | `ExecutionContext` 固定主路径 | 配置系统兼容；历史语义保留 | 是。`test_dag_agent_phase8.py` | 无配置/测试再读取 | 与 policy compat 清理解耦，可更早删 |
| `ENABLE_EXECUTION_DECISION` | `deprecated config` | `AgentExecutionPolicy.decide()` 固定主路径 | 配置系统兼容；历史语义保留 | 是。`test_dag_agent_phase8.py` | 无配置/测试再读取 | `choose_path()` 外部迁移后可一起删，或更早删 config 位 |
| `ENABLE_UNIFIED_EXECUTION_ENGINE` | `compatibility flag` | 默认主路径仍是 `ExecutionEngine` | 真实控制 unified engine 与 legacy orchestrator 的切换 | 是。`test_dag_agent_phase8.py` | `TaskOrchestrator` fallback 删除 | 与 `TaskOrchestrator` 同步 |
| `ENABLE_WORKFLOW_GRAPH` | `compatibility flag` | 默认主路径仍是 `Planner.plan_graph()` | 真实控制 planned 路径是否回退到 `plan()+from_execution_plan()` | 是。`test_workflow_executor_phase6.py`、`test_dag_agent_phase8.py` | graph fallback 删除；`Planner.plan()` 降为纯兼容输出或移除 | 晚于 planner graph 稳定后 |
| `ENABLE_UNIFIED_EXECUTION_TRACE` | `deprecated config` | unified trace 固定开启 | 历史观测位；配置层兼容 | 是。`test_dag_agent_phase8.py` | 无配置/测试再依赖；trace 输出不再需要旧配置痕迹 | 可在事件/trace 稳定后较早删除 |
| `ENABLE_UNIFIED_EXECUTION_EVENTS` | `deprecated config` | unified events 固定开启 | 历史观测位；配置层兼容 | 是。`test_dag_agent_phase8.py` | 无配置/测试再依赖 | 可在事件适配稳定后较早删除 |

## 关键引用依据

以下不是完整 grep 清单，只列影响删除判断的关键位置：

- `RequestRouter.route()` / `RouteDecision`
  - 生产代码：`src/agent/request_router.py`、`src/agent/streaming_service.py`、`src/agent/task_orchestrator.py`、`src/agent/execution/*`
  - 测试：`test_task_profile_phase1.py`、`test_refactor_baseline.py`、`test_execution_engine_phase4.py`
- `choose_path()`
  - 生产代码仅定义于 `src/agent/governance.py`
  - 测试仍覆盖：`test_execution_decision_phase3.py`、`test_refactor_baseline.py`、`test_agent_governance_phase0.py`
- `TaskOrchestrator` / `StepExecutor`
  - 生产代码：`src/agent/__init__.py`、`src/agent/task_orchestrator.py`、`src/agent/streaming_service.py`
  - 测试：`test_refactor_baseline.py`、`test_execution_engine_phase4.py`、`test_dag_agent_phase8.py`
- `Planner.plan()` / `ExecutionPlan`
  - 生产代码：`src/agent/planner.py`、`src/agent/execution/planned_executor.py`、`src/agent/streaming_service.py`
  - 测试：`test_workflow_graph_phase5.py`、`test_workflow_executor_phase6.py`、`test_latency_optimization.py`
- 前端兼容事件
  - 生产代码：`src/agent/frontend_event_adapter.py`、`src/agent/streaming_service.py`
  - 测试：`test_execution_trace_phase7.py`、`test_latency_optimization.py`

## 维护建议

删除顺序建议：

1. 先删 `deprecated config`，因为它们多数已不再切换主路径
2. 再删 `choose_path()`、`route()` 这类兼容包装
3. 最后清理 `TaskOrchestrator`、`ExecutionPlan`、`StepExecutor` 这类跨层兼容门面

删除前检查：

1. `rg` 确认无生产代码引用
2. `rg` 确认无测试仅为兼容保留
3. 若删除 flag 或改默认值，必须补测试
4. 若外部协议仍依赖旧事件名，不得删除前端兼容适配层
