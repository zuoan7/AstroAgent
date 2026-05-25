# Agent Compatibility Matrix

更新时间：2026-05-25

本文档基于当前源码与测试引用情况整理，目标是明确 DAG Agent 重构过程中仍保留的兼容层、它们的替代方案、保留原因与删除条件。本文档不表示“已批准删除”，只有在源码引用、外部依赖和测试基线都收口后，兼容层才可进入删除阶段。

权威性约定：

- 主路径与兼容层判断以 `src/agent/`、`src/core/config.py`、`tests/unit/` 当前引用为准
- 如与旧设计文档冲突，以当前源码和本文档为准
- 已无运行分支的 deprecated config 已在收敛阶段移除；其余对象仍需按引用、外部依赖和测试基线逐项收口

## 主路径摘要

当前内部主路径已经收口为：

- `RequestRouter.profile(query) -> TaskProfile`
- `TaskProfile.capability_hints -> CapabilitySelector -> CapabilityDecision`
- `AgentExecutionPolicy.decide(profile, context) -> ExecutionDecision`
- `ExecutionEngine.run_context(decision, context, ...)`
- planned 路径：`Planner.plan_graph_for_profile() -> WorkflowGraph -> WorkflowExecutor.execute()`；`WorkflowExecutor` 支持 DAG 依赖、有界并发、步骤失败策略与证据聚合
- atomic tool 选择：`ToolSelector` 只输出 `capability_kind=tool`，参数由 `CapabilityParamBuilder` / atomic adapter 构造
- 前端兼容事件：`FrontendExecutionEventAdapter`

## Skill / Tool Convergence Matrix

| 对象 | 当前状态 | 新替代方案 | 当前保留原因 | 删除条件 / 后续动作 |
| --- | --- | --- | --- | --- |
| `TaskProfile.capability_hints` | `primary semantics` | 无 | 内部路由、planner、capability selector 的主能力提示字段 | 固定保留 |
| `TaskProfile.matched_skills` | `legacy alias` | `capability_hints` | 旧 API、旧测试和前端元信息仍读取该字段；现在可能镜像 atomic tool hint | 外部调用方迁移到 `capability_hints` 后再删除或仅保留输出 DTO |
| `RouteDecision.capability_hints` | `compat bridge` | `TaskProfile.capability_hints` | legacy `RouteDecision` 仍需携带主能力语义，用于外部兼容 adapter 与旧元信息输出 | `RouteDecision` 退场时同步删除 |
| `get_nasa_apod` / `web_search` 的 `SkillSpec` 身份 | `removed` | `ToolRegistry` / `ToolDefinition` atomic tool specs | 两者不再是 logical skill；通过 `CapabilityKit.call_tool()` 或 ReAct atomic adapter 调用 | 保持不回归到 skill registry |
| `weather-lookup` | `high-level skill` | 底层 allowed tool 为 `get_weather` | 仍承载“观测天气”领域语义，不等同于裸 MCP tool | 若未来产品决定只保留 atomic weather，再单独迁移 |
| `CapabilityKit.to_langchain_tools()` atomic 暴露 | `compat adapter` | high-level skills + selected atomic tools from `ToolRegistry` | ReAct 仍需要 LangChain Tool 入口；trace adapter 会映射回 `capability_kind` | ReAct 选择层重构完成后再评估 |

## Compatibility Matrix

| 对象 | 当前状态 | 新替代方案 | 当前保留原因 | 是否仍被测试覆盖 | 删除条件 | 建议删除阶段 |
| --- | --- | --- | --- | --- | --- | --- |
| `RequestRouter.route(query)` | `deprecated compatibility` | `RequestRouter.profile(query)` | 外部兼容 API；无 `profile()` 的旧调用方；旧 stream/event 元信息 | 是。`tests/regression/compat/test_task_profile_compat.py`、`tests/regression/compat/test_agent_compatibility_baseline.py`、`tests/unit/test_latency_optimization.py` | 不再有外部调用方依赖 `RouteDecision`；`StreamingService` 不再需要 legacy route 元信息 | 与 `RouteDecision` 同步评估 |
| `RouteDecision` | `compatibility` | `TaskProfile` + `ExecutionDecision` | legacy route 元信息载体；`ExecutionEngine` legacy adapter；旧事件输出；测试基线 | 是。大量路由、执行引擎、基线测试仍直接构造 | `ExecutionEngine` legacy adapter 与 stream metadata 不再消费它；外部 API 完成迁移 | 与 `route()` 同步评估 |
| `AgentExecutionPolicy.choose_path(route)` | `deprecated compatibility` | `AgentExecutionPolicy.decide(profile, context)` | 外部调用方仍可能需要 `direct/planned/react` 字符串；老测试仍锁定旧返回值 | 是。`tests/regression/compat/test_execution_decision_compat.py`、`tests/regression/compat/test_agent_compatibility_baseline.py`、`tests/regression/compat/test_agent_governance_baseline.py` | 无生产代码和外部调用依赖旧字符串接口；兼容测试退场 | `ExecutionDecision` 外部消费稳定后 |
| `TaskOrchestrator` | `removed internal legacy` | `ExecutionEngine` | 不再保留运行 shim；旧文件已删除 | 是。`tests/regression/compat/test_dag_agent_compatibility.py`、`tests/unit/test_context_first_execution_contract.py` 覆盖删除边界 | 已完成 | 已完成 |
| `TaskOrchestrator.build_execution_plan()` | `removed internal legacy` | `ExecutionEngine.preview_plan_context()` / `Planner.plan_graph_for_profile()` | legacy orchestrator planned 展示入口已删除 | 是。删除边界由 context-first contract 覆盖 | 已完成 | 已完成 |
| `Planner.plan()` | `deprecated compatibility` | `Planner.plan_graph_for_profile()` | 旧调用方仍需要 `ExecutionPlan` 视图；部分 planner/graph 测试仍覆盖兼容计划 | 是。`tests/regression/compat/test_workflow_graph_compat.py`、`tests/unit/test_workflow_executor_dag.py` | 不再需要 `ExecutionPlan` 兼容输出；旧调用方迁移完成 | `ExecutionPlan` 降为纯输出 DTO 后再评估 |
| `ExecutionPlan` | `compatibility representation` | `WorkflowGraph` | 旧序列化；展示层 plan 视图；`FinalResponse.execution_plan`；graph/plan 双向兼容 | 是。workflow/planner/streaming/engine 多组测试 | 前端/API/审计不再输出 plan 兼容结构；plan fallback 删除 | 最后批次，晚于 `Planner.plan()` |
| `StepExecutor` | `removed internal legacy` | `WorkflowExecutor` | 线性 `ExecutionPlan` executor 已删除；planned 能力执行、失败、retry、并行由 `WorkflowExecutor` 覆盖 | 是。`tests/regression/compat/test_dag_agent_compatibility.py`、`tests/unit/test_context_first_execution_contract.py` 覆盖删除边界 | 已完成 | 已完成 |
| DAG 重构 flags | `removed` | 固定 context-first 主路径 | 旧 DAG 灰度 env var 均不再声明；由 settings `extra=ignore` 忽略 | 是。`tests/regression/compat/test_dag_agent_compatibility.py` | 已完成 | 已完成 |
| 旧前端事件适配逻辑 | `compatibility output` | 内部主事件协议 `ExecutionEvent`；适配器为 `FrontendExecutionEventAdapter` | 前端协议仍要求 `route_decision` / `plan_update` / `step_start` / `step_end` / `final_answer` 等旧事件名；不要求前端同步升级 | 是。`tests/regression/compat/test_execution_trace_compat.py`、`tests/unit/test_latency_optimization.py`、`tests/regression/compat/test_agent_compatibility_baseline.py` | 前端/调用方完成新事件协议迁移；不再要求旧事件名稳定输出 | 前端协议升级后 |

## DAG Flags Matrix

| Flag | 当前状态 | 新替代方案 | 当前保留原因 | 是否仍被测试覆盖 | 删除条件 | 建议删除阶段 |
| --- | --- | --- | --- | --- | --- | --- |
| `ENABLE_TASK_PROFILE` | `removed` | `RequestRouter.profile()` 固定主路径 | 无运行分支 | `tests/regression/compat/test_dag_agent_compatibility.py` 覆盖字段已退场 | 已完成 |
| `ENABLE_EXECUTION_CONTEXT` | `removed` | `ExecutionContext` 固定主路径 | 无运行分支 | `tests/regression/compat/test_dag_agent_compatibility.py` 覆盖字段已退场 | 已完成 |
| `ENABLE_EXECUTION_DECISION` | `removed` | `AgentExecutionPolicy.decide()` 固定主路径 | 无运行分支 | `tests/regression/compat/test_dag_agent_compatibility.py` 覆盖字段已退场 | 已完成 |
| `ENABLE_WORKFLOW_GRAPH` | `removed` | `Planner.plan_graph()` 固定主路径 | 无运行分支 | `tests/regression/compat/test_dag_agent_compatibility.py` 覆盖字段已退场 | 已完成 |
| `ENABLE_UNIFIED_EXECUTION_TRACE` | `removed` | unified trace 固定开启 | 无运行分支 | `tests/regression/compat/test_dag_agent_compatibility.py` 覆盖字段已退场 | 已完成 |
| `ENABLE_UNIFIED_EXECUTION_EVENTS` | `removed` | unified events 固定开启 | 无运行分支 | `tests/regression/compat/test_dag_agent_compatibility.py` 覆盖字段已退场 | 已完成 |

## 关键引用依据

以下不是完整 grep 清单，只列影响删除判断的关键位置：

- `RequestRouter.route()` / `RouteDecision`
  - 生产代码：`src/agent/request_router.py`、`src/agent/streaming_service.py`、`src/agent/execution/engine.py`
  - 测试：`tests/regression/compat/test_task_profile_compat.py`、`tests/regression/compat/test_agent_compatibility_baseline.py`
- `choose_path()`
  - 生产代码仅定义于 `src/agent/governance.py`
  - 测试仍覆盖：`tests/regression/compat/test_execution_decision_compat.py`、`tests/regression/compat/test_agent_compatibility_baseline.py`、`tests/regression/compat/test_agent_governance_baseline.py`
- `TaskOrchestrator` / `StepExecutor`
  - 生产代码：已删除
  - 测试：`tests/regression/compat/test_dag_agent_compatibility.py`、`tests/unit/test_context_first_execution_contract.py` 覆盖删除边界
- `Planner.plan()` / `ExecutionPlan`
  - 生产代码：`src/agent/planner.py`、`src/agent/streaming_service.py`
  - 测试：`tests/regression/compat/test_workflow_graph_compat.py`、`tests/unit/test_workflow_executor_dag.py`、`tests/unit/test_latency_optimization.py`
- 前端兼容事件
  - 生产代码：`src/agent/frontend_event_adapter.py`、`src/agent/streaming_service.py`
  - 测试：`tests/regression/compat/test_execution_trace_compat.py`、`tests/unit/test_latency_optimization.py`

## 维护建议

删除顺序建议：

1. 继续清理 `matched_skills` 的内部引用，只保留外部输出别名
2. 再删 `choose_path()`、`route()` 这类兼容包装
3. 最后评估 `Planner.plan()` / `ExecutionPlan` 这类跨层兼容 DTO

删除前检查：

1. `rg` 确认无生产代码引用
2. `rg` 确认无测试仅为兼容保留
3. 若删除 flag 或改默认值，必须补测试
4. 若外部协议仍依赖旧事件名，不得删除前端兼容适配层
