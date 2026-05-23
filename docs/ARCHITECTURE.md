# AstroAgent 架构说明（当前执行主路径）

## 概览

AstroAgent 是基于 DAG 执行图的天文领域多工具 Agent，支持三种执行路径（direct / planned / react）并通过统一 trace/event 模型向前端输出流式事件。

---

## 主路径

```
用户请求
  ↓
StreamingService.generate_events() / generate_sse()
  ↓
_generate_internal_events()
  ├─ RequestRouter.profile() → TaskProfile
  ├─ RouteDecision compatibility adapter（仅旧链路/事件透传）
  ├─ AgentExecutionPolicy.decide() → ExecutionDecision
  ↓
_run_orchestrated_path()
  └─ ExecutionDecision(mode=direct|planned|react) → ExecutionEngine.run_context()
        ├─ direct → DirectExecutor.run_context()
        ├─ planned → Planner.plan_graph_for_profile()
        │             WorkflowGraph → WorkflowExecutor.execute()
        └─ react → ReactExecutor.run_context() / astream_events_context()
  ↓
FinalResponse.execution_events / execution_trace
  ↓
FrontendExecutionEventAdapter → 旧前端事件序列
```

## 执行引擎

| 组件 | 职责 | 文件 |
|------|------|------|
| `ExecutionEngine` | 统一入口，按 mode 分发 | `src/agent/execution/engine.py` |
| `DirectExecutor` | smalltalk / single_tool_lookup / simple_qa | `src/agent/execution/direct_executor.py` |
| `PlannedExecutor` | plan → execute(DAG) → synthesize | `src/agent/execution/planned_executor.py` |
| `ReactExecutor` | react 非流式执行 + 原始事件代理 | `src/agent/execution/react_executor.py` |
| `WorkflowExecutor` | 按 DAG 依赖并发执行 ready 节点，处理 retry / failure strategy / evidence aggregation | `src/agent/execution/workflow_executor.py` |

---

## 工作流图

`WorkflowGraph` 是执行计划的 DAG 表示：

- 节点（`WorkflowNode`）对应 `PlanStep`
- 有向边（`WorkflowEdge`）表示依赖关系
- `from_execution_plan()` 将兼容 `ExecutionPlan` 转换为 DAG，支持 `depends_on` 和 `parallel_group`
- `topological_order()` / `topological_generations()` 使用 Kahn 算法，支持循环检测和并发分层
- 当前 `WorkflowExecutor` 为 `PlannedExecutor` 的唯一执行引擎

---

## 统一 Trace/Event（Phase 7 引入，Phase 8 开启）

| 模型 | 职责 | 文件 |
|------|------|------|
| `ExecutionTraceEntry` | 统一三种路径的步骤执行结果 | `src/agent/models/execution_trace_entry.py` |
| `ExecutionEvent` | 内部主事件协议，StreamingService 优先消费 | `src/agent/models/execution_event.py` |
| `StepExecutionResult.to_trace_entry()` | 从旧结果对象转换为 trace | `src/agent/executor.py` |
| `FrontendExecutionEventAdapter.emit_response_execution_events()` | ExecutionEvent → 旧前端事件适配层 | `src/agent/frontend_event_adapter.py` |
| `FrontendExecutionEventAdapter.emit_trace_events()` | trace → 旧前端事件兼容回退 | `src/agent/frontend_event_adapter.py` |

前端事件名兼容映射（不变）：

| 内部类型 | 前端事件名 |
|----------|-----------|
| `route_decided` | `route_decision` |
| `plan_built` | `plan_update` |
| `plan_created` | `plan_update` |
| `step_started` | `step_start` |
| `step_finished` | `step_end` |
| `answer_ready` | `final_answer` |
| `tool_called` | `tool_start` |
| `tool_result` / `tool_returned` | `tool_end` |

---

## 兼容层与 deprecated 接口

兼容层的权威清单、测试覆盖与删除条件见 [Agent Compatibility Matrix](agent_compatibility_matrix.md)。

| 接口 | 状态 | 收敛计划 |
|------|------|---------|
| `RouteDecision` | 长期兼容输出 | 继续保留给外部 API / 测试 / 旧执行入口 |
| `RequestRouter.route()` | **deprecated** 兼容入口 | 内部已改走 `profile()`，外部暂不删除 |
| `AgentExecutionPolicy.choose_path()` | **deprecated** 兼容入口 | 内部已改走 `decide()`，外部暂不删除 |
| `ExecutionPlan` | 兼容计划表示 | planned 主路径已 graph-first，暂不删除 |
| `FinalResponse.execution_trace: list[dict]` | 兼容层（未升级为 List[ExecutionTraceEntry]） | 后续可再收口到结构化 trace |
| `ExecutionEvent.to_frontend_type()` 内联 MAP | 兼容层 | Phase 10 迁入 FrontendJsonEventAdapter |
| 旧前端事件名（`route_decision`/`plan_update`/`step_start`/`step_end`/`final_answer`） | 长期兼容输出 | 继续由 StreamingService adapter 输出 |
| `SkillParamBuilder` | direct/planned 共享技能参数构造器 | 已稳定，无收敛需求 |

---

## Feature Flags

flags 的权威状态以 [Agent Compatibility Matrix](agent_compatibility_matrix.md) 为准。

| Flag | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_LLM_INTENT_FALLBACK` | False | 低置信路由/天文边界场景启用 LLM 意图分类兜底 |
| `ENABLE_LLM_PLANNER_FALLBACK` | False | 模板/规则无法生成计划时启用 LLM 计划兜底 |
| `ENABLE_REACT_FALLBACK` | True | planned 失败时允许进入 ReAct 兜底 |
| `ENABLE_PLANNER` | False | legacy governance 策略仍读取的旧开关，不控制 planned DAG 主路径 |

---

## 测试覆盖

| 测试文件 | 覆盖内容 |
|---------|---------|
| `tests/regression/compat/test_dag_agent_compatibility.py` | removed flags、ExecutionEngine 新路径、engine 缺失失败、legacy 边界 |
| `tests/regression/compat/test_execution_trace_compat.py` | ExecutionTraceEntry、ExecutionEvent、FrontendExecutionEventAdapter |
| `test_workflow_executor_dag.py` | WorkflowExecutor DAG 依赖、并发、retry、失败策略、证据聚合 |
| `tests/regression/compat/test_workflow_graph_compat.py` | WorkflowGraph DAG 构建、拓扑排序 |
| `tests/regression/compat/test_execution_decision_compat.py` | ExecutionDecision 模型 |
| `tests/regression/compat/test_execution_context_compat.py` | ExecutionContext 统一上下文 |
| `tests/regression/compat/test_task_profile_compat.py` | TaskProfile 任务画像 |
| `tests/regression/compat/test_agent_compatibility_baseline.py` | 各阶段兼容基准回归 |
