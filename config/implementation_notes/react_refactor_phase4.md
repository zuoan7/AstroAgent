# ReAct Refactor 阶段 4 改动说明

> 历史记录：本文保留阶段性实现说明原貌。当前工具层兼容状态以 `docs/agent_compatibility_matrix.md` 为准。

## 目标

阶段 4 参考 `docs/AstroAgent_ReAct_Refactor_Plan.md`，把当前 AstroAgent 从“已有 Planner + Executor 的结构化链路”继续补齐到“可治理、可审计、可限制资源消耗”的企业级形态。

本次改动重点不再是新增主流程，而是给现有主流程补上策略层、预算层、细粒度 fallback、版本化与请求审计。

## 本次主要实现的功能

### 1. 引入模型策略层

新增 `ModelPolicy`，用于按角色区分模型使用层级：

- Router / ParamParser / 轻量总结类场景预留使用小模型策略
- Planner 默认走主模型
- ResponseSynthesizer 支持按配置选择主模型或小模型

在运行时初始化中，`AstroAgent.create_session_runtime()` 已按策略分别构建：

- 主执行模型
- Planner 模型
- Synthesizer 模型

这样后续不需要把所有步骤都绑定在同一个模型上，具备了成本与时延优化入口。

### 2. 引入请求预算控制

新增 `RequestBudget` 和 `RequestBudgetTracker`，实现单请求资源限制与统计，支持以下预算项：

- `max_llm_calls`
- `max_tool_calls`
- `max_total_time_ms`
- `max_parallelism`
- `max_context_chars`

当前接入位置：

- `TaskOrchestrator` 在每次请求开始时创建独立预算追踪器
- `ResponseSynthesizer` 在调用 LLM 前登记上下文长度与 LLM 调用次数
- `TaskOrchestrator` 的简单 QA 主模型调用也会计入预算
- `StepExecutor` 在执行工具步骤和并行步骤时登记工具次数和并行度

当预算超限时，会抛出 `BudgetExceededError`，`StreamingService` 会返回可解释的提前终止信息，而不是静默超时。

### 3. 引入细粒度 fallback 策略

新增 `FallbackPolicy`，把 fallback 从原来的统一“降级”拆分为可区分的策略类型：

- `tool_retry`
- `alternate_tool`
- `cached_answer`
- `partial_answer`
- `web_fallback`
- `react_fallback`

当前主链路已正式接入：

- `TaskOrchestrator` 会根据 `ExecutionOutcome` 判断是否需要记录 fallback path
- 当 planned 执行中可选步骤失败时，会标记为 `partial_answer`
- 当必需步骤失败导致链路中断时，会记录为 `react_fallback`
- `FallbackService` 新增了面向联网搜索的 fallback 分类能力

这样后续线上问题可以明确知道本次请求是“预算中断”“步骤失败后部分回答”还是“回退到 web/react”。

### 4. 引入请求审计日志

新增 `RequestAuditLogger`，按 JSONL 形式写入请求审计日志。

当前每次请求会记录的核心字段包括：

- `route_decision`
- `plan`
- `step_results`
- `final_response`
- `latency_profile`
- `fallback_path`

审计写入位置已经接入 `StreamingService` 的统一收尾逻辑，因此 direct/planned/react 路径都可以走同一套落盘入口。

这使得线上问题具备 route/plan/step/final 级别的回放基础。

### 5. 补齐 Prompt / Schema / Planner 版本化

本次对结构化对象增加了版本字段：

- `ExecutionPlan`
  - `planner_version`
  - `schema_version`
  - `budget_policy_version`
- `FinalResponse`
  - `versions`
  - `route_decision`
  - `budget_usage`
  - `fallback_path`

当前响应中可回溯的主要版本包括：

- `router_policy_version`
- `planner_version`
- `schema_version`
- `synth_prompt_version`
- `fallback_policy_version`
- `budget_policy_version`

这样后续出现回答差异时，可以明确回答“这次是哪个 planner/schema/synth prompt 版本生成的”。

## 主要模块实现

### 1. 策略模块

新增目录：`src/agent/policies/`

核心文件：

- `model_policy.py`
  - 负责模型分层选择
- `budget_policy.py`
  - 负责请求预算定义、消耗统计、超限拦截
- `fallback_policy.py`
  - 负责 fallback 策略分类与执行结果解释

这一层的作用是把企业级策略从业务执行代码中抽离出来，避免 `TaskOrchestrator` 和 `StreamingService` 继续堆积条件分支。

### 2. 审计模块

新增目录：`src/agent/audit/`

核心文件：

- `request_audit.py`
  - 提供统一 JSONL 审计写入器

它的职责是把请求执行过程结构化落盘，而不是参与回答生成。

### 3. 执行与响应元数据增强

涉及文件：

- `src/agent/models/execution_plan.py`
- `src/agent/models/final_response.py`
- `src/agent/task_orchestrator.py`
- `src/agent/executor.py`
- `src/agent/response_synthesizer.py`

这部分主要完成：

- 给 plan / final response 补齐版本、预算、fallback、route 元数据
- 在 executor 中登记工具调用次数和并行度
- 在 synthesizer 中登记上下文长度和 LLM 调用次数
- 在 orchestrator 中为每次请求创建独立预算 tracker，并把治理信息写回最终响应

### 4. 运行时接线

涉及文件：

- `src/agent/__init__.py`
- `src/agent/streaming_service.py`
- `src/core/config.py`

这部分主要完成：

- 新增阶段 4 所需配置项
- 在运行时初始化中创建模型策略、fallback 策略、审计 logger
- 在 streaming 收尾阶段统一落审计日志
- 在预算超限时返回明确错误信息，并保留治理上下文

## 测试与验证

本次新增了阶段 4 单测，覆盖：

- 预算限制生效
- 模型策略选择
- planned 响应包含版本/预算/fallback 元数据
- 请求审计日志成功写入

建议执行：

```bash
pytest -q tests/unit/test_enterprise_governance_stage4.py tests/unit/test_planner_executor_stage3.py tests/unit/test_agent_governance_phase0.py
```

## 当前边界

虽然阶段 4 的企业级基础能力已经接入，但仍有几个边界需要说明：

- `ModelPolicy` 已完成运行时接线，但 Router / ParamParser 目前仍主要是规则逻辑，尚未真正启用小模型推理
- `FallbackPolicy` 目前已完成分类与记录，alternate tool / cached answer 仍以策略位为主，尚未形成完整缓存体系
- 审计目前先落 JSONL 文件，尚未接入数据库或检索式回放后台
- 预算控制目前覆盖主链路关键节点，后续仍可继续补齐到 memory / RAG / fallback web path 这类外围调用

## 结论

阶段 4 改造完成后，当前 AstroAgent 已具备以下企业级能力基础：

- 可按角色分层选模型
- 可限制单请求资源消耗
- 可区分不同 fallback 路径
- 可记录结构化审计日志
- 可回溯 planner/schema/prompt/policy 版本

这使项目从“能跑的 Plan-and-Solve 主链路”进一步进入“可发布、可治理、可回放”的工程化阶段。
