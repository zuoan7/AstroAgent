# AstroAgent ReAct Refactor 实现说明（阶段0 + 阶段1）

本文档记录基于 `docs/AstroAgent_ReAct_Refactor_Plan.md` 已完成的两轮改造实现，重点说明本次代码里实际落下的结构、行为变化、文件位置和当前边界。

## 1. 改造范围概览

### 阶段0

目标是先补齐治理基线，不大改主链路。

本轮已经落地：

- 基准数据集
- Feature Flag 框架
- 延迟/回退/解析成功率/路由偏差的观测能力
- Router 基准评估能力

### 阶段1

目标是让 ReAct 退出默认主链路，建立清晰的执行模式分层。

本轮已经落地：

- `RouteDecision` 升级
- `direct_task / planned_task / fallback_react` 三种路由
- 默认 direct/planned 主链路
- ReAct 改为惰性初始化，仅在 fallback 路径触发
- `TaskOrchestrator` 增加 planned task 执行

## 2. 阶段0实现细节

### 2.1 Feature Flag

文件：`src/core/config.py`

新增 `AgentGovernanceConfig`：

- `AGENT_MODE`
- `ENABLE_STRUCTURED_SKILL_RESULT`
- `ENABLE_PLANNER`
- `ENABLE_REACT_FALLBACK`
- `PHASE0_BENCHMARK_PATH`

当前默认值仍偏保守：

- `AGENT_MODE=hybrid`
- `ENABLE_PLANNER=false`
- `ENABLE_REACT_FALLBACK=true`

这意味着系统已经支持策略切换，但不会在阶段0直接强行改变全部行为。

### 2.2 治理模块

文件：`src/agent/governance.py`

新增了三类能力：

1. `AgentExecutionPolicy`

- 负责把 feature flag 转成运行期策略
- 提供 `choose_path(route)`，把路由结果映射到 `direct / planned / react`

2. Benchmark 能力

- `BenchmarkCase`
- `load_phase0_benchmark_cases()`
- `evaluate_router_benchmark()`

用于离线对照高频请求，统计当前路由偏差。

3. 运行观测

- `RequestObservation`
- `GovernanceMetricsRegistry`

当前支持聚合：

- `P50/P90/max latency`
- `fallback_rate`
- `output_schema_parse_rate`
- `route_mismatch_rate`
- `by_mode`
- `by_route`

### 2.3 基准数据集

文件：`config/benchmarks/agent_phase0_benchmark.json`

当前内置 40+ 条样本，覆盖：

- smalltalk
- simple_qa
- single_tool_lookup
- observation_recommendation
- celestial_event_analysis
- deep_sky_guidance
- astrophotography_advice

说明：

- 最初尝试放在 `docs/`，但仓库 `.gitignore` 忽略了 `docs/`
- 最终移动到了 `config/benchmarks/`，便于后续提交和版本管理

### 2.4 Streaming 观测接入

文件：`src/agent/streaming_service.py`

在流式链路里补充了观测点，记录：

- `agent_mode`
- `execution_path`
- `fallback_used`
- `output_schema_parse_success`
- `request_total_ms`

这些数据会汇总进 `GovernanceMetricsRegistry`。

### 2.5 AstroAgent 对外能力

文件：`src/agent/__init__.py`

新增：

- `get_governance_metrics_snapshot()`
- `evaluate_phase0_router_benchmark()`

便于在 API、调试脚本或后续运维面板里直接消费。

## 3. 阶段1实现细节

### 3.1 RouteDecision 升级

文件：`src/agent/request_router.py`

旧结构：

- `route`
- `confidence`
- `reason`
- `matched_skills`

新结构：

- `route`
- `task_type`
- `confidence`
- `reason`
- `matched_skills`
- `expected_output_schema`

同时补充了：

- `is_direct_task`
- `is_planned_task`
- `is_fallback_react`

### 3.2 路由结果改为三类执行模式

文件：`src/agent/request_router.py`

当前路由语义：

1. `direct_task`

用于：

- `smalltalk`
- `simple_qa`
- `single_tool_lookup`

2. `planned_task`

用于：

- 多技能任务
- 复杂但可模板化的观测/天象/深空/摄影请求

3. `fallback_react`

用于：

- 明显开放式、非模板化问题
- 当前规则无法稳定归类的问题

### 3.3 task_type 与 output schema

文件：`src/agent/request_router.py`

当前已落地的 `task_type`：

- `smalltalk`
- `simple_qa`
- `single_tool_lookup`
- `observation_recommendation`
- `celestial_event_analysis`
- `deep_sky_guidance`
- `astrophotography_advice`
- `open_domain_reasoning`

对应的 `expected_output_schema` 也已经由 Router 直接给出。

这一步的意义是：

- Router 不再只回答“复杂不复杂”
- 上游开始向“任务识别器”演进
- 为阶段2/3 的 `SkillResult`、`FinalResponse`、`Planner` 预留稳定接口

### 3.4 Execution Policy 改造

文件：`src/agent/governance.py`

`AgentExecutionPolicy.choose_path()` 现在按新路由工作：

- `direct_task -> direct`
- `planned_task -> planned`
- `fallback_react -> react`（受 `ENABLE_REACT_FALLBACK` 控制）

也就是说：

- 默认 ReAct 不再拦截全部复杂请求
- 只有 Router 明确打成 `fallback_react` 时，才进入 react

### 3.5 TaskOrchestrator 扩展为 direct + planned 双执行器

文件：`src/agent/task_orchestrator.py`

原先只支持：

- smalltalk
- tool_task
- simple_qa

现在支持：

- `direct_task`
- `planned_task`

其中：

1. `direct_task`

内部按 `task_type` 分发：

- `smalltalk`
- `simple_qa`
- `single_tool_lookup`

2. `planned_task`

当前是阶段1的轻量实现，还不是阶段3里的正式 Planner/Executor。

已实现行为：

- 根据 `matched_skills` 或 `task_type` 解析出一组技能
- 顺序调用技能
- 收集 `tools_used` / `sources`
- 使用 LLM 做一次最终整合回答

这让复杂任务在不开 ReAct 的情况下也能完成主流程。

### 3.6 复杂任务的最小编排策略

文件：`src/agent/task_orchestrator.py`

当前 planned task 还没有独立 `Planner` 对象，因此采用阶段1允许的最小模板化实现：

- `observation_recommendation -> weather + observation`
- `celestial_event_analysis -> events`
- `deep_sky_guidance -> deep sky`
- `astrophotography_advice -> photography + weather`

同时补了轻量参数提取：

- 城市
- 目标
- 日期/时间
- 器材
- 事件类型

这部分是阶段1过渡层，不是最终执行内核。

### 3.7 StreamingService 改造

文件：`src/agent/streaming_service.py`

关键变化有三点：

1. 支持 `agent_executor_factory`

- 不再要求启动时必须拿到 ReAct executor
- 可以在真正需要 fallback_react 时再创建

2. 新增 `_ensure_agent_executor()`

- 惰性初始化 ReAct executor
- 避免默认主链路依赖 ReAct

3. 主流程分流

当前行为：

- `direct_task` 和 `planned_task` 都先走 orchestrator
- 只有 `fallback_react` 才进入 ReAct 流式循环

同时，`generate_response()` 同步路径也做了同样的执行模式切换，不再默认先调 ReAct。

### 3.8 AstroAgent 改造为惰性 ReAct 初始化

文件：`src/agent/__init__.py`

原行为：

- `create_session_runtime()` 总是构建 `AgentExecutor`

现在的行为：

- 启动时只初始化 `llm + task_orchestrator + streaming_service`
- 通过 `_get_or_create_agent_executor()` 在 fallback 发生时再构建 ReAct executor
- `StreamingService` 通过 `agent_executor_factory` 回调拿到 executor

这一步是阶段1最关键的落地点之一，因为它真正把 ReAct 从默认主链路移出了。

## 4. 基准集在阶段1中的同步升级

文件：`config/benchmarks/agent_phase0_benchmark.json`

为了让 benchmark 和当前系统语义一致，已把 `expected_route` 从旧值升级为新值：

- `smalltalk/simple_qa/tool_task -> direct_task`
- `complex_agent -> planned_task`

这样 `evaluate_router_benchmark()` 统计出来的偏差，才对应当前系统的真实目标架构。

## 5. 测试与验证

新增/更新测试文件：

- `tests/unit/test_agent_governance_phase0.py`
- `tests/unit/test_latency_optimization.py`

覆盖点包括：

- 新路由值和 `task_type`
- governance 指标聚合
- execution policy 行为
- direct_task 仍走低延迟路径
- `planned_task` 不触发 ReAct 初始化
- `fallback_react` 才会惰性构建 ReAct executor

本地验证命令：

```bash
python -m pytest tests/unit/test_agent_governance_phase0.py tests/unit/test_latency_optimization.py -k 'not test_mcp_parallel_calls_are_truly_concurrent'
python -m py_compile src/agent/request_router.py src/agent/task_orchestrator.py src/agent/streaming_service.py src/agent/governance.py src/agent/__init__.py
```

## 6. 当前状态评估

到这里，系统已经从：

- ReAct 作为复杂任务默认主链路

推进到：

- Router 输出明确执行模式
- 默认 direct/planned 主链路
- ReAct 只做 fallback
- 复杂任务在不开 ReAct 时也能走完主流程

这符合文档中阶段1的主目标。

## 7. 仍然保留的边界与后续建议

本次是阶段1，不是阶段2/3，所以还有这些刻意保留的边界：

1. `planned_task` 仍是轻量模板执行，不是正式 `Planner + StepExecutor`

- 目前还没有 `ExecutionPlan`
- 也没有 step-level retry / timeout / parallel_group

2. 技能层仍返回字符串

- 还没有 `SkillResult`
- `sources` 还是上层根据字符串摘要拼出来的

3. 最终输出还不是正式 `FinalResponse` 契约

- 前端结构比之前稳定了
- 但还没进入阶段2的 schema 驱动模式

4. `fallback_react` 的判定仍是规则启发式

- 还不是训练/评估驱动的业务任务识别

建议下一步直接进入阶段2：

- `SkillResult`
- `FinalResponse`
- `ResponseSynthesizer`

然后再进入阶段3做真正的 `Planner + StepExecutor`。
