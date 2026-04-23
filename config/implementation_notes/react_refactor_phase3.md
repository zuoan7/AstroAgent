# AstroAgent ReAct Refactor 实现说明（阶段3）

本文档记录基于 `docs/AstroAgent_ReAct_Refactor_Plan.md` 已完成的阶段3实际改动。重点说明当前代码中已经落地的 `Planner + StepExecutor` 主链路、流式执行对齐方式，以及本阶段涉及的主要模块。

## 1. 阶段3目标与当前结论

阶段3的目标是把项目从“planned_task 只是模板顺序调用”推进到“真正的 Plan-and-Solve 执行内核”，解决两个核心问题：

- `planned_task` 只有路由概念，没有正式 `ExecutionPlan`
- 前端 plan 事件是展示型伪步骤，不对应真实执行步骤

当前仓库里，阶段3已经实际落地的内容主要是：

- 新增 `Planner`，把复杂任务转成正式 `ExecutionPlan`
- 新增 `StepExecutor`，负责按 `PlanStep` 执行、并行组调度、重试与超时控制
- 扩展 `ExecutionPlan / PlanStep` 数据结构，补齐 title、description、parallel_group、retry_policy、timeout_ms 等字段
- `TaskOrchestrator` 的 planned path 改为 `build_execution_plan -> execute -> synthesize`
- `StreamingService` 在 planned path 中先展示真实计划，再根据 `execution_trace` 回放步骤级事件
- `FinalResponse` 增加 `execution_plan / execution_trace`，支持步骤回放和结果审计

需要明确的是：

- 阶段3已经完成“正式 Planner + Step Executor 接入主链路”
- 当前 Planner 以模板型规划为主，还没有把 LLM Planner 作为默认执行方式
- `fallback_react` 仍然保留，开放式复杂任务仍可走 ReAct 兜底

## 2. 本阶段已实现的功能

### 2.1 `ExecutionPlan / PlanStep` 从占位模型变为正式契约

文件：

- `src/agent/models/execution_plan.py`

本轮扩展后的 `PlanStep` 字段包括：

- `id`
- `kind`
- `title`
- `description`
- `skill`
- `params`
- `required`
- `parallel_group`
- `retry_policy`
- `timeout_ms`

同时补充了：

- `PlanStep.to_dict()`
- `ExecutionPlan.to_dict()`
- `ExecutionPlan.to_frontend_steps()`

这一步的意义是：

- 后端不再只知道“要跑哪些技能”
- 而是正式知道“每一步叫什么、是否必需、是否可以并行、最多重试几次、前端应该怎么展示”

### 2.2 新增 `Planner`

文件：

- `src/agent/planner.py`

当前 `Planner` 已正式接入 planned path，负责根据：

- `query`
- `route_decision`
- `matched_skills`
- `task_type`
- `chat_history`
- `user_profile`

输出 `ExecutionPlan`。

目前已落地的是模板型 Planner，覆盖了阶段3要求的高频任务：

- `observation_recommendation`
- `celestial_event_analysis`
- `deep_sky_guidance`
- `astrophotography_advice`

同时保留了通用兜底规划：

- 若命中已识别技能但没有专用模板，则按 `matched_skills` 生成 generic plan
- LLM Planner 目前只保留扩展位，没有变成默认主路径

### 2.3 新增 `StepExecutor`

文件：

- `src/agent/executor.py`

`StepExecutor` 是阶段3真正新增的执行内核，当前职责包括：

- 接收 `ExecutionPlan`
- 按步骤执行 `PlanStep`
- 支持同一 `parallel_group` 内的步骤并发执行
- 支持 `retry_policy`
- 支持 `timeout_ms`
- 区分 `required=true/false`
- 产出正式执行结果 `ExecutionOutcome`

内部补充了两个结构：

- `StepExecutionResult`
- `ExecutionOutcome`

当前行为：

- 成功步骤会记录 `summary / sources / latency_ms / input_params`
- 失败步骤会统一映射到 `SkillResult.from_error(...)`
- 如果必需步骤失败，执行器会停止后续步骤并返回 halted 状态
- 非必需步骤失败不会阻断整个计划

这一步让 planned task 第一次具备了“步骤级执行、步骤级失败表达、并行组调度”的正式能力。

### 2.4 `TaskOrchestrator` 改为真正的 Plan-and-Solve 主链路

文件：

- `src/agent/task_orchestrator.py`

阶段2中，planned path 仍然只是：

- 根据 task_type 推出技能列表
- 逐个调用技能
- 最后统一 synthesize

阶段3改造后，planned path 变为：

1. `build_execution_plan(...)`
2. `StepExecutor.execute(...)`
3. `ResponseSynthesizer.synthesize(...)`

同时 `TaskOrchestrator` 新增正式依赖：

- `Planner`
- `StepExecutor`

并暴露：

- `build_execution_plan()`

这让 orchestrator 从“模板型多技能顺序执行器”升级为“带正式计划对象的复杂任务编排层”。

### 2.5 `FinalResponse` 增加执行计划与执行轨迹

文件：

- `src/agent/models/final_response.py`
- `src/agent/response_synthesizer.py`

本轮新增两个关键字段：

- `execution_plan`
- `execution_trace`

其中：

- `execution_plan` 保存 Planner 生成的正式计划
- `execution_trace` 保存 Executor 产出的步骤执行结果

`ResponseSynthesizer.synthesize()` 也同步支持接收这两个字段并写入 `FinalResponse`。

这一步的意义是：

- planned path 的最终响应不再只包含“答案 + 来源 + tools_used”
- 还包含“这份答案是按照什么计划执行出来的”

这为后续的步骤回放、故障排查、治理审计打下了基础。

### 2.6 `StreamingService` 改为展示真实计划步骤

文件：

- `src/agent/streaming_service.py`

阶段2里：

- direct/planned path 虽然已经使用 `FinalResponse`
- 但前端看到的 plan 仍然是 `understand / memory / tools / answer` 这种展示型伪步骤

阶段3改造后，planned path 的流式行为变为：

1. 路由完成后，先调用 orchestrator 的 `build_execution_plan()`
2. 把真实 `PlanStep` 转成前端 plan steps
3. 执行结束后，根据 `execution_trace` 回放每个 step 的：
   - `step_start`
   - `evidence_found`
   - `step_end`
4. 最后再进入 `answer` 步骤并输出最终答案

当前实现仍然保留：

- `understand`
- `memory`
- `answer`

这三个顶层步骤用于前端统一展示；

但中间的执行步骤已经不再是单个笼统的 `tools`，而是由 `ExecutionPlan` 中的真实步骤展开。

这意味着 planned path 的事件流已经从“伪计划展示”切到“真实计划回放”。

### 2.7 Runtime 正式接入 Planner 与 StepExecutor

文件：

- `src/agent/__init__.py`

`create_session_runtime()` 现在会显式初始化：

- `ResponseSynthesizer`
- `Planner`
- `StepExecutor`
- `TaskOrchestrator`
- `StreamingService`

这说明阶段3能力已经不是局部实验代码，而是正式进入了 AstroAgent 的主 runtime 组装流程。

## 3. 本阶段主要模块

阶段3实际涉及的核心模块可以分成四层。

### 3.1 计划契约层

文件：

- `src/agent/models/execution_plan.py`
- `src/agent/planner.py`

职责：

- 定义正式计划对象
- 把复杂任务转换为可执行步骤
- 为前端提供一致的计划视图

### 3.2 步骤执行层

文件：

- `src/agent/executor.py`

职责：

- 执行计划中的步骤
- 管理并行组
- 管理重试与超时
- 输出步骤级执行结果

### 3.3 编排与响应层

文件：

- `src/agent/task_orchestrator.py`
- `src/agent/response_synthesizer.py`
- `src/agent/models/final_response.py`

职责：

- 在 planned path 中串起 planner、executor、synthesizer
- 汇总步骤结果形成最终答案
- 把执行计划和执行轨迹写入最终响应

### 3.4 流式接线层

文件：

- `src/agent/streaming_service.py`
- `src/agent/__init__.py`

职责：

- 在流式主链路里先生成真实计划
- 在前端事件流中展示真实步骤
- 把阶段3执行能力接入统一 runtime

## 4. 测试与验证

本轮新增并验证了阶段3相关测试：

- `tests/unit/test_planner_executor_stage3.py`

覆盖内容包括：

- Planner 能否生成正式 `ExecutionPlan`
- StepExecutor 能否执行并行步骤并产生执行轨迹
- StreamingService 能否在 planned path 中展示真实计划步骤

本次已验证通过：

```bash
pytest -q tests/unit/test_planner_executor_stage3.py tests/unit/test_agent_governance_phase0.py
```

## 5. 当前状态评估

和阶段2相比，阶段3最关键的变化不是“多了几个类”，而是复杂任务主链路已经从：

- 路由后直接顺序调用技能

变为：

- 路由
- 规划
- 步骤执行
- 响应合成

也就是说，planned task 现在已经具备了真正的 Plan-and-Solve 骨架。

## 6. 当前边界与后续建议

当前阶段3已经落地，但仍有几个边界需要明确：

1. 当前 Planner 仍以模板规划为主

- 适合高频、边界清晰的天文业务
- 还没有把 LLM Planner 作为默认复杂任务规划器

2. 当前 `StreamingService` 对 planned path 的步骤事件采用“执行后按 execution_trace 回放”

- 计划和步骤已经真实
- 但还不是 executor 执行时的逐步实时推送

3. `fallback_react` 仍然保留

- 开放式问题或未识别场景仍可回退到 ReAct
- 系统还没有完全移除 `Thought:` / `Final Answer:` 文本解析

4. 预算治理还没有进入执行器内核

- 当前已经有步骤级 timeout、retry 和 required 标记
- 但还没有正式的 `max_tool_calls / max_total_time_ms / budget policy`

5. 步骤间数据依赖还比较轻

- 当前步骤参数主要由 query 解析得到
- 还没有把“前一步结构化输出作为后一步输入”做成通用变量绑定机制

下一步如果继续推进，建议优先进入阶段4：

- 引入预算控制
- 引入策略分层
- 把 execution trace 和治理指标打通
- 再考虑把 executor 事件变为真正的实时 step streaming
