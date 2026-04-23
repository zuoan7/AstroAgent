# AstroAgent ReAct Refactor 实现说明（阶段2）

本文档记录基于 `docs/AstroAgent_ReAct_Refactor_Plan.md` 已完成的阶段2实际改动。重点说明当前代码中已经落地的结构化执行契约、主链路接线方式，以及阶段2涉及的主要模块。

## 1. 阶段2目标与当前结论

阶段2的目标是把系统从“字符串拼接驱动”推进到“结构化协议驱动”，解决两个核心问题：

- 技能层返回值以字符串为主，上层只能猜结果结构
- 最终响应缺少稳定契约，流式链路和前端消费不够统一

当前仓库里，阶段2已经实际落地的内容主要是：

- 引入 `SkillResult` 作为统一技能返回结构
- 引入 `FinalResponse` 作为统一最终响应结构
- 新增 `ResponseSynthesizer` 负责汇总技能结果并生成最终回答
- `SkillManager / AstronomySkillRouter / handler` 全链路切换到结构化返回
- `TaskOrchestrator` 改为返回 `FinalResponse`
- `StreamingService` 在 direct/planned 路径直接消费 `FinalResponse`
- 保留 LangChain/ReAct 兼容层，避免一次性推翻旧调用方式

需要明确的是：

- 阶段2已经完成“结构化结果与结构化响应”改造
- 阶段3规划中的正式 `Planner + StepExecutor` 还没有落地
- `ExecutionPlan` 数据模型已新增，但当前还未成为主执行内核

## 2. 本阶段已实现的功能

### 2.1 技能结果统一为 `SkillResult`

文件：`src/agent/models/skill_result.py`

新增统一技能结果结构：

- `skill_name`
- `success`
- `data`
- `summary`
- `sources`
- `error_code`
- `error_message`
- `latency_ms`

同时补充了几个关键辅助方法：

- `to_legacy_str()`：供旧 LangChain Tool 接口继续返回字符串
- `to_dict()`：供结构化消费和调试
- `to_tool_timeline_entry()`：统一转成工具时间线条目
- `to_evidence_entry()`：统一转成证据条目
- `from_error()`：统一错误返回格式

这一步意味着：

- 上层不再只拿到一段自然语言
- 技能输出的“原始结构化数据、用户可读摘要、来源、错误码、耗时”被拆开建模

### 2.2 最终响应统一为 `FinalResponse`

文件：`src/agent/models/final_response.py`

新增统一最终响应结构：

- `answer`
- `summary`
- `sources`
- `tools_used`
- `confidence`
- `structured_payload`
- `route`
- `task_type`
- `memory_hits`

并提供：

- `to_dict()`：面向结构化输出
- `to_legacy_dict()`：兼容旧响应消费方式

这让 direct/planned 路径不再依赖临时 dict 拼装，而是统一回到同一个响应契约。

### 2.3 新增 `ResponseSynthesizer`

文件：`src/agent/response_synthesizer.py`

新增统一答案合成器，负责把技能结果转换为最终回答与前端可消费结构。

当前已实现四类合成入口：

- `synthesize()`：多技能 planned task，调用 LLM 做最终整合
- `synthesize_direct()`：单技能 direct task，直接封装结构化响应
- `synthesize_qa()`：简单问答场景，携带 RAG 证据
- `synthesize_smalltalk()`：闲聊快速返回

合成器统一负责：

- 汇总 `sources`
- 汇总 `tools_used`
- 构建 `structured_payload`
- 计算 `confidence`
- 输出 `FinalResponse`

这一步把“最终答案生成”和“工具结果结构化整理”集中到了一个独立模块，而不是分散在 orchestrator 或 streaming 层里。

### 2.4 技能路由层改为结构化返回

涉及文件：

- `src/agent/skill_manager.py`
- `src/skills/router.py`

本轮关键变化：

- `SkillManager.call_skill()` 现在返回 `SkillResult`
- `AstronomySkillRouter.call()` 对所有技能统一返回 `SkillResult`
- 简单技能调用 MCP 后，会解析结果并写入 `data / summary / sources / latency_ms`
- MCP 错误会统一映射到 `SkillResult.from_error(...)`

同时保留兼容层：

- LangChain Tool 注册仍可继续使用
- Tool 函数通过 `result.to_legacy_str()` 返回旧式字符串

这意味着：

- 新主链路可以使用结构化结果
- ReAct fallback 和旧工具接口不需要同时重写

### 2.5 复杂 handler 从“拼大段文本”改为“结构化数据 + 摘要”

文件：`src/skills/skill_handlers.py`

阶段2已完成的 handler 统一改造包括：

- `ObservationPlannerHandler`
- `CelestialEventsForecastHandler`
- `DeepSkyObservingGuideHandler`
- `NeoTrackerHandler`
- `AstrophotographyCalculatorHandler`
- `CelestialPositionCalculatorHandler`

这些 handler 当前的共同特征是：

- 返回 `SkillResult`
- 在 `data` 中保留结构化字段
- 在 `summary` 中保留面向用户的可读摘要
- 在 `sources` 中记录工具来源
- 在 `latency_ms` 中记录调用耗时

其中几个典型例子：

- `observation-planner` 会返回观测日期、地点、天气数据、本周天象、今晚最佳目标
- `celestial-events-forecast` 会返回时间范围、事件类型、原始天象文本
- `deep-sky-observing-guide` 会返回目标、设备、目标信息、星系补充信息
- `neo-tracker` 会返回过滤后的近地天体列表与总量
- `celestial-position-calculator` 会返回目标、时间、经纬度、位置结果

这一步是阶段2最关键的实际改动之一，因为 handler 不再只承担“生成一段最终文案”的职责，而是开始产出可复用的数据结构。

### 2.6 `TaskOrchestrator` 改为返回 `FinalResponse`

文件：`src/agent/task_orchestrator.py`

本轮改造后：

- direct task 返回 `FinalResponse`
- planned task 返回 `FinalResponse`
- 单技能执行收集 `SkillResult`
- 多技能执行收集 `List[SkillResult]`
- 最终统一交给 `ResponseSynthesizer`

当前执行逻辑：

- `smalltalk` -> `synthesize_smalltalk()`
- `simple_qa` -> RAG 检索 + LLM 回答 + `synthesize_qa()`
- `single_tool_lookup` -> 调用单个技能 + `synthesize_direct()`
- `planned_task` -> 顺序调用匹配技能 + `synthesize()`

和阶段1相比，主要变化不是“多了 Planner”，而是 orchestrator 已经不再直接拼响应 dict，而是消费结构化 `SkillResult` 并产出结构化 `FinalResponse`。

### 2.7 流式链路在主路径上直接消费 `FinalResponse`

文件：`src/agent/streaming_service.py`

当前行为已经分成两类：

1. `direct` / `planned` 路径

- 调用 orchestrator
- 直接拿到 `FinalResponse`
- 从 `final_resp.answer / sources / tools_used` 生成流式事件

2. `react` fallback 路径

- 仍然走 ReAct 流
- 仍然需要解析 `Thought:` / `Final Answer:`

这意味着阶段2已经实现了：

- 主链路不再依赖 ReAct 文本格式来拿最终答案

但也保留了一个清晰边界：

- fallback_react 仍然沿用旧式文本解析逻辑

因此更准确的说法是：

- direct/planned 主链路已切到 `FinalResponse` 契约
- 整个系统尚未完全移除 ReAct 文本解析

### 2.8 Runtime 初始化接入 `ResponseSynthesizer`

文件：`src/agent/__init__.py`

`create_session_runtime()` 现在会显式初始化：

- `llm`
- `ResponseSynthesizer`
- `TaskOrchestrator`
- `StreamingService`

这说明结构化响应能力已经进入正式 runtime 组装流程，而不是停留在独立实验代码中。

## 3. 本阶段主要模块

阶段2实际涉及的核心模块可以分成四层。

### 3.1 数据契约层

文件：

- `src/agent/models/skill_result.py`
- `src/agent/models/final_response.py`
- `src/agent/models/execution_plan.py`
- `src/agent/models/__init__.py`

职责：

- 定义技能结果结构
- 定义最终响应结构
- 预留后续 `ExecutionPlan / PlanStep` 数据模型

说明：

- `ExecutionPlan` 和 `PlanStep` 已经存在
- 但当前还没有正式的 planner/executor 去驱动它们

### 3.2 技能适配层

文件：

- `src/agent/skill_manager.py`
- `src/skills/router.py`
- `src/skills/skill_handlers.py`

职责：

- 把 MCP 工具与高层技能统一包装
- 输出 `SkillResult`
- 保留旧 LangChain Tool 兼容能力

### 3.3 响应合成层

文件：

- `src/agent/response_synthesizer.py`

职责：

- 根据技能执行结果生成最终回答
- 汇总来源、工具、结构化 payload、置信度
- 输出 `FinalResponse`

### 3.4 执行接线层

文件：

- `src/agent/task_orchestrator.py`
- `src/agent/streaming_service.py`
- `src/agent/__init__.py`

职责：

- 执行 direct/planned 主路径
- 在流式链路中消费 `FinalResponse`
- 组装运行时依赖并接入主服务

## 4. 测试与验证

已能在测试中看到阶段2结构化契约的直接覆盖。

相关文件：

- `tests/unit/test_skill_registry_refactor.py`
- `tests/unit/test_agent_governance_phase0.py`
- `tests/unit/test_latency_optimization.py`

覆盖点包括：

- 简单技能调用返回 `SkillResult`
- `SkillManager` 与 `AstronomySkillRouter` 的结构化兼容行为
- `StreamingService` 在 orchestrated path 中消费 `FinalResponse`
- planned task 仍可在不初始化 ReAct 的情况下完成主流程

## 5. 当前状态评估

到阶段2为止，系统已经从：

- Router 决定 direct/planned/react
- orchestrator 直接拼答案
- 技能层主要返回字符串

推进到：

- 技能层统一返回 `SkillResult`
- 主路径统一返回 `FinalResponse`
- 最终回答由 `ResponseSynthesizer` 负责生成
- 流式主路径直接消费结构化响应

这说明阶段2的核心目标已经基本完成：主链路开始从“字符串拼接”转向“结构化协议”。

## 6. 当前边界与后续建议

当前还保留以下边界：

1. 还没有正式 `Planner + StepExecutor`

- `ExecutionPlan` 只是数据模型预留
- planned task 仍然是轻量技能顺序执行

2. fallback ReAct 仍保留旧式文本解析

- 只有 orchestrated path 已切换到 `FinalResponse`
- react path 仍依赖 `Final Answer:` 提取

3. `structured_payload` 已存在，但 schema 版本治理还未建立

- 目前更偏“按技能名聚合 data”
- 还没有完整的业务级 output schema 版本化

建议下一步进入阶段3：

- 引入正式 `Planner`
- 引入 `StepExecutor`
- 让 planned task 从“轻量模板执行”升级为“可回放、可重试、可并行”的正式执行内核
