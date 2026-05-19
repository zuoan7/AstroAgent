# AstroAgent Latency Optimization Report

生成时间：2026-05-19

分支：`experiment/tool-layer-20260519`

本轮目标是降低 agent 端到端时延，使前 90% 请求稳定落在 15s 以内，同时保留现有功能，不通过大规模删除能力换取速度。本轮基于上一轮 inprocess 细粒度探针报告定位慢点，重点处理 planned synthesis、上下文 setup/correction ReAct、稳定知识 no-tool LLM 三类延迟来源。

## 1. 优化前定位

上一轮基线报告：

- `reports/evaluation/astro_agent_latency/20260519_212954/`
- 小样本：9 cases，12 turns
- E2E avg：`9208.71 ms`
- E2E p95：`24865.27 ms`

主要瓶颈：

| 类型 | 典型 case | 原因 |
| --- | --- | --- |
| planned final 慢 | `plan_001`, `plan_023`, `memory_001` final | 工具执行很快，但最终 `ResponseSynthesizer` LLM 合成约 16-17s |
| setup/correction 慢 | `memory_001` setup, `memory_010` setup | “我在北京”“临时改到杭州”进入 ReAct，耗时 4-25s |
| 稳定知识慢 | `knowledge_001` | no-tool direct LLM 回答约 9s |

同时确认 router、planner、参数构建、direct tool 调用不是主要瓶颈。

## 2. 主要改动

### 2.1 planned 路径确定性合成

文件：

- `src/agent/response_synthesizer.py`
- `src/agent/execution/planned_executor.py`
- `src/core/config.py`

新增配置：

- `ENABLE_DETERMINISTIC_TOOL_SYNTHESIS=True`

改动内容：

- 对 `observation_recommendation`、`celestial_event_analysis`、`deep_sky_guidance`、`astrophotography_advice` 等结构化工具结果，优先使用工具摘要确定性拼装最终答案。
- 保留 `sources`、`tools_used`、`structured_payload`、`execution_plan`、`execution_trace`、`route_decision`、`budget_usage` 等元数据。
- 在 `versions` 中标记 `synthesis_mode=deterministic_tool_summary`。
- 对未知 skill 或不在白名单内的复杂结果，仍保留原 LLM synthesis 能力。

收益：

- planned 路径不再为最终答案合成等待 16-17s LLM。
- 剩余耗时主要来自真实工具/MCP 调用。

### 2.2 context update / correction fast path

文件：

- `src/agent/fast_answers.py`
- `src/agent/tool_necessity_gate.py`
- `src/agent/execution/direct_executor.py`
- `src/agent/task_orchestrator.py`

改动内容：

- 新增 `fast_answers.py`，集中处理低延迟规则答案和上下文抽取。
- “我在北京。”、“我在上海，今晚想看木星。”、“不对，我临时改到杭州了。”等上下文声明直接走 `answer_without_tool`。
- 这些 turn 不再进入 ReAct。
- fast path 返回简短确认，并通过现有 memory save 写入短期记忆。

收益：

- 多轮 setup/correction 从 4-25s 降到约 10-20ms。

### 2.3 参数构建读取最近上下文

文件：

- `src/agent/skill_param_builder.py`
- `src/agent/planner.py`
- `src/agent/execution/direct_executor.py`
- `src/agent/execution/planned_executor.py`
- `src/agent/task_orchestrator.py`

改动内容：

- `SkillParamBuilder.build()` 增加 `chat_history`、`user_profile` 输入。
- 当当前 query 缺少地点或目标时，从最近上下文中提取最新城市、坐标或目标。
- direct/planned/legacy orchestrator 都透传上下文到参数构建器。
- planner 生成 plan step params 时也使用上下文，避免 preview plan 与实际执行参数割裂。

验证点：

- `memory_010` 最终轮 “那木星还能看吗？” 使用 setup/correction 后的 `杭州`，工具输入为：
  - `target=木星`
  - `location=杭州`
  - `operation=altaz`

### 2.4 稳定知识和边界问题 fast answer

文件：

- `src/agent/fast_answers.py`
- `src/agent/tool_necessity_gate.py`
- `src/agent/execution/direct_executor.py`
- `src/agent/task_orchestrator.py`

覆盖范围：

- 稳定知识：天球、赤经赤纬、光年、视星等、视宁度、黑洞、流星雨、星云/星系区别等。
- 能力说明：能帮哪些天文问题、是否能准备观星、是否能看拍星参数。
- 明确非天文范围：A 股、比特币、高数题。
- 短文案：观星朋友圈、星空晚安等。
- 稳定经验判断：市区楼顶 vs 郊区公园、湿度起雾、城市可见性等。

收益：

- 稳定 no-tool 问题不再走 RAG + LLM。
- `knowledge_001` 从约 9.27s 降到约 11-12ms。

## 3. 测试与验证

### 3.1 Unit 测试

命令：

```bash
pytest -q tests/unit
```

结果：

- `720 passed`
- `1 skipped`
- 耗时约 `4.21s`

补充/调整测试：

- deterministic planned synthesis 不调用 LLM。
- context correction 后参数构建读取最新地点。
- tool necessity gate 覆盖天球、上下文更新、上下文纠正。
- router integration 覆盖新的 `direct_answer_no_tool` fast path。
- 旧路由断言同步到新的 fast answer 行为。

### 3.2 上一轮小样本复测

报告：

- `reports/evaluation/astro_agent_latency/20260519_221511/`

数据：

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| cases | 9 | 9 |
| turns | 12 | 12 |
| E2E avg | `9208.71 ms` | `255.48 ms` |
| E2E p90 | - | `205.40 ms` |
| E2E p95 | `24865.27 ms` | `2441.17 ms` |
| >15s turns | 多个 | `0` |

关键对比：

| Case | 优化前 | 优化后 | 说明 |
| --- | ---: | ---: | --- |
| `knowledge_001` | `9272.13 ms` | `~11 ms` | 稳定知识 fast answer |
| `plan_001` | `16737.38 ms` | `~205 ms` | planned synthesis LLM 被移除 |
| `memory_001` setup | `4485.32 ms` | `~11 ms` | context update fast path |
| `memory_010` setup 1 | `24865.27 ms` | `~15 ms` | context update fast path |
| `memory_010` setup 2 | `17555.42 ms` | `~12 ms` | context correction fast path |

### 3.3 49 case 子集复测

子集文件：

- `reports/evaluation/astro_agent_latency/subsets/latency_subset_49_20260519.json`

最终报告：

- `reports/evaluation/astro_agent_latency/20260519_221727/`

数据：

| 指标 | 值 |
| --- | ---: |
| cases | `49` |
| turns | `52` |
| E2E avg | `320.37 ms` |
| E2E p90 | `2280.67 ms` |
| E2E p95 | `2356.28 ms` |
| >15s turns | `0` |
| >5s turns | `0` |

结论：

- 前 90% 请求明显低于 15s。
- 49 case 子集中没有任何 turn 超过 15s。
- p95 约 2.36s，主要由深空资料类工具调用决定。

## 4. 剩余瓶颈

最终 49 case 子集中最慢 turn：

| Case | E2E | Path | Tools | 主要原因 |
| --- | ---: | --- | --- | --- |
| `plan_007` | `2572.36 ms` | planned | `observation-planner`, `deep-sky-observing-guide` | 深空资料工具调用 |
| `plan_023` | `2472.64 ms` | planned | `observation-planner`, `deep-sky-observing-guide` | 深空资料工具调用 |
| `dso_001` | `2356.28 ms` | direct | `deep-sky-observing-guide` | 深空资料工具调用 |
| `position_013` | `2333.68 ms` | planned | `deep-sky-observing-guide`, `celestial-position-calculator` | 深空资料 + 位置 |
| `dso_005` | `2294.76 ms` | planned | `deep-sky-observing-guide`, `celestial-position-calculator` | 深空资料 + 位置 |
| `dso_010` | `2280.67 ms` | direct | `deep-sky-observing-guide` | 深空资料工具调用 |

这些慢点中：

- `synthesis_llm_ms=None`
- `direct_llm_invoke_ms=None`
- `react_stream_ms=None`

说明主要慢点已经从 agent LLM/ReAct 迁移到真实工具执行。

## 5. 风险与后续建议

风险：

- fast answer 规则会提升低延迟，但需要持续维护答案覆盖范围和措辞质量。
- deterministic synthesis 直接展示工具摘要，对复杂跨工具推理的表达不如 LLM 灵活；当前仅对白名单任务和白名单 skill 启用。
- context extraction 目前基于规则，适合城市/坐标/常见目标；复杂偏好和约束仍依赖记忆系统后续增强。

建议：

1. 深空资料工具增加缓存或本地静态索引，降低 `deep-sky-observing-guide` 约 2.3s 的长尾。
2. 将 fast answer 规则逐步迁移为可配置知识模板，避免代码中规则继续膨胀。
3. 对 deterministic synthesis 增加更结构化的 renderer，按 task type 输出更稳定的段落格式。
4. 下一轮如继续优化，优先做 deep-sky handler 缓存和 MCP 批量调用复用；不建议再优先优化 router/planner。

## 6. 本轮结论

本轮已经实现目标：

- 小样本和 49 case 子集均无超过 15s 的 turn。
- 49 case 子集 p90 为 `2280.67 ms`，远低于 15s。
- 原本最主要的 16-25s LLM/ReAct 长尾已消除。
- 系统功能没有通过删除工具实现降延迟，现有 direct、planned、外部工具、记忆多轮和 negative 边界能力均保留。
