# AstroAgent Full E2E Evaluation Comparison

生成时间：2026-05-19

分支：`experiment/tool-layer-20260519`

本报告基于 `config/benchmarks/astro_agent_eval_dataset.json` 全量 200 条 ability case，按 `docs/agent_evaluation_dataset_design.md` 中定义的五个主要指标进行 live `/query` 端到端评测，并与第一轮 200 case baseline 对比。

## 1. 本轮测试配置

数据集：

- `config/benchmarks/astro_agent_eval_dataset.json`
- 静态校验：`200 cases, 0 errors, 0 warnings`

有效测试报告：

- `reports/evaluation/astro_agent/20260519_224725/`
- 主要文件：`summary.json`、`cases.jsonl`、`failures.md`、`events/*.json`

运行配置：

- API：`http://localhost:8002/query`
- 并发：`1`
- `suite=ability`
- `mcp_scoring=observed`
- 长期记忆：关闭
- SSE events：保存
- `ENABLE_LLM_INTENT_FALLBACK=true`
- `ENABLE_LLM_PLANNER_FALLBACK=true`
- `ENABLE_DETERMINISTIC_TOOL_SYNTHESIS=true`
- `RATE_LIMIT_PER_MINUTE=10000`

运行命令：

```bash
python scripts/evaluation/evaluate_astro_agent_dataset.py \
  --dataset config/benchmarks/astro_agent_eval_dataset.json \
  --base-url http://localhost:8002 \
  --save-events \
  --concurrency 1 \
  --request-timeout-sec 120
```

说明：

- 第一轮最佳 baseline 使用 `intent + planner fallback`，因此本轮也启用相同 fallback 配置，便于质量指标公平对比。
- 本轮第一次全量尝试 `reports/evaluation/astro_agent/20260519_224614/` 被 API 默认限流污染：30 条后触发 170 个 HTTP 429。该结果无效，不参与对比。
- 本地 benchmark 放宽 API 限流只影响评测环境，不改变 agent 行为。

## 2. 五个主要指标

### 2.1 与第一轮三组 baseline 对比

| Run | E2E 通过 | E2E 成功率 | 工具选择准确率 | Skill 准确率 | MCP observed 准确率 | 工具调用成功率 | 首事件 P95 | E2E avg | E2E P90 | E2E P95 | >15s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 第一轮：规则 only | 108/200 | 54.0% | 54.0% | 57.5% | 72.8% | 100.0% | 10.45ms | 9.09s | 22.55s | 25.28s | 44 |
| 第一轮：intent fallback | 108/200 | 54.0% | 54.0% | 60.5% | 69.4% | 100.0% | 11.05ms | 10.49s | 26.91s | 32.19s | 50 |
| 第一轮：intent + planner fallback | 112/200 | 56.0% | 56.0% | 62.0% | 71.2% | 100.0% | 11.39ms | 9.78s | 24.72s | 29.80s | 51 |
| 本轮：优化后 full E2E | 197/200 | 98.5% | 98.5% | 98.5% | 100.0% | 100.0% | 11.81ms | 3.70s | 12.39s | 17.48s | 17 |

相对第一轮最佳组 `intent + planner fallback`：

- E2E 成功率：`56.0% -> 98.5%`，提升 `+42.5pp`，通过 case 增加 `+85`。
- 工具选择准确率：`56.0% -> 98.5%`，提升 `+42.5pp`。
- Skill 准确率：`62.0% -> 98.5%`，提升 `+36.5pp`。
- MCP observed 准确率：`71.2% -> 100.0%`，提升 `+28.8pp`。
- 工具调用成功率：保持 `100.0%`。
- 首事件 P95：`11.39ms -> 11.81ms`，基本持平。
- E2E avg：`9.78s -> 3.70s`，下降约 `62.2%`。
- E2E P90：`24.72s -> 12.39s`，下降约 `49.9%`。
- E2E P95：`29.80s -> 17.48s`，下降约 `41.3%`。
- 超过 15s 的 case：`51 -> 17`，减少 `34` 个。

### 2.2 15s 目标判定

原始时延目标是“前 90% 的结果在 15s 以内，允许少数样例超出”。

本轮 full set 结果：

- E2E P90：`12.39s`
- `<=15s`：`183/200 = 91.5%`
- `>15s`：`17/200 = 8.5%`
- E2E P95：`17.48s`

结论：

- `前 90% <= 15s` 的目标已经满足。
- 但五项指标中的 `E2E P95` 仍高于 15s，说明仍有少量长尾需要继续处理。

## 3. 分类通过率对比

以下对比使用第一轮最佳组 `intent + planner fallback` 作为基线。

| Category | 第一轮通过 | 本轮通过 | 变化 | 本轮 E2E P95 |
| --- | ---: | ---: | ---: | ---: |
| `astronomy_knowledge_qa` | 15/20 | 20/20 | +5 | 0.02s |
| `astrophotography` | 11/20 | 20/20 | +9 | 14.23s |
| `celestial_events` | 6/20 | 20/20 | +14 | 8.79s |
| `celestial_object_info` | 14/15 | 15/15 | +1 | 5.66s |
| `control_no_tool` | 15/15 | 15/15 | +0 | 2.08s |
| `deep_sky_guidance` | 9/15 | 14/15 | +5 | 19.24s |
| `nasa_neo_external_data` | 5/10 | 10/10 | +5 | 18.41s |
| `negative_ambiguous_safety` | 7/15 | 15/15 | +8 | 16.96s |
| `observation_planning` | 10/20 | 19/20 | +9 | 22.51s |
| `observing_conditions` | 5/15 | 15/15 | +10 | 15.91s |
| `position_coordinates_visibility` | 8/20 | 19/20 | +11 | 17.92s |
| `stateful_memory` | 7/15 | 15/15 | +8 | 32.07s |

主要变化：

- 第一轮最大失败源 `no_tool_case_used_tool` 已从 `46` 降为 `0`。
- `requires_tool_but_no_tool_observed` 从 `16` 降为 `2`。
- celestial events、observing conditions、position、memory、negative 边界类都有明显提升。
- 剩余失败集中在 3 条 case，不再是大面积路由边界失控。

## 4. 失败 Case

本轮失败 `3/200`：

| Case | Category | 主要原因 | 实际行为 |
| --- | --- | --- | --- |
| `position_007` | `position_coordinates_visibility` | 缺少 `celestial-position-calculator` | 被稳定知识 fast answer 吃掉，返回赤经/赤纬解释 |
| `plan_012` | `observation_planning` | 缺少 `celestial-events-forecast` | 只走了 `observation-planner`，未单独暴露事件 forecast skill |
| `dso_008` | `deep_sky_guidance` | 缺少 `deep-sky-observing-guide` | 被跳星法知识 fast answer 吃掉 |

失败原因分桶：

| Failure reason | Count |
| --- | ---: |
| `requires_tool_but_no_tool_observed` | 2 |
| `missing_expected_skills:celestial-position-calculator` | 1 |
| `missing_expected_skills:celestial-events-forecast` | 1 |
| `missing_expected_skills:deep-sky-observing-guide` | 1 |

## 5. 延迟长尾

本轮最慢 case：

| Case | Category | E2E | Pass | Tools | 主要观察 |
| --- | --- | ---: | --- | --- | --- |
| `memory_011` | `stateful_memory` | 32.07s | pass | `deep-sky-observing-guide` | route decision 约 15.55s，工具约 1.14s |
| `plan_016` | `observation_planning` | 25.40s | pass | `observation-planner` | route decision 约 12.43s，工具约 0.36s |
| `plan_015` | `observation_planning` | 22.51s | pass | `observation-planner` | route decision 约 10.51s，工具约 0.36s |
| `plan_020` | `observation_planning` | 22.11s | pass | `observation-planner` | route decision 约 11.47s，工具约 0.36s |
| `position_012` | `position_coordinates_visibility` | 20.33s | pass | `celestial-position-calculator` | route decision 约 10.39s，工具约 0.01s |
| `plan_007` | `observation_planning` | 19.85s | pass | `observation-planner`, `deep-sky-observing-guide` | planned + deep sky |
| `memory_005` | `stateful_memory` | 19.78s | pass | `observation-planner`, `celestial-position-calculator` | final route decision 约 8.64s；setup turn 另有约 13.88s |
| `dso_012` | `deep_sky_guidance` | 19.24s | pass | `deep-sky-observing-guide` | route decision 约 9.66s，工具约 1.10s |
| `external_004` | `nasa_neo_external_data` | 18.41s | pass | `neo-tracker` | route decision 长尾 |
| `position_011` | `position_coordinates_visibility` | 17.92s | pass | `celestial-position-calculator` | route decision 长尾 |

长尾来源已经和第一轮不同：

- 第一轮主要慢在 planned final synthesis、ReAct、多轮 setup 误入 ReAct。
- 本轮 planned deterministic synthesis 已显著降低多数 plan case，例如 `plan_001/003/005/010/014/018/019/021` 均为 200ms 左右。
- 本轮剩余长尾主要来自启用 `ENABLE_LLM_INTENT_FALLBACK=true` 后的 `route_decision_ms`。为了和第一轮最佳 baseline 公平对比，本轮保留了该配置；它提升了质量，但仍会在低置信或复杂边界请求上引入 8-15s 级路由判断。
- 另有少数稳定经验/摄影建议仍走 direct LLM，产生 6-16s 延迟。
- 少数 device/preference setup turn 还没有完全进入 context fast path，例如“我有一台 80mm 小折射镜。”这类设备上下文声明仍可能进 ReAct。

## 6. 主要结论

本轮 full E2E 结果显示：

- 质量侧已经大幅改善：E2E 成功率从第一轮最佳 `56.0%` 提升到 `98.5%`。
- 路由和工具选择明显稳定：工具选择准确率、skill 准确率均为 `98.5%`，MCP observed 准确率为 `100.0%`。
- 工具执行稳定性没有退化：工具调用成功率保持 `100.0%`。
- 首事件延迟仍健康：P95 为 `11.81ms`，与第一轮同量级。
- E2E 时延显著改善：avg 从 `9.78s` 降到 `3.70s`，P90 从 `24.72s` 降到 `12.39s`。
- 原始目标“前 90% 在 15s 内”已满足，但 E2E P95 仍为 `17.48s`，还有 17 个 case 超过 15s。

## 7. 后续建议

下一轮如果继续压低 full set P95，优先级建议如下：

1. 限制或缓存 LLM intent fallback。
   - 当前多个长尾 case 的 `route_decision_ms` 在 8-15s。
   - 可以对高置信规则路由、明确工具关键词和已覆盖 benchmark pattern 跳过 fallback。

2. 扩展 context fast path。
   - 覆盖设备、摄影器材、偏好、观测时段等上下文声明。
   - 目标是把“我有 80mm 折射镜”“我用入门单反和三脚架”“我不想熬夜”这类 setup 从 ReAct 拉回毫秒级。

3. 收窄 fast answer 的误吃边界。
   - `position_007` 和 `dso_008` 都是 fast answer 覆盖过宽导致 requires-tool case 漏工具。
   - 对包含“坐标大概在哪里”“找目标/怎么找这个目标”等任务语义的 query，应优先进入工具或澄清。

4. 对 planned route 做 route-decision 复用。
   - `plan_015/016/020/022` 工具执行本身约 0.36s，但 route decision 是主要耗时。
   - 可考虑在 router profile 已明确 planned task 时跳过 fallback 或复用模板判断。

5. 保留 deterministic synthesis。
   - 当前多数 planned 工具结果已在 200ms 级完成，说明最终合成 LLM 不再是主瓶颈。
   - 下一步重点不应回到 synthesis，而应处理前置 fallback 和残余 ReAct。
