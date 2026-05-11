# Agent Memory Context Evaluation

## 1. 背景

AstroAgent 的短期记忆系统不是简单保存最近 N 轮对话，而是记录 `Message`、`ToolCallRecord`、`TaskState`、`SummarySnapshot` 等结构化信息，并通过 `RetrievalPlanner` 构建 prompt-facing `context_text`。

这些结构让系统可以在上下文预算内选择相关消息、工具证据、任务状态和摘要快照，避免把完整历史无差别塞进 prompt。

## 2. 评估目标

本评估不测试最终回答质量，也不调用真实 LLM 或真实 MCP。它只测试 `MemoryService.build_context` 的上下文构建质量。

核心指标包括：

- `memory_hit_rate`
- `tool_evidence_reuse_rate`
- `paraphrase_hit_rate`
- `irrelevant_memory_injection_rate`
- `wrong_tool_evidence_injection_rate`
- `harmful_wrong_injection_rate`
- `stale_tool_evidence_present_rate`
- `harmful_stale_evidence_present_rate`
- `fresh_evidence_primary_rate`
- `context_build_latency`

## 3. V1 Smoke Eval

V1 用于验证基础链路是否可用，包括消息写入、工具调用写入、`task_state` 写入、`build_context`、工具证据复用和 token saving。

最终结果：

| Metric | Value |
|---|---:|
| `memory_hit_rate` | 100.00% |
| `tool_evidence_reuse_rate` | 100.00% |
| `avg_irrelevant_memory_injection_rate` | 3.33% |
| `context_build_latency_p95_ms` | 9.38 ms |

## 4. V2 Stress Eval

V2 用于压力测试，覆盖：

- 模糊追问
- 长历史噪声
- 多工具冲突
- 新旧证据冲突
- paraphrase robustness
- token budget sweep

最终结果：

| Metric | Value |
|---|---:|
| `memory_hit_rate` | 100.00% |
| `tool_evidence_reuse_rate` | 100.00% |
| `paraphrase_hit_rate` | 100.00% |
| `wrong_tool_evidence_injection_rate` | 0.00% |
| `harmful_wrong_injection_rate` | 0.00% |
| `stale_tool_evidence_present_rate` | 0.00% |
| `stale_message_present_rate` | 0.00% |
| `harmful_stale_evidence_present_rate` | 0.00% |
| `fresh_evidence_primary_rate` | 100.00% |
| `context_build_latency_p95_ms` | 3.26 ms |

## 5. Strict vs Harmful Metrics

`wrong_tool_injection_rate` 是 legacy strict 指标：任何 wrong keyword 出现在 context 里都会算，包括 `task_state` 里的负向约束。

`harmful_wrong_injection_rate` 只统计真正可能误导模型的污染，例如错误工具证据或非约束消息噪声。

示例：

- “上海天气云量 68%”作为 `selected_tool_call` 进入北京问题上下文，是 harmful。
- “不要混入上海”作为 `task_state` constraint 出现，是 guardrail，不是 harmful。

同理，`stale_evidence_present_rate` 是 strict 指标；`harmful_stale_evidence_present_rate` 只统计 stale selected tools 或非约束消息。

## 6. 优化过程总结

初始 V2 中 `memory_hit_rate` 是 100%，但 `wrong_tool_injection_rate` 也是 100%，说明系统是高召回、低精度：相关证据能召回，但错误工具证据、旧证据和无关历史也一起进入 context。

后续只修改读取侧，不改 `MemoryService` 接口、数据库 schema 或 write side：

- 增加 tool evidence metadata 解析
- 增加 query/task focus 解析
- 增强 entity-aware tool evidence ranking
- 增加 stale/superseded evidence 抑制
- 细分 strict/harmful 评估口径

## 7. 当前结论

当前版本可以阶段性收敛。相关记忆和工具证据能稳定召回，harmful tool evidence 已经清零，剩余 strict failure 主要来自 `task_state` guardrail constraints。

这些 guardrail constraints 包括“不要混入上海”“不要混入 M31”“旧参数冲突”等，它们用于约束模型不要采用错误证据，不属于 harmful tool evidence。

## 8. 后续触发条件

只有满足以下条件时才继续修改 `RetrievalPlanner`：

- `harmful_wrong_injection_rate > 0`
- `stale_tool_evidence_present_rate > 0`
- `stale_message_present_rate > 0` 且 message 不是负向约束
- V1/V2 `memory_hit_rate` 明显下降
- `current_goal` / `next_action` 被人工确认引导错误实体

## 9. 运行命令

```bash
python scripts/evaluation/evaluate_memory_context.py \
  --dataset data/eval/memory/memory_context_eval_v1.json

python scripts/evaluation/evaluate_memory_context_v2.py \
  --dataset data/eval/memory/memory_context_eval_v2.json

pytest -q tests/evaluation/test_memory_context_eval.py tests/evaluation/test_memory_retrieval_planner_focus.py
```
