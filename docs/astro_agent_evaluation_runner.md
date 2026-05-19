# AstroAgent Evaluation Runner

本文档说明如何测试 `config/benchmarks/astro_agent_eval_dataset.json`。
当前数据集目标是 200 条纯文本可运行 case，覆盖工具选择准确率、工具调用成功率、端到端任务成功率、首事件延迟和端到端 P95 响应时延。

## 1. 脚本

本评测包含两个脚本：

| 脚本 | 作用 | 是否调用 Agent |
| --- | --- | --- |
| `scripts/evaluation/validate_astro_agent_dataset.py` | 校验数据集结构、字段、枚举、分类计数和工具期望冲突 | 否 |
| `scripts/evaluation/evaluate_astro_agent_dataset.py` | 调用 live `/query` SSE API，采集 trace、工具调用和延迟并打分 | 是 |

## 2. 静态校验

静态校验只依赖标准库，适合放进 CI 或 PR 检查。

```bash
python scripts/evaluation/validate_astro_agent_dataset.py \
  --dataset config/benchmarks/astro_agent_eval_dataset.json \
  --expected-total 200
```

如果希望 warning 也导致非零退出：

```bash
python scripts/evaluation/validate_astro_agent_dataset.py --strict --expected-total 200
```

静态校验会检查：

- `case_id` 唯一且非空。
- `cases` 数量等于 `target_distribution` 目标数量。
- 每个 case 的 `category`、`subcategory`、`suite`、`difficulty` 等枚举合法。
- `expected_skills` 和 `forbidden_skills` 不冲突。
- `expected_mcp_tools` 和 `forbidden_mcp_tools` 不冲突。
- `requires_tool=false` 的 case 不声明 expected tool。
- 多轮 case 的 `prompt` 与最后一轮用户输入是否一致。

## 3. Live E2E 评测

Live 评测需要先启动 MCP 和 FastAPI：

```bash
make run-mcp
make run-api
```

也可以用：

```bash
make start-all
```

先跑一个小样本 smoke：

```bash
python scripts/evaluation/evaluate_astro_agent_dataset.py \
  --sample-per-category 1 \
  --save-events
```

跑完整 200 条：

```bash
python scripts/evaluation/evaluate_astro_agent_dataset.py \
  --dataset config/benchmarks/astro_agent_eval_dataset.json \
  --base-url http://localhost:8002 \
  --suite ability
```

只跑某一类：

```bash
python scripts/evaluation/evaluate_astro_agent_dataset.py \
  --category negative_ambiguous_safety
```

只跑指定 case：

```bash
python scripts/evaluation/evaluate_astro_agent_dataset.py \
  --case-id negative_013 \
  --save-events
```

## 4. 输出

每次 live 评测会创建一个目录：

```text
reports/evaluation/astro_agent/<timestamp>/
  summary.json
  cases.jsonl
  failures.md
  events/                 # 仅 --save-events 时生成
```

`summary.json` 汇总整体和分 category 指标：

- `tool_selection_accuracy`
- `skill_selection_accuracy`
- `mcp_selection_accuracy_observed`
- `tool_call_success_rate`
- `end_to_end_task_success_rate`
- `first_event_latency_p95_ms`
- `end_to_end_p95_latency_ms`

`cases.jsonl` 每行是一条 case 的评测结果，包括期望工具、实际工具、失败原因、首事件延迟、端到端耗时和最终回答预览。

`failures.md` 面向人工排查，按 case 展示失败原因和关键 trace 摘要。

## 5. MCP 工具打分口径

当前 API 事件流稳定暴露高层 skill/tool；底层 MCP tool 是否出现在 trace 中取决于具体执行路径。因此 live 脚本提供三种 MCP 打分模式：

| 模式 | 参数 | 行为 |
| --- | --- | --- |
| 观测模式 | `--mcp-scoring observed` | 默认值。若 trace 没有暴露 MCP 层，不因缺少 expected MCP 直接判失败；一旦观测到 MCP，就检查 expected/forbidden。 |
| 严格模式 | `--mcp-scoring strict` | expected MCP 未出现即失败。适合 MCP trace 已完整接入后使用。 |
| 忽略模式 | `--mcp-scoring ignore` | 不检查 expected MCP，仅检查已观测到的 forbidden MCP。 |

默认使用 `observed`，避免把“trace 不可见”误判为“工具选择错误”。后续如果事件流统一暴露底层 MCP 调用，可以在门禁中切换到 `strict`。

## 6. 当前打分边界

Live 脚本第一版采用确定性打分：

- 工具选择：比较 expected/forbidden skill，以及可观测 MCP tool。
- 稳定知识题：当前不强制 `RAGRetrieve`。数据集中带 `rag_optional` 的 case 可直接回答；如果系统实际调用 RAG 且该 case 未禁止 RAG，也不按“无工具题误用工具”扣分。
- 工具成功：根据 `final_answer` 事件中的 `tool_success_count` 和 `tool_error_count`，缺失时回退到 `tool_end` 事件。
- 端到端成功：要求响应成功、最终答案非空、工具选择通过、工具调用无错误；`should_clarify=true` 的 case 还要求回答中出现澄清提示词。
- 延迟：使用客户端视角 wall time，`first_event_latency_ms` 从请求发出到第一条 SSE data，`e2e_ms` 从请求发出到流结束。

脚本暂不做 LLM judge。`success_criteria` 中的语义要求仍需要后续接入 judge 后补充评分。

## 7. 建议使用方式

PR 或日常改动：

```bash
python scripts/evaluation/validate_astro_agent_dataset.py --expected-total 200
python scripts/evaluation/evaluate_astro_agent_dataset.py --sample-per-category 1
```

发布前或路由逻辑改动后：

```bash
python scripts/evaluation/evaluate_astro_agent_dataset.py \
  --suite ability \
  --mcp-scoring observed \
  --save-events
```

如果要让失败 case 触发非零退出：

```bash
python scripts/evaluation/evaluate_astro_agent_dataset.py \
  --sample-per-category 1 \
  --fail-on-failed-cases
```
