# 测试报告

> AstroAgent 记忆系统生产化改进 (Phase 1-5) 测试结果。

## 测试命令

```bash
# 完整测试套件
pytest -q

# 仅单元测试
pytest tests/unit -q

# Phase 1 — 长期记忆提取器
pytest tests/unit/test_long_term_memory_extractor_phase1.py -q

# Phase 2 — PromptBudgetManager
pytest tests/unit/test_prompt_budget_manager.py -q
pytest tests/unit/test_direct_executor_prompt_budget.py -q
pytest tests/unit/test_response_synthesizer_prompt_budget.py -q

# Phase 3 — Summary Snapshot 自动触发
pytest tests/unit/test_summary_autocompact_phase3.py -q

# Phase 4 — 工具结果预算治理
pytest tests/unit/test_tool_evidence_budget_phase4.py -q
pytest tests/unit/test_response_synthesizer_tool_evidence_budget.py -q
```

## 测试结果（最终运行）

```
pytest -q
752 passed, 1 skipped, 2 warnings in 37.34s
```

### 按分类统计

| 类别 | 通过 | 跳过 | 失败 |
| --- | --- | --- | --- |
| 单元测试 (tests/unit) | 604 | 1 | 0 |
| 集成测试 (tests/integration) | 148 | 0 | 0 |
| **合计** | **752** | **1** | **0** |

### 新增测试文件（Phase 1-4）

| 文件 | 用例数 | 覆盖内容 |
| --- | --- | --- |
| `test_long_term_memory_extractor_phase1.py` | 77 | 触发条件、LLM 禁用、LLM 配置、fallback 保守性 |
| `test_prompt_budget_manager.py` | 26 | required 保留、优先级排序、max_chars、预算不溢出 |
| `test_direct_executor_prompt_budget.py` | 4 | RAG context 裁剪、query 保留、开关禁用 |
| `test_response_synthesizer_prompt_budget.py` | 6 | tool outputs 预算、query/instruction 保留、sources 不丢失 |
| `test_summary_autocompact_phase3.py` | 12 | 阈值触发、assistant-only、rebase、失败隔离 |
| `test_tool_evidence_budget_phase4.py` | 17 | 单工具 cap、总预算、success 优先、error 截断 |
| `test_response_synthesizer_tool_evidence_budget.py` | 7 | Compact 接入、异常 fallback、sources/tools_used 兼容 |

### 修改的已有测试

| 文件 | 修改说明 |
| --- | --- |
| `tests/unit/test_long_term_memory.py` | 更新 `test_should_attempt` 断言匹配 Phase 1 行为 |
| `tests/integration/test_astronomy_integration.py` | 移除 `frequent_topics` 断言匹配 Phase 1 行为 |

## Skipped 测试

- `tests/unit/test_memory_refactor_p0.py::V1EvaluationTestSuite` — V1 评估套件，需外部 API key

## Warnings

- `DeprecationWarning: on_event is deprecated` — FastAPI 已知弃用提示，来自 `src/api/main.py:354`，不影响功能

## 未覆盖但有已知计划的测试场景

- ReAct scratchpad 压缩（Phase 5+）
- DirectExecutor 单工具摘要裁剪（Phase 5+）
- Token-level prompt budget（Phase 5+）
- 记忆系统性能基准测试（Phase 5+）
