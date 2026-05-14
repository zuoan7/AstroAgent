# 记忆系统生产化改进报告

> 记录 Phase 1-4 各阶段解决的问题、修改方向和剩余风险。

---

## Phase 0：架构冻结与边界确认

**问题**：记忆系统缺乏明确的架构文档和模块边界，后续修改很容易演变为大重构。

**产出**：
- `docs/memory_current_design.md`：9 章完整设计文档，覆盖写入/读取/摘要/工具结果/长期记忆/已知问题
- `docs/memory_refactor_boundary.md`：明确 12 个核心模块不可推倒重写，定义允许和禁止的修改方式
- README 新增记忆系统概览小节

**结论**：当前分层事件化架构保留，不做推倒重写，后续只做小步增量修复。

---

## Phase 1：长期记忆提取链路稳定性收敛

**问题**：
1. `should_attempt_extraction()` 触发条件过宽——任何包含天体关键词的消息都会触发 LLM 抽取
2. 测试环境无法禁用 LLM 抽取，行为不可预测
3. `extract_with_llm()` 使用主模型 (`MODEL_NAME`)、15s timeout、无 retry 控制
4. Fallback 规则从普通天文主题生成 frequent_topics / observation_type，污染长期记忆

**修改**：
- `src/core/config.py`：新增 `LTM_EXTRACT_ENABLED`、`LTM_LLM_EXTRACT_ENABLED`、`LTM_EXTRACT_TIMEOUT_SECONDS`、`LTM_EXTRACT_MAX_RETRIES`、`LTM_EXTRACT_MODEL_NAME`
- `src/core/llm_factory.py`：`build_chat_model()` 增加 `max_retries` 可选参数，默认保持 2
- `src/memory/long_term_memory/extractor.py`：
  - 重写 `should_attempt_extraction`：仅当用户表达明确画像信号（偏好/设备/地点/技能/长期记忆关键词）时返回 True
  - `extract_from_conversation`：增加 `LTM_EXTRACT_ENABLED` 总开关和 `LTM_LLM_EXTRACT_ENABLED` LLM 开关
  - `extract_with_llm`：使用轻量模型、短 timeout(6s)、0 retry
  - Fallback：移除 frequent_topics 从天文关键词生成、移除 observation_type 自动检测

**测试**：77 个单元测试，覆盖触发条件、LLM 禁用、配置使用、fallback 保守性

---

## Phase 2：PromptBudgetManager 全局上下文预算治理

**问题**：
- `DirectExecutor._run_simple_qa()` 使用硬编码 `user_profile[:400]`、`chat_history[:800]`、`context[:2400]`
- `ResponseSynthesizer.synthesize()` 使用硬编码 `user_profile[:400]`、`chat_history[:600]`、`collected_outputs[:5000]`
- 缺少统一的、可解释的 prompt budget 机制

**修改**：
- `src/core/config.py`：新增 `PROMPT_BUDGET_ENABLED`、`PROMPT_MAX_INPUT_CHARS`、`PROMPT_RESERVED_OUTPUT_CHARS`、`PROMPT_SECTION_MIN_CHARS`、`PROMPT_BUDGET_LOG_TRIMMED`
- `src/agent/policies/prompt_budget.py`（新建）：`PromptSection`、`PromptBudgetResult`、`PromptBudgetManager`
- `src/agent/execution/direct_executor.py`：`_run_simple_qa()` 接入 PromptBudgetManager
- `src/agent/response_synthesizer.py`：`synthesize()` 接入 PromptBudgetManager

**核心规则**：
- Priority 越高越优先保留，required section（query、instruction）保证不丢弃
- Per-section `max_chars` 先做 cap，再做全局 fitting
- 输出 `trimmed_sections` / `dropped_sections` 供日志和测试
- `PROMPT_BUDGET_ENABLED=False` 时保留旧逻辑

**测试**：26 个 PromptBudgetManager 测试 + 4 个 DirectExecutor + 6 个 ResponseSynthesizer

---

## Phase 3：Summary Snapshot 自动触发闭环

**问题**：
- `MemoryConfig` 中已有 `MEMORY_SUMMARY_TRIGGER_MESSAGES` 和 `MEMORY_SUMMARY_TRIGGER_TOKENS`
- `create_summary_snapshot()` 和 `rebase_summary_snapshot()` 已实现但只能手动调用
- 长历史无法自动压缩为摘要

**修改**：
- `src/core/config.py`：新增 `MEMORY_AUTO_SUMMARY_ENABLED`、`MEMORY_SUMMARY_MIN_NEW_EVENTS`
- `src/memory/application/memory_maintenance_service.py`：新增 `SummaryTriggerDecision` dataclass + `should_create_summary_snapshot()` + `_estimate_event_tokens()`
- `src/memory/api/memory_service.py`：`append_message()` 在 assistant 消息后调用 `_maybe_auto_summary_snapshot()`

**触发逻辑**：
- 首次：uncovered 事件数 >= `MEMORY_SUMMARY_TRIGGER_MESSAGES` 或 token 估算 >= `MEMORY_SUMMARY_TRIGGER_TOKENS` → create
- 已有：uncovered 新事件数 >= `MEMORY_SUMMARY_MIN_NEW_EVENTS` 或 token 估算 >= `MEMORY_SUMMARY_TRIGGER_TOKENS` → rebase
- User 消息后不触发，assistant 后触发
- 异常不影响 `append_message` 主流程

**测试**：12 个单元测试，覆盖阈值触发、assistant-only、rebase、开关关闭、token 阈值、失败隔离、build_context 集成

---

## Phase 4：工具结果预算治理与轻量 Microcompact

**问题**：
- 多工具调用时，多个 summary 拼接总长度可能撑爆 prompt
- 缺少单工具 summary 最大长度限制
- 失败工具摘要占空间过多

**修改**：
- `src/core/config.py`：新增 `TOOL_EVIDENCE_BUDGET_ENABLED`、`TOOL_EVIDENCE_MAX_SINGLE_CHARS`、`TOOL_EVIDENCE_MAX_TOTAL_CHARS`、`TOOL_EVIDENCE_ERROR_MAX_CHARS`、`TOOL_EVIDENCE_COMPACTED_MAX_CHARS`
- `src/agent/policies/tool_evidence_budget.py`（新建）：`ToolEvidenceItem`、`ToolEvidenceBudgetResult`、`ToolEvidenceCompactor`
- `src/agent/response_synthesizer.py`：`synthesize()` 先通过 `ToolEvidenceCompactor.compact_skill_results()` 压缩，再进入 PromptBudgetManager

**核心规则**：
- Step 1：单工具 cap（success 800、error 300）
- Step 2：按 priority 排序（success=60 + sources boost，error=30）
- Step 3：累积至总预算 3000，超出裁剪/丢弃
- Step 4：二次压缩至 compacted_max=1800
- 不修改 `FinalResponse.sources` / `tools_used` / `structured_payload`

**测试**：17 个 compactor 测试 + 7 个 ResponseSynthesizer 集成测试

---

## 剩余风险与后续计划

| 风险 | 严重度 | 计划 |
| --- | --- | --- |
| ReAct scratchpad 无界增长 | 高 | Phase 5+ 在 AgentExecutor 层面改造 |
| 字符级预算精度有限 | 中 | 未来升级为 tokenizer-based |
| DirectExecutor 单工具场景无裁剪 | 低 | 后续补充轻量裁剪 |
| 工具结果写入阶段未 compacted | 低 | 后续在 MemoryWriteService 层补充 |

详见 README 中的 Roadmap 小节。
