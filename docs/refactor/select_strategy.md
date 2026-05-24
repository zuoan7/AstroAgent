# 记忆系统选择策略状态

本文档原为 `src/memory/` 选择策略改进方案。当前已改为状态文档，用于说明 v1 已收口范围、仍保留的 vNext backlog，以及后续维护边界。

## 当前结论

**记忆系统选择策略 v1 可以收口。**

v1 的目标是把短期上下文、工具证据、长期记忆注入、候选转正、摘要和反馈闭环从零散规则收敛到一套可追踪的选择策略。当前主链路已落地并通过全量回归：

- 统一任务画像：`TaskContextProfile` 已接入短期 planner、长期 retrieval/injection 和 Agent streaming 主链路。
- 场景自适应选择：短期上下文和工具证据已按 task/context scene 动态调整配额与权重。
- 多阶段筛选：短期上下文已具备 recent/lexical/focus 召回、MMR 去重和预算降级；长期记忆已具备策略召回、语义缓存召回、Top30 rerank、去重、类型配额和预算裁剪。
- 反馈闭环：长期记忆注入记录 `shown`，回答结束记录 `hit` / `miss`，并预留 `denied` 入口；反馈事件复用 `memory_event_log`。
- 可观测性：短期和长期选择过程均有 trace，短期 `decision_trace.candidates` 已输出候选明细。
- 策略配置外置：权重、阈值、配额、TTL 和词表已可通过 `config/memory/selection_strategy.yaml` 覆盖，代码默认值仍作为兜底。

v1 收口后，除 bugfix、测试补强和小范围稳定化外，不再继续扩大本轮选择策略改造范围。

## v1 Done

### 1. 短期上下文选择

状态：**Done**

落地内容：

- `BuildContextRequest` 支持可选 `task_profile` / `task_context_profile`。
- `RetrievalPlanner` 将 Agent `TaskProfile`、旧 `task_type`、`capability_hints` 和短期 focus 合成为 `TaskContextProfile`。
- 支持 `observation`、`computation`、`learning_qa`、`debugging`、`general` 场景配额。
- 支持 `focus_stack`、焦点漂移检测、地点/目标/工具类型 focus 加权。
- 候选召回使用 recent、lexical、focus 三路信号。
- 使用 MMR 做多样性去冗，避免重复消息、事实或工具结果占用预算。
- 预算不足时按降级链裁剪，并保留省略占位。
- `decision_trace` 输出 `profile`、`selected`、`omitted`、`fallbacks`、`scores`。
- `decision_trace.candidates` 已按 section 输出候选 `id/source_type/text/score/selected/recall_sources/fallback`。

主要文件：

- `src/memory/api/dto.py`
- `src/memory/application/memory_read_service.py`
- `src/memory/retrieval/planner.py`
- `src/memory/task_context.py`

### 2. 工具证据选择

状态：**Done**

落地内容：

- 工具证据打分已归一化，按 scene 使用不同权重。
- 支持 `params_hash`、`produced_at`、`effective_until`、`superseded_by` 等结构化 freshness 元数据。
- freshness 使用工具类型 TTL 和指数衰减，不再只依赖文本关键词。
- 支持 superseded 链路识别，保留最新成功结果和代表性失败。
- 支持同 `tool_type`、同 `target` 的多样性约束。
- 错误、超时、参数失败等失败信号可进入候选，避免重复错误调用。

主要文件：

- `src/memory/retrieval/planner.py`

### 3. 摘要生成与选择

状态：**Done**

落地内容：

- `build_context` 返回 `summary_needed` 后会自动触发 summary create/rebase。
- 自动触发条件包括未覆盖事件数量、未覆盖 token 估算、上下文压力、省略数量、话题切换和工具调用链完成。
- 摘要已结构化为固定 JSON schema：`topics`、`decisions`、`open_questions`、`established_facts`、`tool_results_index`。
- 支持 L1 segment snapshot 和 L2 working snapshot。
- `rebase_summary` 做结构化 merge，支持去重和 superseded decision 标记。
- 注入时按 query/focus 选择结构化摘要字段。

主要文件：

- `src/memory/application/compression_service.py`
- `src/memory/application/memory_maintenance_service.py`
- `src/memory/api/memory_service.py`
- `src/memory/retrieval/planner.py`

### 4. 长期记忆抽取 Gating

状态：**Done**

落地内容：

- 三层 gating 已落地：快速规则、最近 K=4 轮窗口聚合、LLM 判别 + 抽取一体。
- 支持稳定信号、临时信号、用户自指、设备/地点/技能等画像信号判断。
- 支持撤回/作废表达识别，并进入 revoke 流程。
- LLM 抽取输出支持 `solid` / `tentative` / `inferred` 等等级，供候选转正使用。
- 失败或关闭 LLM 时保守降级为规则抽取。

主要文件：

- `src/memory/long_term_memory/extractor.py`
- `src/memory/long_term_memory/service.py`

### 5. 长期记忆转正

状态：**Done**

落地内容：

- 候选转正从简单次数阈值升级为 `confidence * stability * consistency * source_weight`。
- 支持按 memory type 设置门槛：preference/habit、constraint、background、fact 分级处理。
- 出现次数使用 30 天衰减的 `effective_count`。
- 冲突被区分为 same、extension、refinement、unknown、conflict。
- FACT 默认不自动转正，BACKGROUND 仅允许可校验类别自动转正。
- 新转正记忆进入 probation，并在冲突时回退/归档。
- 用户否定反馈可降置信度，二次否定归档。

主要文件：

- `src/memory/long_term_memory/candidate.py`
- `src/memory/long_term_memory/quality.py`
- `src/memory/long_term_memory/service.py`

### 6. 长期记忆注入

状态：**Done**

落地内容：

- 乘法打分已替换为加法归一化评分。
- 打分组件包括 confidence、type_weight、source_bonus、query_relevance、semantic_similarity、recency、constraint_bonus、stale_penalty。
- 双阶段召回已落地：规则 Top100 + embedding 缓存语义 Top100 union。
- 对策略 Top30 做可降级 rerank。
- 注入预算与主上下文预算耦合：`clamp(0.1 * total_context_budget, 200, max_prompt_tokens)`。
- 支持同 key 去重、同 memory_type 配额和 constraint 优先。
- 反馈统计驱动 task_type x memory_type 自适应类型权重。

主要文件：

- `src/memory/long_term_memory/retrieval.py`
- `src/memory/long_term_memory/prompt_injector.py`

### 7. Agent / Streaming 主链路接入

状态：**Done**

落地内容：

- Streaming 开头先对原始 query 调 `router.profile()`，禁止回退到 `router.route()`。
- 原始 query 的 `TaskProfile` 传入短期 `BuildContextRequest`。
- 计算 `effective_query` 后，query 改变则重算 profile，否则复用原 profile。
- effective profile 用于长期记忆注入、retrieval explain、执行决策和反馈记录。
- 同步 `generate_response()` 和异步 `generate_events()` 主链路均已接入。
- 长期 user profile 在主链路中尽量只渲染一次，避免重复注入导致重复 shown 事件。

主要文件：

- `src/agent/streaming_service.py`

### 8. 统一反馈事件

状态：**Done**

落地内容：

- 新增 `MemoryFeedbackRecord`。
- 反馈事件复用现有 `memory_event_log`，事件类型为 `feedback_recorded`。
- 真正注入 prompt 时记录 `shown`。
- 回答结束后记录 `hit` / `miss`。
- 提供 `record_memory_feedback(... outcome="denied")` 作为通用入口。
- 提供 `get_feedback_records(user_id, memory_id=None, limit=50, offset=0)` 查询结构化反馈记录。
- `get_event_logs()` 支持可选 `event_type` 过滤。
- `/events` API 支持可选 `event_type` 查询参数。
- 保留 `metadata.injection_stats`、`by_task_type` 和 `last_feedback`，不破坏原自学习统计。

主要文件：

- `src/memory/feedback.py`
- `src/memory/long_term_memory/models.py`
- `src/memory/long_term_memory/repository.py`
- `src/memory/long_term_memory/event_log.py`
- `src/memory/long_term_memory/service.py`
- `src/api/main.py`

### 9. 策略配置外置

状态：**Done**

落地内容：

- 新增 `config/memory/selection_strategy.yaml` 作为默认策略配置文件。
- 新增 `MemorySelectionStrategyConfig` 和 `get_memory_selection_strategy_config(overrides=None)`。
- 加载顺序为代码默认值兜底、YAML 覆盖、构造/test overrides 最后覆盖。
- YAML 缺失或局部非法不会阻断启动；非法字段回退对应默认值并记录 warning。
- 短期上下文 section ratios、top_k、MMR、工具 TTL/权重、多样性配额和 summary trigger 阈值已接入配置。
- 长期 retrieval 权重/先验/关键词、injection 预算/配额/rerank 融合、候选转正阈值/来源权重、抽取 gating 词表和阈值已接入配置。
- 短期和长期选择 trace 增加 `strategy_config_version`，默认 `memory_selection_strategy_v1`。

主要文件：

- `config/memory/selection_strategy.yaml`
- `src/memory/selection_strategy_config.py`
- `src/memory/retrieval/planner.py`
- `src/memory/application/memory_maintenance_service.py`
- `src/memory/long_term_memory/retrieval.py`
- `src/memory/long_term_memory/prompt_injector.py`
- `src/memory/long_term_memory/extractor.py`
- `src/memory/long_term_memory/quality.py`

## vNext Backlog

以下内容不属于 v1 收口范围，保留为后续迭代。

### 1. 短期语义召回增强

当前短期上下文候选主要来自 recent、lexical、focus。vNext 可接入：

- 短期事件 embedding 缓存。
- 短期候选 cross-encoder 或 LLM rerank。
- 否定/蕴含关系语义判别。
- 针对中文同义词和天文简称的召回增强。

### 2. A/B 灰度框架

v1 没有实现新旧策略并行评估。vNext 可增加：

- 策略版本标识。
- shadow run trace。
- hit rate、denied rate、latency、token cost 指标。
- 按用户或会话灰度切换。

### 3. Dashboard / UI

v1 只提供服务层和 `/events` 事件查询能力。vNext 可增加：

- 反馈事件列表。
- 单条 memory 的 shown/hit/miss/denied 时间线。
- decision_trace 可视化。
- 候选转正和 probation 状态面板。

### 4. 自动 denied 推断

v1 只提供 `record_memory_feedback(... outcome="denied")` 入口，不自动判断用户否定。vNext 可增加：

- 用户回复中的否定语义检测。
- 否定目标 memory 对齐。
- denied 到候选撤回/归档的自动联动。

### 5. 工具写入端元数据审计

planner 已支持结构化 freshness 元数据，但仍需逐个审计工具写入端：

- 确认所有关键工具稳定写入 `params_hash`。
- 明确 `produced_at` 和 `effective_until` 的来源。
- 为外部天气、星历、可见性、目录查询分别定义 TTL。
- 减少依赖 output 文本推导元数据的 fallback。

### 6. 更强的隐式信号挖掘

v1 已有窗口 gating 和候选等级，但隐式信号仍可增强：

- 多 session 重复设备/地点识别。
- 用户多次纠错形成 constraint 候选。
- 时区、语言、单位偏好的自动候选。
- 低置信 inferred 候选的人工确认体验。

## 收口标准

v1 选择策略后续只接受以下类型改动：

- 回归测试失败修复。
- 选择 trace 字段兼容性修复。
- 明确 bugfix，例如重复 shown、profile 丢失、候选 selected 标记错误。
- 小范围测试补强。
- 文档与注释维护。

以下改动应新开 vNext 任务，不再塞入 v1：

- 新增 embedding/cross-encoder 短期召回主链路。
- 新增 Dashboard 或 UI。
- 新增 A/B 或策略灰度框架。
- 大规模配置系统改造。
- 自动 denied 推断与归档联动。

## 验证状态

最近一次完整验证：

```bash
pytest -q
```

结果：

```text
1003 passed, 2 warnings
```

两个 warning 均为 FastAPI `on_event` deprecation warning，不影响本次记忆选择策略收口。
