# 长期记忆功能说明

## 概述

AstroAgent 的长期记忆模块用于保存跨会话稳定信息，例如用户偏好、习惯、约束、背景和事实。当前实现已经从“直接维护用户画像”升级为分层架构：

```text
extract -> candidate -> confirm/promote -> memory item -> profile projection -> prompt injection
```

核心原则是：`memories`、`memory_candidates`、`memory_versions`、`memory_confirmations` 和事件/审计日志是主存；`user_profiles` 只是可重建的投影，用于快速展示和 prompt 注入。

## 核心模块

- `LongTermMemoryService`：统一门面，承载长期记忆主流程。
- `LongTermMemoryManager`：兼容层，保留旧接口并逐步委托给 service。
- `CandidateManager`：唯一候选管理入口，负责候选累计、阈值判断和晋升。
- `ProfileProjection`：从 active memories 重建 `user_profiles`。
- `LongTermMemoryRetriever`：按 query/task_type 选择最相关长期记忆。
- `PromptInjector`：把检索结果格式化为 prompt 片段。
- `LongTermMemoryDeletionService`：执行 tombstone 删除并写入 audit log。
- `LongTermMemoryRepository`：SQLite P0 存储实现。

## 数据分层

### 主存

- `memories`：正式长期记忆项。
- `memory_candidates`：候选记忆，只由 `CandidateManager` 维护。
- `memory_versions`：记忆值变更版本。
- `memory_confirmations`：待用户确认项。
- `memory_event_log`：生命周期事件。
- `ltm_deletion_audit`：删除审计。

### 投影

- `user_profiles`：从 active memories 聚合得到的用户画像。
- 删除 profile 不代表删除长期记忆主存；可通过 `ProfileProjection.rebuild()` 由 active memories 重建。

## 统一候选晋升规则

候选晋升只走 `CandidateManager`：

- 显式表达且 `explicit_bypass=True`：直接晋升。
- `occurrence_count >= candidate_occurrence_threshold`：晋升。
- `confidence >= candidate_confidence_threshold`：晋升。
- 临时请求不会进入长期记忆。

默认阈值：

```python
candidate_occurrence_threshold = 2
candidate_confidence_threshold = 0.6
candidate_explicit_bypass = True
```

旧的 `memory_events` 候选流保留为兼容 API，但不再作为主晋升链路。

## 编程接口

### 推荐：使用 LongTermMemoryService

```python
from src.memory.long_term_memory.service import LongTermMemoryService

ltm = LongTermMemoryService("./memory/long_term_memory/user_profiles.sqlite")

items = ltm.extract_and_store(
    user_message="我喜欢简短回答，我经常看火星",
    assistant_message="好的，我会记住",
    user_id="user_123",
)

profile = ltm.load_profile("user_123")
prompt_context = ltm.format_smart_prompt("user_123", "今晚用望远镜观测什么")
hits = ltm.explain_memory_hits("user_123", "今晚用望远镜观测什么")
```

### 兼容：继续使用 LongTermMemoryManager

```python
from src.memory.long_term_memory import LongTermMemoryManager

manager = LongTermMemoryManager()
manager.extract_and_store("我喜欢简短回答", "好的", "user_123")
prompt_context = manager.format_smart_prompt("user_123", "黑洞是什么")
```

`LongTermMemoryManager` 保留旧方法名，内部主要委托给 `LongTermMemoryService`。

## Prompt 注入

Prompt 注入不再只依赖 profile merge，而是按 query 和 task_type 检索 active memories：

- 任务类型包括 `general`、`qa`、`learning`、`observation` 等。
- 约束类记忆权重更高。
- 显式表达、用户确认、高置信度会提升排序。
- `explain_memory_hits()` 可返回命中原因，便于调试和展示。

## 删除能力

支持 tombstone + audit log：

```python
from src.memory.long_term_memory.models import LongTermMemoryDeletionRequest

ltm.delete(LongTermMemoryDeletionRequest(
    user_id="user_123",
    scope="memory",
    target_id="memory_id",
    reason="user request",
))
```

当前支持 scope：

- `memory`：删除单条正式记忆，查询结果不可见，并重建 profile projection。
- `candidate`：删除单条候选记忆。
- `profile`：只删除用户画像投影，不删除主存。
- `user_all`：删除该用户所有 memories/candidates/profile，并标记旧兼容 events。

删除操作会写入 `ltm_deletion_audit` 和 `memory_event_log`。P0 实现采用 tombstone，不做物理 purge。

## SQLite 配置

默认路径来自配置：

```python
LONG_TERM_MEMORY_PATH = "./memory/long_term_memory/user_profiles.sqlite"
DEFAULT_USER_ID = "anonymous"
```

Repository 初始化会自动创建或补齐表结构。P0 不引入向量数据库或重量级第三方依赖。

## 数据管理

查看主存：

```bash
sqlite3 memory/long_term_memory/user_profiles.sqlite
SELECT id, user_id, memory_type, key, value, status FROM memories;
SELECT id, user_id, memory_type, key, value, status FROM memory_candidates;
SELECT * FROM ltm_deletion_audit ORDER BY created_at DESC;
```

重建 profile projection：

```python
ltm.rebuild_profile("user_123")
```

## 测试

运行长期记忆单测：

```bash
pytest -q tests/unit/test_long_term_memory.py
```

当前重点覆盖：

- 旧 manager 接口兼容。
- CandidateManager 单轨候选晋升。
- Profile projection 可由 active memories 重建。
- 删除后 memory/candidate 查询不可见。
- Query-aware prompt injection 与命中解释。

## 后续扩展

- 将 retrieval 从规则评分升级为 embedding/rerank。
- 增加物理 purge job 和导出能力。
- 增强 confirmation 冲突处理策略。
- 增加长期记忆 replay 工具，根据 event/audit 重放状态。
