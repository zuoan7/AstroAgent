# AstroAgent

AstroAgent 是一个面向天文场景的 AI Agent 项目，围绕“天文知识问答 + 实时天文计算 + 观测辅助 + 多模态交互”构建。项目同时提供：

- `FastAPI` 主 API 服务，用于文本、图片、语音问答及记忆管理
- `FastMCP` 天文工具服务，用于向 Agent 暴露标准 MCP 工具能力
- `Vue 3 + Vite` 前端工作台，用于交互式调试与验证

项目当前实现采用 `LangChain + LangGraph` 驱动 Agent 编排，结合本地 RAG、MCP 工具调用、短期记忆和长期用户画像，实现面向天文问答与观测规划的智能助手。

## 项目概述

AstroAgent 的目标不是单纯的聊天，而是把自然语言理解、天文计算、外部数据源与领域知识库组合成一个可落地的天文 Agent 系统。当前版本已覆盖以下核心能力：

- 天文知识问答：基于本地 RAG 知识库进行天文概念解释、观测知识检索和背景补充
- 天体位置计算：支持行星位置、坐标转换、升落时间、当前可观测天体等计算
- 天象事件预测：支持周度/月度天象事件查询和“今晚最佳观测目标”推荐
- 观测计划生成：结合天气、天象和日期信息生成观测建议
- 深空观测指导：针对星系、星云等深空目标生成观测与设备建议
- 近地天体追踪：基于 NASA NEO 数据查询近地天体飞掠信息
- 天文摄影参数建议：输出曝光、叠加、极轴校准等摄影建议
- 多模态问答：支持图片问答和语音问答
- 用户记忆：包含短期会话记忆与基于 SQLite 的长期用户画像
- 降级容错：MCP 或工具失败时，可通过回退逻辑降低系统不可用风险

## 关键设计文档

- [Memory Directory Structure](docs/MEMORY_DIRECTORY_STRUCTURE.md)：记忆模块目录拆分、分层职责与迁移结构
- [Memory Module Documentation](docs/MEMORY_MODULE_DOCUMENTATION.md)：短期记忆、长期记忆、兼容层与仓储实现说明
- [Memory Context Evaluation](docs/memory_context_evaluation.md)：短期记忆上下文构建的 V1/V2 评估目标、指标口径、最终结果与回归命令
- [Agent Evaluation Dataset Design](docs/agent_evaluation_dataset_design.md)：Agent 工具选择、工具调用、端到端成功率和延迟评测的数据集字段与天文问题类型设计
- [AstroAgent Evaluation Runner](docs/astro_agent_evaluation_runner.md)：200 条 Agent benchmark 的静态校验、live SSE 评测脚本、指标口径与运行命令
- [Streaming Event Bus](docs/streaming-event-bus.md)：流式输出统一事件总线、适配器接口、插件机制与兼容性说明
- [Agent Compatibility Matrix](docs/agent_compatibility_matrix.md)：DAG Agent 重构后各兼容层、deprecated 接口、feature flags 与删除条件清单


## 核心功能

### 1. Agent 智能编排

- `AstroAgent` 是总入口，负责初始化 LLM、RAG、技能管理、记忆和流式输出
- Agent 采用 ReAct 风格，通过高层技能工具完成任务分发
- 工具由 `SkillManager` 统一注册，避免 Agent 直接依赖底层实现细节

### 2. 技能路由与 MCP 工具调用

- `SkillManager` 提供 8 个高层技能工具，其中包含 1 个 RAG 工具和 7 个领域技能
- `AstronomySkillRouter` 负责将技能请求路由到技能处理器或底层 MCP 工具
- `MCPClient` 通过 Streamable HTTP 与 `FastMCP` 服务器通信，支持会话初始化、并发工具调用和重连

### 3. RAG 检索

当前实现为增强型三级检索流水线：

1. 多路召回：向量检索 + BM25 + 天文实体检索 + 多模态检索
2. RRF 融合：通过 Reciprocal Rank Fusion 合并多路结果
3. 重排序与过滤：支持 Rerank、缓存、结果过滤和检索指标采集

### 4. 记忆系统

- 短期记忆：由 `src/memory/api/`、`src/memory/application/`、`src/memory/retrieval/`、`src/memory/infrastructure/`、`src/memory/domain/` 等分层模块管理，入口为 `MemoryService`
- 长期记忆：由 `src/memory/long_term_memory/` 管理用户偏好、事实、约束、事件与画像合并，入口为 `LongTermMemoryService`
- `src/memory/memory.py`：提供对旧调用方兼容的门面导出
- 用户画像提取：优先通过 LLM 结构化抽取，失败时降级为关键词提取

#### 记忆系统概览

AstroAgent 的记忆系统采用**事件化短期记忆 + 长期用户画像**分层设计，不是简单的最近 N 轮对话缓存：

- **短期记忆（事件溯源）**：消息和工具调用以 append-only 事件写入 EventStore（SQLite），工具结果原文落盘到 ArtifactStore，prompt 侧仅使用 output_digest 摘要。
- **上下文构建（检索式组装）**：`build_context` 由 `RetrievalPlanner` 按 token budget 从 task state、summary snapshot、messages、salient facts、tool calls 中检索和排序，组装成结构化上下文文本。
- **摘要快照自动触发**：assistant 消息写入后自动检查是否满足阈值，自动执行 create/rebase summary snapshot，长历史自动压缩为摘要，`build_context` 自动读取最新 snapshot。
- **长期记忆精细化提取**：用户偏好、习惯、约束、背景、事实等由 `LongTermMemoryService` 管理。提取触发条件已收敛为仅针对明确用户画像表达（偏好/设备/地点/技能），普通天文问题不再触发抽取。支持 LLM 提取和规则 fallback 双路径。
- **PromptBudgetManager 全局预算治理**：对 `DirectExecutor._run_simple_qa()` 和 `ResponseSynthesizer.synthesize()` 的关键 LLM 调用做统一的 prompt section 级预算裁剪。高优先级 section（query、instruction）优先保留，低优先级 section（chat_history、user_profile）超预算时被裁剪或丢弃。
- **ToolEvidenceCompactor 工具结果预算**：多工具调用场景下，工具结果摘要先在 `ResponseSynthesizer` 中经过 `ToolEvidenceCompactor` 做预算压缩（单工具 cap、总预算 cap、成功工具优先、失败工具缩简），再进入 PromptBudgetManager，避免工具输出撑爆 prompt。

详见 [Memory Current Design](docs/memory_current_design.md)、[Production Readiness](docs/production_readiness.md) 和 [Test Report](docs/test_report.md)。

### 5. 多模态交互

- 图片问答：上传图片后，调用视觉服务补全查询语义
- 语音问答：上传或录制音频后，先转写再进入 Agent 处理流程
- API 采用统一事件总线驱动的 SSE 流式返回，可逐步输出思考过程、文本、图片与转写结果

## 技术栈

| 类别 | 主要技术 |
| --- | --- |
| Agent 编排 | LangChain, LangGraph |
| 大模型 | Tongyi/Qwen（`ChatTongyi`、DashScope） |
| Web 服务 | FastAPI, Uvicorn, SlowAPI |
| MCP 服务 | FastMCP, langchain-mcp-adapters |
| 前端/调试 UI | Vue 3, Vite, Pinia |
| RAG | ChromaDB, DashScope Embeddings, rank-bm25 |
| 天文计算 | Skyfield, Astropy, PyEphem, Astroquery |
| 外部数据源 | NASA API, 高德天气 API, Tavily Search |
| 配置管理 | Pydantic Settings, python-dotenv |
| 存储 | SQLite, Chroma 持久化目录 |
| 测试 | pytest, pytest-asyncio, pytest-cov |

## 技术架构与模块划分

### 架构分层

```text
Client / UI
├── Vue3 Workbench
├── FastAPI 调用方
└── MCP Client / 外部集成

Application Layer
├── AstroAgent
├── StreamingService
├── VisionService
├── SpeechService
└── FallbackService

Skill Orchestration Layer
├── SkillManager
├── AstronomySkillRouter
├── Skill Handlers
└── MCPClient

Domain / Data Layer
├── Astronomy 模块
├── RAG 检索模块
├── Memory 模块
└── Config / Prompt / Skill Definitions

Infrastructure Layer
├── FastAPI 服务（8002）
├── FastMCP 服务（8001）
└── Vue3 Frontend（5173）
```

### 目录结构

```text
AstroAgent/
├── config/
│   ├── astronomy/              # 天象静态数据
│   ├── environments/           # 环境变量模板
│   ├── prompts/                # Agent Prompt 模板
│   └── skills/                 # 技能定义
├── data/
│   └── documents/              # RAG 文档语料
├── scripts/
│   └── start.sh                # 一键启动脚本（Unix-like 环境）
├── frontend/                   # Vue 3 + Vite 前端工作台
├── src/
│   ├── agent/                  # Agent 核心编排与多模态服务
│   ├── api/                    # FastAPI 服务入口
│   ├── astronomy/              # 天文计算与外部数据服务
│   ├── core/                   # 配置、日志、错误处理
│   ├── memory/                 # 分层记忆模块（core / infrastructure / short_term_memory / long_term_memory）
│   ├── rag/                    # 检索、融合、重排序、缓存
│   ├── services/               # MCP Server
│   ├── skills/                 # 技能路由、MCP 客户端、处理器
│   └── utils/                  # 通用工具
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── boundary/
│   └── performance/
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

### 关键模块说明

| 模块 | 说明 |
| --- | --- |
| `src/agent/__init__.py` | `AstroAgent` 主入口，初始化 LLM、RAG、记忆和技能工具 |
| `src/agent/skill_manager.py` | 高层技能注册与统一入口 |
| `src/agent/streaming_events.py` | 流式事件模型、校验器与文本/JSON/SSE 适配器 |
| `src/agent/streaming_service.py` | 统一事件总线驱动的流式输出与记忆写入 |
| `src/skills/router.py` | 技能路由，连接技能处理器与 MCP 工具 |
| `src/skills/skill_handlers.py` | 观测计划、天象预报、深空观测等复杂技能实现 |
| `src/skills/mcp_client.py` | Streamable HTTP MCP 客户端，负责会话与工具调用 |
| `src/services/mcp_server.py` | FastMCP 服务入口，注册天文工具并暴露 MCP 服务 |
| `src/api/main.py` | FastAPI 服务入口，暴露问答与记忆管理接口 |
| `src/rag/online_retriever.py` | 三级混合检索主流程 |
| `src/memory/memory.py` | 记忆兼容门面，复用新的短期/长期记忆实现 |
| `src/memory/api/memory_service.py` | 短期记忆 facade，聚合 write/read/maintenance/retrieval/deletion |
| `src/memory/long_term_memory/service.py` | 长期记忆主服务，负责画像抽取、融合、查询与持久化 |

## 核心实现逻辑

### 请求处理链路

1. 客户端调用 `FastAPI` 接口或通过 `Vue3` 前端触发请求
2. API 层根据 `user_id` 获取或创建会话，绑定短期记忆与流式服务
3. `StreamingService` 组织上下文、用户画像和工具执行过程
4. `AstroAgent` 调用 ReAct Agent，选择 RAG 或高层技能工具
5. 高层技能通过 `SkillManager -> AstronomySkillRouter -> MCPClient` 调用 MCP 工具，或直接访问 RAG
6. 底层天文服务返回结果后，统一内部事件总线生成标准事件，再按文本流 / JSON / SSE 适配输出给客户端

### MCP 工具集合

当前 `src/services/mcp_server.py` 注册的工具包括：

- `get_planet_position`
- `get_altaz`
- `coordinate_transformation`
- `get_rise_set_times`
- `get_current_sky_objects`
- `get_astrophysical_object_info`
- `get_galaxy_data`
- `get_nasa_apod`
- `get_neo_data`
- `get_weather`
- `web_search`
- `get_tonight_best`
- `get_weekly_events`
- `get_monthly_events`

说明：启动日志中显示“13 个工具”，但代码实际定义了 14 个 `@mcp.tool()` 工具；README 以代码实现为准。

### MCP 返回协议

所有 MCP 工具统一返回 JSON envelope 字符串，不再混用普通字符串、裸 dict、错误 dict 或手工 `json.dumps(...)`：

```json
{"ok": true, "data": "... or {...}", "meta": {"tool_name": "get_weather", "schema_version": "1.0"}}
```

```json
{"ok": false, "error": {"code": "TOOL_CALL_FAILED", "message": "...", "details": {}}, "meta": {"tool_name": "get_weather", "schema_version": "1.0"}}
```

实现约束：

- 服务端入口统一由 `src/core/errors.py::safe_tool_call` 包装输出。
- 协议模型、输入校验和解析辅助函数统一放在 `src/core/mcp_protocol.py`。
- `src/services/mcp_server.py` 中每个工具都使用对应的 Pydantic 输入模型校验参数。
- `src/skills/mcp_client.py`、Router、Streaming、Fallback 等上层模块按 envelope 解析，不再猜测底层返回形态。

### 高层技能集合

当前系统向 Agent 暴露的高层技能如下：

| 技能名 | 说明 |
| --- | --- |
| `RAGRetrieve` | 检索本地天文知识库 |
| `WeatherLookup` | 查询天气 |
| `ObservationPlanner` | 生成观测计划 |
| `CelestialEventsForecast` | 查询天象事件 |
| `DeepSkyObservingGuide` | 深空观测指导 |
| `NEOTracker` | 近地天体追踪 |
| `AstrophotographyCalculator` | 天文摄影参数建议 |
| `CelestialPositionCalculator` | 天体位置计算 |

## 环境配置与依赖管理

### 运行环境

- Python `>= 3.10`
- Node.js `>= 18`
- 推荐使用虚拟环境
- 需要可访问的端口：`8002`、`8001`、`5173`

### 主要依赖来源

- 生产依赖：`pyproject.toml` 与 `requirements.txt`
- 开发依赖：`requirements-dev.txt` 或 `pyproject.toml` 的 `.[dev]`

说明：

- `pyproject.toml` 是项目元数据与推荐安装入口
- `requirements.txt` 提供直接安装清单
- `Makefile` 和 `scripts/start.sh` 默认按 Unix-like 环境编写；在 Windows PowerShell 中通常需要手动执行 Python/npm 命令

### 必要环境变量

`.env` 由 `Pydantic Settings` 自动加载。当前代码默认读取项目根目录下的 `.env` 文件。

| 变量名 | 必填 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 是 | Qwen/DashScope API Key；未配置时主 Agent 无法初始化 |
| `NASA_API_KEY` | 建议 | NASA APOD/NEO 接口密钥 |
| `AMAP_API_KEY` | 建议 | 高德天气接口密钥 |
| `TAVILY_API_KEY` | 否 | 联网搜索 Key，作为增强/降级能力 |
| `MODEL_NAME` | 否 | 主模型，默认 `qwen-max` |
| `EMBEDDING_MODEL_NAME` | 否 | 向量嵌入模型，默认 `text-embedding-v2` |
| `VISION_MODEL_NAME` | 否 | 视觉模型，默认 `qwen-vl-plus` |
| `SPEECH_MODEL_NAME` | 否 | 语音模型，默认 `paraformer-realtime-v2` |
| `API_HOST` | 否 | API 监听地址，默认 `0.0.0.0` |
| `API_PORT` | 否 | API 端口，默认 `8002` |
| `MCP_PORT` | 否 | MCP 端口，默认 `8001` |
| `MCP_SERVER_URL` | 否 | MCP 客户端连接地址，默认 `http://localhost:8001/mcp` |
| `VITE_API_BASE_URL` | 否 | 前端直接请求的 API 地址；未配置时开发环境默认走 `/api` 代理 |
| `VITE_API_PROXY_TARGET` | 否 | Vite 开发代理目标，默认 `http://localhost:8002` |
| `LONG_TERM_MEMORY_PATH` | 否 | 长期记忆 SQLite 路径 |
| `VECTOR_DB_PATH` | 否 | Chroma 向量库路径 |
| `EPHEMERIS_FILE` | 否 | 星历文件路径，默认 `./data/ephemeris/de421.bsp` |

### 配置模板

仓库已提供模板文件：

- `config/environments/.env.template`

建议复制到项目根目录后使用：

```bash
cp config/environments/.env.template .env
```

注意：模板注释写的是复制为 `.env.local`，但当前代码实际只自动读取 `.env`。如使用 `.env.local`，需要自行导出环境变量或调整配置加载方式。

## 安装与部署

### 方式一：基于 `pyproject.toml` 安装

```bash
git clone <repository-url>
cd AstroAgent

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install -e .
```

安装前端依赖：

```bash
cd frontend
npm install
cd ..
```

如需开发依赖：

```bash
pip install -e ".[dev]"
```

### 方式二：基于 requirements 安装

```bash
git clone <repository-url>
cd AstroAgent

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

安装前端依赖：

```bash
cd frontend
npm install
cd ..
```

### 启动前检查

请确认以下资源已准备妥当：

- `.env` 已配置必要 API Key
- `EPHEMERIS_FILE` 指向有效星历文件
- `data/documents/` 中有可供 RAG 使用的文档
- 目标端口未被占用

### 启动方式

#### 1. 手动启动，适合所有平台

先启动 MCP 服务：

```bash
python -m src.services.mcp_server
```

再启动 FastAPI：

```bash
python -m src.api.main
```

最后启动 Vue3 前端：

```bash
cd frontend
npm install
npm run dev
```

#### 2. 使用 Makefile

```bash
make run-mcp
make run-api
make run-ui
```

一键启动：

```bash
make start-all
```

#### 3. 使用脚本一键启动

```bash
./scripts/start.sh
./scripts/start.sh stop
./scripts/start.sh restart
./scripts/start.sh status
```

说明：`scripts/start.sh` 依赖 `bash`、`nohup`、`lsof`、`ss`、`ssh` 等 Unix 工具，不适用于原生 Windows PowerShell 环境。

### 服务访问地址

- FastAPI: `http://localhost:8002`
- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`
- MCP Endpoint: `http://localhost:8001/mcp/`
- Vue3 Frontend: `http://localhost:5173`

## 使用指南

### 文本问答

适用于天文知识、观测建议、事件预测等场景：

```bash
curl -N -X POST "http://localhost:8002/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"今晚北京适合观测什么？\",\"user_id\":\"demo-user\"}"
```

### 图片问答

```bash
curl -N -X POST "http://localhost:8002/query_with_image" \
  -F "query=这张图片里的天体是什么？" \
  -F "user_id=demo-user" \
  -F "image=@./example.jpg"
```

### 语音问答

```bash
curl -N -X POST "http://localhost:8002/query_with_audio" \
  -F "query=请回答音频里的问题" \
  -F "user_id=demo-user" \
  -F "audio=@./question.wav"
```

### 添加知识到 RAG

```bash
curl -X POST "http://localhost:8002/add_knowledge" \
  -H "Content-Type: application/json" \
  -d "{\"knowledge\":[\"火星是太阳系第四颗行星\"],\"user_id\":\"demo-user\"}"
```

### 清理会话记忆

```bash
curl -X POST "http://localhost:8002/clear_memory?user_id=demo-user"
```

### Vue3 前端

前端代码位于 `frontend/`，由 `Vue 3 + Vite + Pinia` 构建，默认开发地址为 `http://localhost:5173`。

开发环境中，前端默认使用 `/api` 代理访问 FastAPI，代理目标为 `http://localhost:8002`。如需连接其他后端地址，可设置 `VITE_API_PROXY_TARGET`；如需绕过代理直接请求后端，可设置 `VITE_API_BASE_URL`。

- 执行总览：展示每轮总耗时、工具调用数、证据条目数和记忆命中数
- 执行时间线：逐条展示工具名称、输入参数、返回摘要与单次调用耗时
- 最终回答区：展示最终答案、置信度以及本轮聚合统计
- 记忆与证据面板：同步展示长期记忆命中与证据引用来源

当前后端未直接托管前端静态资源，开发阶段以前后端分离方式运行。

## API 接口文档

### 1. 健康检查与基础信息

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 基础信息与 Agent 可用状态 |
| `GET` | `/health` | 健康检查、Agent 状态、活跃会话数 |

### 2. 问答接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/query` | 文本问答，SSE 流式响应 |
| `POST` | `/query_with_image` | 图片 + 文本问答，SSE 流式响应 |
| `POST` | `/query_with_audio` | 音频 + 文本问答，SSE 流式响应 |

#### `/query` 请求体

```json
{
  "query": "今晚上海能看到木星吗？",
  "user_id": "demo-user"
}
```

#### SSE 事件示例

```text
data: {"type":"thinking","content":"正在分析问题..."}

data: {"type":"text","content":"今晚上海可以观测到木星..."}
```

当前实现中常见事件类型：

- `thinking`：中间推理或处理状态
- `text`：最终文本内容
- `image`：图片 URL
- `transcription`：语音识别文本
- `plan_update` / `step_start` / `step_end`：阶段规划与进度状态
- `tool_start` / `tool_end`：工具名称、参数、摘要和单次耗时
- `final_answer`：最终答案、整轮总耗时、工具统计、证据统计和记忆命中

说明：

- 这些对外事件由统一内部事件总线生成，再经 `FrontendJsonEventAdapter` / `SSEEventAdapter` 转换输出。
- 内部主事件协议已收口为 `ExecutionEvent`；上述事件名属于前端兼容输出层，不要求前端立刻迁移。
- 纯文本流、JSON 事件流和 SSE 流已收敛到同一底层事件序列，减少不同输出路径之间的逻辑漂移。
- `final_answer` 事件当前会额外携带 `total_duration_sec`、`tool_count`、`tool_success_count`、`tool_error_count`、`evidence_count`、`memory_hit_count` 等前端可观测字段。


### 3. 知识库接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/add_knowledge` | 动态写入知识到 RAG |

请求体：

```json
{
  "knowledge": ["木星的大红斑是一个巨型风暴系统"],
  "user_id": "demo-user"
}
```

### 4. 用户画像与记忆接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/profile` | 获取长期用户画像 |
| `DELETE` | `/profile` | 删除长期用户画像 |
| `POST` | `/clear_memory` | 清理当前用户短期会话记忆 |

### 5. 自动化 API 文档

FastAPI 已内置 OpenAPI 文档：

- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`



## 未来路线图

### P1 — 核心稳定性

- **ReAct scratchpad 压缩**：ReAct AgentExecutor 内部 scratchpad 无界增长，需要在 AgentExecutor 层面改造
- **Token-level prompt budget**：当前为字符级预算，未来可升级为 tokenizer-based 精确预算
- **线上可观测性与 tracing**：补充 token 使用监控、记忆命中率统计、LLM 调用耗时分布

### P2 — 降级与细节完善

- **Compact 失败丢弃旧 API round 重试**：多轮 ReAct 场景下系统性降级策略
- **DirectExecutor 单工具摘要裁剪**：`_run_tool_task()` → `synthesize_direct()` 路径的轻量裁剪
- **工具结果写入阶段 compacted evidence event**：在 MemoryWriteService 层做更早期的压缩

### P3 — 体验与多租户

- **记忆可视化管理界面**：前端支持查看/编辑/删除用户画像记忆
- **多用户/多租户治理增强**：用户间记忆隔离、跨会话记忆继承
- **更细粒度的记忆冲突解决**：长期记忆更新时的冲突检测与合并策略


## 许可证

本项目采用 [MIT License](LICENSE)。
