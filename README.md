# AstroAgent

AstroAgent 是一个面向天文场景的 AI Agent 项目，围绕“天文知识问答 + 实时天文计算 + 观测辅助 + 多模态交互”构建。项目同时提供：

- `FastAPI` 主 API 服务，用于文本、图片、语音问答及记忆管理
- `FastMCP` 天文工具服务，用于向 Agent 暴露标准 MCP 工具能力
- `Streamlit` 测试界面，用于本地交互验证

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

- 短期记忆：维护最近对话窗口，用于上下文续写
- 长期记忆：将用户偏好、习惯、限制持久化到 SQLite
- 用户画像提取：优先通过 LLM 结构化抽取，失败时降级为关键词提取

### 5. 多模态交互

- 图片问答：上传图片后，调用视觉服务补全查询语义
- 语音问答：上传或录制音频后，先转写再进入 Agent 处理流程
- API 采用 SSE 流式返回，可逐步输出思考过程与最终文本

## 技术栈

| 类别 | 主要技术 |
| --- | --- |
| Agent 编排 | LangChain, LangGraph |
| 大模型 | Tongyi/Qwen（`ChatTongyi`、DashScope） |
| Web 服务 | FastAPI, Uvicorn, SlowAPI |
| MCP 服务 | FastMCP, langchain-mcp-adapters |
| 前端/调试 UI | Streamlit |
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
├── Streamlit 调试界面
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
├── FastAPI 服务（8000）
├── FastMCP 服务（8001）
└── Streamlit 服务（8501）
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
├── src/
│   ├── agent/                  # Agent 核心编排与多模态服务
│   ├── api/                    # FastAPI 服务入口
│   ├── astronomy/              # 天文计算与外部数据服务
│   ├── core/                   # 配置、日志、错误处理
│   ├── memory/                 # 短期/长期记忆
│   ├── rag/                    # 检索、融合、重排序、缓存
│   ├── services/               # MCP Server 与 Streamlit App
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
| `src/skills/router.py` | 技能路由，连接技能处理器与 MCP 工具 |
| `src/skills/skill_handlers.py` | 观测计划、天象预报、深空观测等复杂技能实现 |
| `src/skills/mcp_client.py` | Streamable HTTP MCP 客户端，负责会话与工具调用 |
| `src/services/mcp_server.py` | FastMCP 服务入口，注册天文工具并暴露 MCP 服务 |
| `src/api/main.py` | FastAPI 服务入口，暴露问答与记忆管理接口 |
| `src/rag/online_retriever.py` | 三级混合检索主流程 |
| `src/memory/memory.py` | 短期记忆、用户画像持久化与提取逻辑 |

## 核心实现逻辑

### 请求处理链路

1. 客户端调用 `FastAPI` 接口或通过 `Streamlit` 触发请求
2. API 层根据 `user_id` 获取或创建会话，绑定短期记忆与流式服务
3. `StreamingService` 组织上下文、用户画像和工具执行过程
4. `AstroAgent` 调用 ReAct Agent，选择 RAG 或高层技能工具
5. 高层技能通过 `SkillManager -> AstronomySkillRouter -> MCPClient` 调用 MCP 工具，或直接访问 RAG
6. 底层天文服务返回结果，流式组装成 SSE 事件输出给客户端

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
- 推荐使用虚拟环境
- 需要可访问的端口：`8000`、`8001`、`8501`

### 主要依赖来源

- 生产依赖：`pyproject.toml` 与 `requirements.txt`
- 开发依赖：`requirements-dev.txt` 或 `pyproject.toml` 的 `.[dev]`

说明：

- `pyproject.toml` 是项目元数据与推荐安装入口
- `requirements.txt` 提供直接安装清单
- `Makefile` 和 `scripts/start.sh` 默认按 Unix-like 环境编写；在 Windows PowerShell 中通常需要手动执行 Python/Streamlit 命令

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
| `API_PORT` | 否 | API 端口，默认 `8000` |
| `MCP_PORT` | 否 | MCP 端口，默认 `8001` |
| `MCP_SERVER_URL` | 否 | MCP 客户端连接地址，默认 `http://localhost:8001/mcp` |
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

最后启动 Streamlit：

```bash
streamlit run src/services/streamlit_app.py
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

- FastAPI: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- MCP Endpoint: `http://localhost:8001/mcp/`
- Streamlit: `http://localhost:8501`

## 使用指南

### 文本问答

适用于天文知识、观测建议、事件预测等场景：

```bash
curl -N -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"今晚北京适合观测什么？\",\"user_id\":\"demo-user\"}"
```

### 图片问答

```bash
curl -N -X POST "http://localhost:8000/query_with_image" \
  -F "query=这张图片里的天体是什么？" \
  -F "user_id=demo-user" \
  -F "image=@./example.jpg"
```

### 语音问答

```bash
curl -N -X POST "http://localhost:8000/query_with_audio" \
  -F "query=请回答音频里的问题" \
  -F "user_id=demo-user" \
  -F "audio=@./question.wav"
```

### 添加知识到 RAG

```bash
curl -X POST "http://localhost:8000/add_knowledge" \
  -H "Content-Type: application/json" \
  -d "{\"knowledge\":[\"火星是太阳系第四颗行星\"],\"user_id\":\"demo-user\"}"
```

### 清理会话记忆

```bash
curl -X POST "http://localhost:8000/clear_memory?user_id=demo-user"
```

### Streamlit 界面

`src/services/streamlit_app.py` 提供三类交互标签页：

- 文本问答
- 图片问答
- 语音问答

该界面更偏向开发测试和功能验证，而不是生产级前端。

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

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 测试与质量保障

### 测试目录

- `tests/unit/`：单元测试
- `tests/integration/`：集成测试
- `tests/boundary/`：边界测试
- `tests/performance/`：性能测试

### 常用命令

```bash
make test
make test-cov
make lint
make format
make type-check
make check
```

或直接使用 `pytest`：

```bash
pytest tests/ -v
```

## 贡献规范

欢迎提交 Issue 和 Pull Request。建议遵循以下约定：

1. 从主分支创建功能分支或修复分支
2. 修改前先确认 README、配置和实现保持一致
3. 新增能力时同步补充测试
4. 如新增技能，需要同步更新：
   - `config/skills/definitions.yaml`
   - `src/skills/router.py` 或 `src/skills/skill_handlers.py`
   - `src/agent/skill_manager.py`
5. 提交前运行至少一轮相关测试与格式化检查

推荐本地开发流程：

```bash
pip install -e ".[dev]"
pre-commit install
make check
```

## 已知事项

- README 以当前代码实现为准，修正了旧文档中的编码损坏问题
- `.env.template` 的注释与实际加载路径存在差异，运行时请优先使用项目根目录 `.env`
- `Makefile` 中的 `clean` 目标使用了 Unix 命令，在 Windows 原生环境可能不可直接执行
- `start.sh` 明确面向 Unix-like 服务器/开发环境
- 项目元数据中的仓库地址和维护者邮箱仍为占位信息，正式开源前建议替换为真实信息

## 许可证

本项目采用 [MIT License](LICENSE)。

## 维护者与联系方式

当前 `pyproject.toml` 中登记的维护者信息为：

- `AstroAgent Developer`
- `developer@example.com`

同时，项目仓库元数据中预留了以下入口占位符：

- Homepage: `https://github.com/yourusername/astroagent`
- Repository: `https://github.com/yourusername/astroagent`
- Issues: `https://github.com/yourusername/astroagent/issues`

如果你计划将该项目公开发布，建议先将上述占位信息替换为真实维护者与仓库地址。
