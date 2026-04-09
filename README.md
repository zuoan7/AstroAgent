# AstroAgent

一个基于 AI 的天文学助手，提供天文知识查询、天体位置计算、天文事件预测、多模态交互等功能。

## 功能特性

- 🌌 **天文知识问答** — 基于 RAG 混合检索（向量 + BM25）的天文知识库
- 🪐 **行星位置计算** — 支持太阳系行星实时位置、坐标转换、升落时间查询
- 🔭 **天文事件预测** — 周预报/月预报/今晚最佳观测目标推荐
- 📅 **观测计划生成** — 综合天气、天象、月相的智能观测计划
- 📡 **近地天体追踪** — 基于 NASA NEO API 的小行星飞掠事件查询
- 📷 **天文摄影参数** — 曝光时间估算、叠加建议、赤道仪校准指导
- 🖼️ **多模态图像理解** — 上传星空/天体图片，AI 识别并给出观测建议
- 🎙️ **语音问答** — 录制或上传音频，自动语音识别后回答天文问题
- 🧠 **用户画像记忆** — 短期对话记忆 + 长期用户偏好（SQLite 持久化）
- 🔄 **降级容错** — 工具调用失败时自动降级到联网搜索
- 📡 **MCP 协议服务器** — 基于 FastMCP 的标准 MCP 协议天文工具服务

## 技术栈

| 类别 | 技术 |
|------|------|
| Agent 框架 | LangChain, LangGraph |
| 大模型 | Qwen（通义千问）— qwen-max / qwen-vl-plus / paraformer-realtime-v2 |
| 向量数据库 | ChromaDB + DashScope Embeddings |
| 混合检索 | 向量检索 + BM25（rank-bm25） |
| 天文计算 | Skyfield, Astropy, PyEphem, Astroquery |
| Web 框架 | FastAPI（API 服务）, Streamlit（前端界面） |
| MCP 服务 | FastMCP（Streamable HTTP 传输） |
| 配置管理 | Pydantic Settings |
| 测试 | pytest, pytest-asyncio, pytest-cov |

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户界面层                             │
│  Streamlit (8501)  │  FastAPI API (8000)  │  MCP 客户端  │
└────────┬───────────┴──────────┬───────────┴──────┬──────┘
         │                      │                  │
         ▼                      ▼                  │
┌─────────────────────────────────────────┐        │
│            Agent 层                      │        │
│  AstroAgent → SkillManager → Router     │        │
│  ├─ StreamingService (SSE 事件流)       │        │
│  ├─ VisionService (图像理解)            │        │
│  ├─ SpeechService (语音识别)            │        │
│  ├─ FallbackService (降级容错)          │        │
│  └─ Memory (短期 + 长期用户画像)        │        │
└────────────────┬────────────────────────┘        │
                 │                                  │
         ┌───────┴───────┐                         │
         ▼               ▼                         │
┌──────────────┐ ┌──────────────┐                  │
│  RAG 检索层   │ │  MCP 服务器   │◄─────────────────┘
│  向量 + BM25  │ │  (8001)      │
└──────────────┘ └──────┬───────┘
                        │
                        ▼
               ┌────────────────┐
               │  天文计算层      │
               │  Skyfield /     │
               │  NASA API /     │
               │  高德天气 /      │
               │  Tavily 搜索    │
               └────────────────┘
```

## 快速开始

### 1. 环境要求

- Python >= 3.10
- 端口 8000、8001、8501 可用

### 2. 安装

```bash
git clone <repository-url>
cd AstroAgent

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp config/environments/.env.template .env
```

编辑 `.env` 文件，填入必要的 API 密钥：

| 变量 | 说明 | 必填 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 通义千问 API Key | ✅ |
| `NASA_API_KEY` | NASA API Key | ✅ |
| `AMAP_API_KEY` | 高德地图 API Key | ✅ |
| `TAVILY_API_KEY` | Tavily 联网搜索 Key | ❌ |
| `MODEL_NAME` | 主模型名称，默认 `qwen-max` | ❌ |
| `EMBEDDING_MODEL_NAME` | 嵌入模型，默认 `text-embedding-v2` | ❌ |
| `VISION_MODEL_NAME` | 视觉模型，默认 `qwen-vl-plus` | ❌ |
| `SPEECH_MODEL_NAME` | 语音模型，默认 `paraformer-realtime-v2` | ❌ |

### 4. 启动服务

**一键启动（推荐）：**

```bash
./scripts/start.sh          # 启动所有服务
./scripts/start.sh stop     # 停止所有服务
./scripts/start.sh restart  # 重启所有服务
./scripts/start.sh status   # 查看运行状态
```

**或使用 Makefile：**

```bash
make start-all    # 启动所有服务
make run-mcp      # 仅启动 MCP 服务器
make run-api      # 仅启动 API 服务
make run-ui       # 仅启动 Streamlit 界面
```

**或手动启动：**

```bash
# 1. MCP 服务器 (端口 8001)
python -m src.services.mcp_server

# 2. API 后端 (端口 8000)
python -m src.api.main

# 3. Streamlit 前端 (端口 8501)
streamlit run src/services/streamlit_app.py
```

启动后访问：
- Streamlit 界面：`http://localhost:8501`
- API 文档：`http://localhost:8000/docs`
- MCP 服务：`http://localhost:8001/mcp/`

## 技能系统

AstroAgent 采用**技能路由架构**，用户请求通过 SkillManager 分发到具体技能，技能内部编排一个或多个 MCP 工具完成复杂任务。

| 技能名称 | 说明 | 底层 MCP 工具 |
|----------|------|---------------|
| `ObservationPlanner` | 生成观测计划 | get_weather, get_weekly_events, get_tonight_best |
| `CelestialEventsForecast` | 天象事件预报 | get_weekly_events, get_monthly_events |
| `DeepSkyObservingGuide` | 深空天体观测指导 | get_astrophysical_object_info, get_galaxy_data, get_weather |
| `NEOTracker` | 近地天体追踪 | get_neo_data |
| `AstrophotographyCalculator` | 天文摄影参数计算 | 经验规则（不依赖 MCP） |
| `CelestialPositionCalculator` | 天体位置计算 | get_planet_position |
| `WeatherLookup` | 天气查询 | get_weather |
| `RAGRetrieve` | 天文知识检索 | 本地向量/BM25 检索 |

技能定义文件：[config/skills/definitions.yaml](config/skills/definitions.yaml)

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/query` | 文本问答（SSE 流式响应） |
| POST | `/query_with_image` | 图片 + 文本问答 |
| POST | `/query_with_audio` | 语音 + 文本问答 |
| POST | `/add_knowledge` | 添加知识到 RAG 系统 |
| GET | `/profile` | 获取用户画像 |
| DELETE | `/profile` | 删除用户画像 |
| POST | `/clear_memory` | 清空短期记忆 |

所有查询接口返回 SSE（Server-Sent Events）流，事件类型包括：

- `thinking` — Agent 思考过程
- `text` — 最终回答文本
- `image` — 返回的图片 URL
- `transcription` — 语音识别结果

## 项目结构

```
AstroAgent/
├── src/
│   ├── agent/                  # Agent 核心模块
│   │   ├── __init__.py         # AstroAgent 主类
│   │   ├── skill_manager.py    # 统一技能管理器
│   │   ├── streaming_service.py# 流式输出服务
│   │   ├── vision_service.py   # 图像理解服务
│   │   ├── speech_service.py   # 语音识别服务
│   │   ├── fallback_service.py # 降级容错服务
│   │   ├── param_parser.py     # 统一参数解析器
│   │   └── tools.py            # 已废弃（迁移到 SkillManager）
│   ├── astronomy/              # 天文计算模块
│   │   ├── __init__.py         # AstronomyTools / AstronomyEventsPredictor 统一入口
│   │   ├── base.py             # 星历数据管理
│   │   ├── planetary.py        # 行星位置计算、坐标转换
│   │   ├── celestial_databases.py  # SIMBAD/NED 天体数据库查询
│   │   ├── nasa_api.py         # NASA APOD/NEO API
│   │   ├── weather_service.py  # 高德天气服务
│   │   ├── search_service.py   # Tavily 联网搜索
│   │   └── events_predictor.py # 天象预测器
│   ├── skills/
│   │   └── router.py           # 技能路由层（技能 → MCP 工具编排）
│   ├── rag/                    # RAG 检索模块
│   │   ├── online_retriever.py # 混合检索器（向量 + BM25）
│   │   ├── bm25_retriever.py   # BM25 检索器
│   │   ├── offline_index.py    # 离线索引构建
│   │   └── build_bm25_index.py # BM25 索引构建脚本
│   ├── memory/
│   │   └── memory.py           # 短期记忆 + 长期用户画像（SQLite）
│   ├── services/
│   │   ├── mcp_server.py       # MCP 服务器（13 个工具）
│   │   └── streamlit_app.py    # Streamlit 前端
│   ├── api/
│   │   └── main.py             # FastAPI 后端
│   ├── core/
│   │   ├── config.py           # Pydantic Settings 统一配置
│   │   ├── errors.py           # 统一错误处理（ErrorCode + AgentError）
│   │   ├── logger.py           # 日志模块
│   │   └── constants.py        # 常量定义
│   └── utils/
│       └── helpers.py          # 工具函数
├── config/
│   ├── environments/
│   │   └── .env.template       # 环境变量模板
│   ├── prompts/
│   │   └── main.txt            # Agent Prompt 模板
│   ├── skills/
│   │   └── definitions.yaml    # 技能定义
│   └── ssl/
│       ├── cert.pem            # SSL 证书
│       └── key.pem             # SSL 密钥
├── data/
│   └── documents/              # RAG 知识库文档
│       ├── astronomy/          # 天文知识
│       ├── observation/        # 观测指南
│       └── science/            # 科普资料
├── tests/
│   ├── unit/                   # 单元测试
│   ├── integration/            # 集成测试
│   ├── boundary/               # 边界测试
│   └── performance/            # 性能测试
├── scripts/
│   ├── start.sh                # 一键启动脚本
│   ├── run_all_tests.py        # 测试运行脚本
│   └── debug/                  # 调试工具
├── pyproject.toml              # 项目配置 & 依赖
├── requirements.txt            # 生产依赖
├── requirements-dev.txt        # 开发依赖
├── Makefile                    # 常用命令
└── .github/workflows/
    └── test-suite.yml          # CI 测试流水线
```

## 开发

### 安装开发依赖

```bash
pip install -r requirements-dev.txt
```

### 常用命令

```bash
make help          # 查看所有可用命令
make test          # 运行测试
make test-cov      # 运行测试并生成覆盖率报告
make lint          # 代码检查（flake8, isort, black）
make format        # 代码格式化
make type-check    # 类型检查（mypy）
make check         # 运行所有检查（lint + type-check + test）
make clean         # 清理构建产物
```

### 添加新技能

1. 在 [config/skills/definitions.yaml](config/skills/definitions.yaml) 中添加技能定义
2. 在 [src/skills/router.py](src/skills/router.py) 的 `_registry` 中注册技能实现
3. 在 [src/agent/skill_manager.py](src/agent/skill_manager.py) 的 `tools_config` 中添加 LangChain Tool 配置
4. 编写测试

### 添加新 MCP 工具

1. 在 [src/astronomy/](src/astronomy/) 中实现工具逻辑
2. 在 [src/services/mcp_server.py](src/services/mcp_server.py) 中使用 `@mcp.tool()` 注册
3. 如需技能编排，在 router.py 中更新对应技能

## 许可证

MIT License
