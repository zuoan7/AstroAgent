# AstroAgent

一个基于 AI 的天文学助手，提供天文知识查询、天体位置计算、天文事件预测等功能。

## 功能特性

- 🌌 天文知识问答与 RAG 检索
- 🪐 行星位置计算
- ⭐ 天体坐标转换
- 🌅 天体升落时间查询
- 🔭 天文事件预测
- 📷 多模态图像理解
- 📡 MCP 协议服务器

## 技术栈

- **框架**: LangChain, LangGraph
- **向量数据库**: ChromaDB
- **大模型**: Qwen (通义千问)
- **天文计算**: Skyfield, Astropy, PyEphem
- **Web框架**: FastAPI, Streamlit
- **MCP**: FastMCP

## 安装

1. 克隆仓库
```bash
git clone <repository-url>
cd AstroAgent
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥等配置
```

## 使用

### 运行 Streamlit 应用
```bash
streamlit run streamlit_app.py
```

### 启动 MCP 服务器
```bash
python mcp_server.py
```

### 运行 API 服务
```bash
python api.py
```

## 项目结构

```
AstroAgent/
├── agent/              # Agent 相关模块
├── rag/                # RAG 检索模块
├── data/               # 数据文件目录
├── uploads/            # 上传文件目录（已忽略）
├── vector_db/          # 向量数据库（已忽略）
├── logs/               # 日志目录（已忽略）
├── astronomy_tools.py  # 天文工具类
├── mcp_server.py       # MCP 服务器
├── streamlit_app.py    # Streamlit 应用
├── api.py              # API 服务
├── config.py           # 配置文件
└── requirements.txt    # 依赖列表
```

## 许可证

MIT License
