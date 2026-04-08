# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AstroAgent is an AI-powered astronomy assistant providing astronomical knowledge querying, celestial position calculation, astronomical event prediction, and multi-modal capabilities. The project uses LangChain/LangGraph for agent orchestration, Qwen (Tongyi Qianwen) as the primary LLM, and integrates various astronomy libraries (Skyfield, Astropy, PyEphem).

## Architecture

### Core Components

1. **Agent Layer** (`agent/`): Main agent implementation with skill management, streaming services, and multi-modal support
   - `AstroAgent` class in `agent/__init__.py` is the main entry point
   - `SkillManager` handles skill routing and execution
   - `StreamingService` provides real-time event streaming
   - `VisionService` and `SpeechService` for multi-modal inputs

2. **Astronomy Tools** (`astronomy/`): Domain-specific astronomical calculations
   - `AstronomyTools` base class with planetary calculations
   - `AstronomyEventsPredictor` for celestial event forecasting
   - `NASA API` integration for APOD and NEO data
   - Weather service for observation planning

3. **RAG System** (`rag/`): Hybrid retrieval system combining vector search and BM25
   - `OnlineRetriever` for real-time knowledge retrieval
   - `BM25Retriever` for keyword-based search
   - Offline indexing capabilities

4. **Memory System** (`memory/`): Short-term and long-term memory management
   - `ShortTermMemory` for conversation context
   - `LongTermMemory` with SQLite storage for user profiles

5. **Skills System** (`skills.py`): Skill routing layer that maps user requests to MCP tools
   - Maps skill names from `skill.yaml` to underlying MCP tools
   - Handles parameter transformation and error handling

### Service Architecture

The system runs three main services:
- **MCP Server** (port 8001): Model Context Protocol server exposing astronomy tools
- **API Backend** (port 8000): FastAPI service with streaming endpoints
- **Streamlit Frontend** (port 8501): Web interface with HTTPS support

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys (DashScope, NASA, AMAP, Tavily)
```

### Running Services

**Start all services (recommended):**
```bash
./start.sh
# or with specific commands
./start.sh start      # Start all services
./start.sh stop       # Stop all services
./start.sh restart    # Restart all services
./start.sh status     # Check service status
```

**Start services individually:**
```bash
# MCP Server (port 8001)
python mcp_server.py

# API Backend (port 8000)
python api.py

# Streamlit Frontend (port 8501, HTTPS if SSL certs exist)
streamlit run streamlit_app.py
```

### Testing
```bash
# Run specific test files
python -m pytest test_file/test_skills.py
python -m pytest test_file/test_agent_skills.py
python -m pytest test_file/test_agent_events.py
```

### Key Configuration

All configuration is centralized in `config.py` using Pydantic settings:
- **Model Configuration**: Qwen model selection (qwen-max, qwen-vl-plus, etc.)
- **API Keys**: DashScope, NASA, AMAP, Tavily
- **RAG Settings**: Vector database path, retrieval settings
- **Memory Settings**: Short-term memory size, long-term storage path
- **Astronomy Settings**: Default location, ephemeris file, supported bodies

## Key Design Patterns

### Skill-Based Architecture
1. Skills are defined in `skill.yaml` with parameter schemas
2. `SkillManager` routes skill calls to appropriate implementations
3. `AstronomySkillRouter` maps skills to MCP tools with parameter transformation
4. Complex skills can orchestrate multiple MCP tool calls

### Streaming Architecture
1. `StreamingService` generates Server-Sent Events (SSE) for real-time updates
2. Event types: `thought`, `action`, `observation`, `final_answer`, `error`
3. Supports multi-modal inputs (images, audio) through dedicated endpoints

### Memory Management
1. **Short-term**: In-memory conversation history with configurable window size
2. **Long-term**: SQLite-based user profiles with preferences and habits
3. Memory is automatically loaded/saved based on user ID

### Error Handling
1. Centralized error handling in `core.errors`
2. Fallback service for graceful degradation when skills fail
3. Comprehensive logging with `logger.py`

## Important Files

- `config.py`: Centralized configuration management
- `skills.py`: Skill routing and orchestration layer
- `skill.yaml`: Skill definitions and parameter schemas
- `prompt_template.txt`: Agent prompt template (external file for easy editing)
- `start.sh`: Service orchestration script
- `api.py`: FastAPI endpoints with streaming support
- `mcp_server.py`: MCP protocol server exposing astronomy tools
- `streamlit_app.py`: Web interface

## Development Notes

### Adding New Skills
1. Add skill definition to `skill.yaml` with parameter schema
2. Implement skill logic in `skills.py` `AstronomySkillRouter` class
3. Register skill in router's `_registry` dictionary
4. Test with `test_file/test_skills.py`

### Adding New MCP Tools
1. Add tool implementation in `astronomy/` modules
2. Register tool in `mcp_server.py` using `@mcp.tool()` decorator
3. Update `skills.py` to map skills to new tools if needed

### Multi-modal Support
- **Images**: Use `/query_with_image` endpoint with image upload
- **Audio**: Use `/query_with_audio` endpoint with audio upload
- Vision and speech services handle preprocessing before agent processing

### RAG System
- Knowledge documents are stored in `vector_db/` directory
- Use `rag/build_bm25_index.py` for offline indexing
- Online retrieval combines vector search and BM25 for hybrid results

## Common Issues

1. **MCP Server fails to start**: Check if `de421.bsp` ephemeris file exists in project root
2. **API key errors**: Verify all required API keys are in `.env` file
3. **Port conflicts**: Services use ports 8000, 8001, 8501 - ensure they're available
4. **SSL certificate missing**: Streamlit runs in HTTP mode without certs (affects audio recording)
5. **Memory persistence**: Long-term memory uses SQLite at `./memory/long_term_memory/user_profiles.sqlite`