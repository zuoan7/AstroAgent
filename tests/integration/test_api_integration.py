import json
import os
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

pytest.importorskip("fastapi")


class TestAPIEndpointsIntegration:
    """测试API层与Agent层的集成"""

    @pytest.fixture
    def test_client(self):
        from fastapi.testclient import TestClient

        with patch("src.api.main.AstroAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.user_id = "test_user"
            mock_agent.long_term_memory = MagicMock()

            mock_profile = {
                "user_id": "test_user",
                "preferences": {"style": "详细"},
                "habits": {"topics": ["火星"]},
                "constraints": [],
                "background": {},
                "facts": [],
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-04-08T00:00:00",
            }
            mock_agent.long_term_memory.get_profile.return_value = mock_profile
            mock_agent.long_term_memory.delete_profile.return_value = True

            MockAgent.return_value = mock_agent

            from src.api.main import app

            client = TestClient(app)
            yield client, mock_agent

    def test_root_endpoint(self, test_client):
        client, _ = test_client
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_query_endpoint(self, test_client):
        client, mock_agent = test_client

        async def mock_generate_events(query, image_path=None):
            yield {"type": "text", "content": "火星是太阳系第四颗行星"}

        mock_agent.generate_events = mock_generate_events

        response = client.post(
            "/query",
            json={"query": "火星是什么", "user_id": "test_user"},
        )
        assert response.status_code == 200

    def test_add_knowledge_endpoint(self, test_client):
        client, mock_agent = test_client
        mock_agent.add_astronomy_knowledge.return_value = {
            "added_count": 1,
            "updated_count": 0,
            "unchanged_count": 0,
            "stored_count": 1,
            "bm25_doc_count": 1,
        }

        with patch("src.api.main.get_agent", return_value=mock_agent):
            response = client.post(
                "/add_knowledge",
                json={"knowledge": ["火星是太阳系第四颗行星"], "user_id": "test_user"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["result"]["stored_count"] == 1

    def test_get_profile_endpoint(self, test_client):
        client, mock_agent = test_client

        response = client.get("/profile?user_id=test_user")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["user_id"] == "test_user"

    def test_delete_profile_endpoint(self, test_client):
        client, mock_agent = test_client

        response = client.delete("/profile?user_id=test_user")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_clear_memory_endpoint(self, test_client):
        client, mock_agent = test_client

        response = client.post("/clear_memory")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_query_with_empty_query(self, test_client):
        client, mock_agent = test_client

        async def mock_generate_events(query, image_path=None):
            yield {"type": "text", "content": "请输入您的问题"}

        mock_agent.generate_events = mock_generate_events

        response = client.post(
            "/query",
            json={"query": "", "user_id": "test_user"},
        )
        assert response.status_code == 200

    def test_list_models_endpoint(self, test_client):
        client, _ = test_client

        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert isinstance(data["providers"], list)
        assert any(item["provider"] == "dashscope" for item in data["providers"])

    def test_switch_session_model_endpoint(self, test_client):
        client, _ = test_client

        response = client.post(
            "/session/model",
            json={
                "user_id": "test_user",
                "session_id": "session-a",
                "model_provider": "dashscope",
                "model_name": "qwen3.6-plus",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_provider"] == "dashscope"
        assert data["model_name"] == "qwen3.6-plus"


class TestSkillManagerIntegration:
    """测试SkillManager与Router的集成"""

    @pytest.fixture
    def mock_rag(self):
        rag = MagicMock()
        rag.get_relevant_context.return_value = "火星是太阳系第四颗行星"
        return rag

    @pytest.fixture
    def skill_manager(self, mock_rag):
        with patch("src.agent.skill_manager.AstronomySkillRouter") as MockRouter:
            mock_router = MagicMock()
            mock_router.list_skills.return_value = {
                "weather-lookup": "查询天气",
                "observation-planner": "生成观测计划",
                "celestial-events-forecast": "查询天象事件",
                "deep-sky-observing-guide": "深空观测指导",
                "neo-tracker": "近地天体追踪",
                "astrophotography-calculator": "天文摄影参数",
                "celestial-position-calculator": "天体位置计算",
            }
            mock_router.call.return_value = "测试结果"
            mock_router.call_mcp_tool.return_value = "MCP工具结果"
            MockRouter.return_value = mock_router

            from src.agent.skill_manager import SkillManager

            sm = SkillManager(rag_retriever=mock_rag)
            return sm, mock_router

    def test_list_skills(self, skill_manager):
        sm, _ = skill_manager
        skills = sm.list_skills()
        assert isinstance(skills, dict)
        assert "weather-lookup" in skills

    def test_call_skill(self, skill_manager):
        sm, mock_router = skill_manager
        result = sm.call_skill("weather-lookup", city="北京")
        assert result == "测试结果"

    def test_call_mcp_tool(self, skill_manager):
        sm, mock_router = skill_manager
        result = sm.call_mcp_tool("get_weather", city="北京")
        assert result == "MCP工具结果"

    def test_get_langchain_tools(self, skill_manager):
        sm, _ = skill_manager
        tools = sm.get_langchain_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_rag_tool_function(self, skill_manager):
        sm, _ = skill_manager
        mock_rag = sm._rag

        mock_rag.get_relevant_context.return_value = "火星是太阳系第四颗行星"
        result = mock_rag.get_relevant_context("火星")
        assert "火星" in result

    def test_weather_param_handler(self):
        from src.agent.skill_manager import SkillManager

        result = SkillManager._weather_param_handler(
            {"city": "北京", "location": "北京"}
        )
        assert result["city"] == "北京"
        assert "location" not in result

    def test_safe_convert_bool(self):
        from src.agent.skill_manager import SkillManager

        assert SkillManager._safe_convert("true", bool) is True
        assert SkillManager._safe_convert("false", bool) is False
        assert SkillManager._safe_convert("1", bool) is True

    def test_safe_convert_float(self):
        from src.agent.skill_manager import SkillManager

        assert SkillManager._safe_convert("3.14", float) == 3.14
        assert SkillManager._safe_convert("invalid", float) == "invalid"


class TestStreamingServiceIntegration:
    """测试流式服务与记忆模块的集成"""

    def test_format_chat_history(self, tmp_path):
        from src.agent.streaming_service import StreamingService
        from src.memory.api.dto import AppendMessageRequest
        from src.memory.api.memory_service import MemoryService

        memory = MemoryService(
            db_path=str(tmp_path / "memory.sqlite"),
            session_id="mem_test_user",
            user_id="test_user",
        )
        memory.append_message(
            AppendMessageRequest(
                session_id="mem_test_user",
                role="user",
                content="你好",
                timestamp=time.time(),
            )
        )
        memory.append_message(
            AppendMessageRequest(
                session_id="mem_test_user",
                role="assistant",
                content="你好！有什么天文问题吗？",
                timestamp=time.time(),
            )
        )

        service = StreamingService(
            agent_executor=None,
            memory=memory,
            user_id="test_user",
        )

        history = service._format_chat_history()
        assert "用户" in history
        assert "你好" in history
        assert "助手" in history

    def test_format_empty_chat_history(self, tmp_path):
        from src.agent.streaming_service import StreamingService
        from src.memory.api.memory_service import MemoryService

        memory = MemoryService(
            db_path=str(tmp_path / "memory.sqlite"),
            session_id="mem_test_user",
            user_id="test_user",
        )

        service = StreamingService(
            agent_executor=None,
            memory=memory,
            user_id="test_user",
        )

        history = service._format_chat_history()
        assert "无" in history and ("对话" in history or "历史" in history)

    def test_check_repeated_action(self):
        from src.agent.streaming_service import StreamingService

        service = StreamingService(
            agent_executor=None,
            memory=MagicMock(),
            user_id="test_user",
        )

        assert service._check_repeated_action("req1", "tool_a", "input1") is False
        assert service._check_repeated_action("req1", "tool_a", "input1") is True

    def test_build_response_from_intermediate_steps(self):
        from src.agent.streaming_service import StreamingService

        service = StreamingService(
            agent_executor=None,
            memory=MagicMock(),
            user_id="test_user",
        )

        mock_action = MagicMock()
        mock_action.tool = "WeatherLookup"
        mock_action.tool_input = '{"city": "北京"}'

        steps = [(mock_action, '{"weather": "晴", "temperature": "22"}')]

        result = service._build_response_from_intermediate_steps("北京天气", steps)
        assert isinstance(result, str)

    def test_extract_and_update_long_term_memory(self):
        import tempfile

        from src.agent.streaming_service import StreamingService
        from src.memory.api.memory_service import MemoryService
        from src.memory.long_term_memory import LongTermMemoryService as LongTermMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")
            ltm = LongTermMemory(db_path=db_path)
            memory = MemoryService(
                db_path=os.path.join(tmpdir, "memory.sqlite"),
                session_id="mem_test_user",
                user_id="test_user",
            )

            service = StreamingService(
                agent_executor=None,
                memory=memory,
                long_term_memory=ltm,
                user_id="test_user",
            )

            service._extract_and_update_long_term_memory(
                "请详细介绍一下火星", "火星是太阳系第四颗行星..."
            )
            service._extract_and_update_long_term_memory(
                "请详细介绍一下火星", "火星是太阳系第四颗行星..."
            )
            if hasattr(ltm, "_extract_executor"):
                ltm._extract_executor.shutdown(wait=True)

            profile = ltm.get_profile("test_user")
            candidates = ltm.list_candidates("test_user", limit=20)
            assert profile is not None or len(candidates) > 0
