import os
import json
import time
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

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

            mock_profile = MagicMock()
            mock_profile.user_id = "test_user"
            mock_profile.preferences = {"style": "详细"}
            mock_profile.habits = {"topics": ["火星"]}
            mock_profile.constraints = []
            mock_profile.created_at = "2026-01-01T00:00:00"
            mock_profile.updated_at = "2026-04-08T00:00:00"
            mock_agent.long_term_memory.load_profile.return_value = mock_profile
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

        response = client.post(
            "/add_knowledge",
            json={"knowledge": ["火星是太阳系第四颗行星"], "user_id": "test_user"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

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

    def test_format_chat_history(self):
        from src.memory.memory import ShortTermMemory
        from src.agent.streaming_service import StreamingService

        with patch("src.memory.memory.settings") as mock_s:
            mock_s.MEMORY_SIZE = 15
            mock_s.MEMORY_WINDOW = 8
            memory = ShortTermMemory()

        memory.add_message("user", "你好", time.time())
        memory.add_message("assistant", "你好！有什么天文问题吗？", time.time())

        service = StreamingService(
            agent_executor=None,
            memory=memory,
            user_id="test_user",
        )

        history = service._format_chat_history()
        assert "用户" in history
        assert "你好" in history
        assert "助手" in history

    def test_format_empty_chat_history(self):
        from src.memory.memory import ShortTermMemory
        from src.agent.streaming_service import StreamingService

        with patch("src.memory.memory.settings") as mock_s:
            mock_s.MEMORY_SIZE = 15
            mock_s.MEMORY_WINDOW = 8
            memory = ShortTermMemory()

        service = StreamingService(
            agent_executor=None,
            memory=memory,
            user_id="test_user",
        )

        history = service._format_chat_history()
        assert "无历史对话" in history

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
        from src.memory.memory import ShortTermMemory, LongTermMemory
        from src.agent.streaming_service import StreamingService
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")
            ltm = LongTermMemory(db_path=db_path)

            with patch("src.memory.memory.settings") as mock_s:
                mock_s.MEMORY_SIZE = 15
                mock_s.MEMORY_WINDOW = 8
                stm = ShortTermMemory()

            service = StreamingService(
                agent_executor=None,
                memory=stm,
                long_term_memory=ltm,
                user_id="test_user",
            )

            service._extract_and_update_long_term_memory(
                "请详细介绍一下火星",
                "火星是太阳系第四颗行星..."
            )

            profile = ltm.load_profile("test_user")
            assert profile is not None
