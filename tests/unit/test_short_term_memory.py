import os
import tempfile
import time
from unittest.mock import MagicMock, patch


def test_short_term_memory_persistence_restore_and_tool_context():
    from src.memory.memory import ShortTermMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "sessions.sqlite")
        with patch("src.memory.memory.settings") as mock_s:
            mock_s.MEMORY_SIZE = 5
            mock_s.MEMORY_WINDOW = 3
            mock_s.STM_CONTEXT_MAX_TOKENS = 1200
            mock_s.STM_CONTEXT_BUDGET = 1200
            mock_s.STM_SUMMARY_MAX_TOKENS = 200
            mock_s.STM_SUMMARY_TRIGGER_MESSAGES = 4
            mock_s.STM_SUMMARY_TRIGGER_TOKENS = 2000
            mock_s.STM_SUMMARY_KEEP_LAST_N = 2
            mock_s.STM_MAX_RECENT_MESSAGES = 3
            mock_s.STM_MAX_RECENT_TOKENS = 600
            mock_s.STM_MAX_TOOL_RECORDS = 5
            mock_s.STM_MAX_SALIENT_FACTS = 8
            mock_s.STM_ENABLE_SUMMARY = True
            mock_s.STM_PERSISTENCE_ENABLED = True
            mock_s.STM_PERSISTENCE_PATH = db_path
            mock_s.STM_IMPORTANCE_HIGH_ROLES = {"user", "system"}
            mock_s.STM_TOOL_RESULT_MAX_LENGTH = 200
            mock_s.DEFAULT_USER_ID = "test_user"
            mock_s.DASHSCOPE_API_KEY = None

            memory = ShortTermMemory(session_id="stm_test", user_id="test_user")
            memory.add_message("user", "请记住我想看木星", time.time())
            memory.add_message("assistant", "好的，我会保留这个目标", time.time())
            memory.add_tool_call("WeatherLookup", '{"city":"北京"}', '{"weather":"晴"}', time.time(), True)

            context = memory.build_context()
            assert "tool summary" in context["context_text"]
            assert "WeatherLookup" in context["context_text"]

            restored = ShortTermMemory.restore_session("stm_test")
            assert restored is not None
            assert restored.user_id == "test_user"
            assert any(item["tool_name"] == "WeatherLookup" for item in restored.get_tool_calls())


def test_streaming_service_tool_end_writes_short_term_memory():
    from src.agent.streaming_service import StreamingService

    memory = MagicMock()
    service = StreamingService(agent_executor=None, memory=memory, user_id="test_user")
    service._tool_runs["run_1"] = {
        "name": "WeatherLookup",
        "input": '{"city":"北京"}',
        "start_time": time.time() - 0.1,
        "request_id": "req_1",
    }

    service._handle_tool_end("req_1", "run_1", {"output": '{"weather":"晴"}'})

    memory.add_tool_call.assert_called_once()
    kwargs = memory.add_tool_call.call_args.kwargs
    assert kwargs["tool_name"] == "WeatherLookup"
    assert kwargs["success"] is True


def test_short_term_memory_long_tool_result_is_summarized():
    from src.memory.memory import ShortTermMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "sessions.sqlite")
        with patch("src.memory.memory.settings") as mock_s:
            mock_s.MEMORY_SIZE = 5
            mock_s.MEMORY_WINDOW = 3
            mock_s.STM_CONTEXT_MAX_TOKENS = 1200
            mock_s.STM_CONTEXT_BUDGET = 1200
            mock_s.STM_SUMMARY_MAX_TOKENS = 200
            mock_s.STM_SUMMARY_TRIGGER_MESSAGES = 4
            mock_s.STM_SUMMARY_TRIGGER_TOKENS = 2000
            mock_s.STM_SUMMARY_KEEP_LAST_N = 2
            mock_s.STM_MAX_RECENT_MESSAGES = 3
            mock_s.STM_MAX_RECENT_TOKENS = 600
            mock_s.STM_MAX_TOOL_RECORDS = 5
            mock_s.STM_MAX_SALIENT_FACTS = 8
            mock_s.STM_ENABLE_SUMMARY = True
            mock_s.STM_PERSISTENCE_ENABLED = True
            mock_s.STM_PERSISTENCE_PATH = db_path
            mock_s.STM_IMPORTANCE_HIGH_ROLES = {"user", "system"}
            mock_s.STM_TOOL_RESULT_MAX_LENGTH = 80
            mock_s.DEFAULT_USER_ID = "test_user"
            mock_s.DASHSCOPE_API_KEY = None

            memory = ShortTermMemory(session_id="stm_test_long_tool", user_id="test_user")
            long_result = "x" * 500
            memory.add_tool_call("WeatherLookup", '{"city":"北京"}', long_result, time.time(), True)

            tool_calls = memory.get_tool_calls()
            assert len(tool_calls) == 1
            assert len(tool_calls[0]["output_summary"]) <= 83
            assert tool_calls[0]["output_summary"].endswith("...")
