import os
import sys
import time
import tempfile
import threading
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.main import (
    sanitize_filename,
    validate_upload_path,
    validate_file_type,
    SessionData,
    SessionManager,
    UPLOAD_DIR,
)
from src.memory.memory import ShortTermMemory, LongTermMemory
from src.core.errors import AgentError, ErrorCode


class TestSanitizeFilename:
    def test_normal_filename(self):
        assert sanitize_filename("photo.jpg") == "photo.jpg"

    def test_filename_with_path(self):
        result = sanitize_filename("/etc/passwd")
        assert result == "passwd"

    def test_filename_with_windows_path(self):
        result = sanitize_filename("C:\\Windows\\System32\\test.png")
        assert "\\" not in result
        assert result == "test.png"

    def test_filename_with_traversal(self):
        result = sanitize_filename("../../etc/shadow")
        assert ".." not in result
        assert "shadow" in result

    def test_filename_with_special_chars(self):
        result = sanitize_filename("file<>:|?*.jpg")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "|" not in result
        assert "?" not in result

    def test_filename_with_double_dots(self):
        result = sanitize_filename("file...jpg")
        assert "..." not in result

    def test_empty_filename(self):
        assert sanitize_filename("") == ""

    def test_none_filename(self):
        assert sanitize_filename(None) == ""

    def test_filename_with_leading_dots(self):
        result = sanitize_filename(".hidden")
        assert not result.startswith(".")

    def test_filename_with_spaces(self):
        result = sanitize_filename("  my file.jpg  ")
        assert result == "my file.jpg"

    def test_filename_with_null_bytes(self):
        result = sanitize_filename("file\x00.jpg")
        assert "\x00" not in result


class TestValidateUploadPath:
    def test_valid_path(self):
        save_path = os.path.join(UPLOAD_DIR, "abc123.jpg")
        assert validate_upload_path(save_path) is True

    def test_valid_path_absolute(self):
        save_path = os.path.abspath(os.path.join(UPLOAD_DIR, "abc123.jpg"))
        assert validate_upload_path(save_path) is True

    def test_traversal_attack(self):
        save_path = os.path.abspath(os.path.join(UPLOAD_DIR, "..", "etc", "passwd"))
        assert validate_upload_path(save_path) is False

    def test_outside_directory(self):
        save_path = "/tmp/malicious.jpg"
        assert validate_upload_path(save_path) is False

    def test_exact_upload_dir(self):
        assert validate_upload_path(UPLOAD_DIR) is True


class TestValidateFileType:
    def test_allowed_image(self):
        assert validate_file_type("photo.jpg", {'.jpg', '.png'}) is True

    def test_disallowed_image(self):
        assert validate_file_type("script.exe", {'.jpg', '.png'}) is False

    def test_case_insensitive(self):
        assert validate_file_type("photo.JPG", {'.jpg', '.png'}) is True

    def test_empty_filename(self):
        assert validate_file_type("", {'.jpg', '.png'}) is False

    def test_none_filename(self):
        assert validate_file_type(None, {'.jpg', '.png'}) is False

    def test_no_extension(self):
        assert validate_file_type("photo", {'.jpg', '.png'}) is False

    def test_double_extension(self):
        assert validate_file_type("photo.png.exe", {'.jpg', '.png'}) is False

    def test_allowed_audio(self):
        assert validate_file_type("audio.mp3", {'.mp3', '.wav'}) is True


class TestShortTermMemory:
    def test_add_and_retrieve(self):
        mem = ShortTermMemory()
        mem.add_message("user", "hello", time.time())
        mem.add_message("assistant", "hi", time.time())
        msgs = mem.get_recent_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    def test_max_size_truncation(self):
        mem = ShortTermMemory()
        mem.max_size = 5
        for i in range(10):
            mem.add_message("user", f"msg_{i}", time.time())
        assert mem.get_size() == 5
        msgs = mem.get_recent_messages()
        assert msgs[0]["content"] == "msg_5"

    def test_clear(self):
        mem = ShortTermMemory()
        mem.add_message("user", "hello", time.time())
        mem.clear()
        assert mem.get_size() == 0

    def test_window_size(self):
        mem = ShortTermMemory()
        for i in range(10):
            mem.add_message("user", f"msg_{i}", time.time())
        msgs = mem.get_recent_messages(window=3)
        assert len(msgs) == 3

    def test_isolation_between_instances(self):
        mem1 = ShortTermMemory()
        mem2 = ShortTermMemory()
        mem1.add_message("user", "user1_msg", time.time())
        mem2.add_message("user", "user2_msg", time.time())
        assert mem1.get_recent_messages()[0]["content"] == "user1_msg"
        assert mem2.get_recent_messages()[0]["content"] == "user2_msg"
        assert mem1.get_size() == 1
        assert mem2.get_size() == 1


class TestSessionData:
    def test_session_creation(self):
        mock_agent = MagicMock()
        mock_agent._agent_executor = MagicMock()
        mock_agent.long_term_memory = MagicMock()
        mock_agent.fallback_service = MagicMock()

        session = SessionData("test_user", mock_agent)
        assert session.user_id == "test_user"
        assert isinstance(session.memory, ShortTermMemory)
        assert session.streaming_service is not None
        assert session.last_access_time > 0

    def test_session_has_own_memory(self):
        mock_agent = MagicMock()
        mock_agent._agent_executor = MagicMock()
        mock_agent.long_term_memory = MagicMock()
        mock_agent.fallback_service = MagicMock()

        session1 = SessionData("user1", mock_agent)
        session2 = SessionData("user2", mock_agent)

        session1.memory.add_message("user", "secret_data", time.time())
        assert session1.memory.get_size() == 1
        assert session2.memory.get_size() == 0


class TestSessionManager:
    def _create_session_manager(self):
        mock_agent = MagicMock()
        mock_agent._agent_executor = MagicMock()
        mock_agent.long_term_memory = MagicMock()
        mock_agent.fallback_service = MagicMock()
        return SessionManager(mock_agent, max_age=3600, cleanup_interval=300)

    def test_get_session_creates_new(self):
        sm = self._create_session_manager()
        session = sm.get_session("user1")
        assert session.user_id == "user1"
        assert sm.get_active_session_count() == 1

    def test_get_session_returns_existing(self):
        sm = self._create_session_manager()
        session1 = sm.get_session("user1")
        session1.memory.add_message("user", "hello", time.time())
        session2 = sm.get_session("user1")
        assert session2.memory.get_size() == 1
        assert session1 is session2

    def test_different_users_isolated(self):
        sm = self._create_session_manager()
        session1 = sm.get_session("user1")
        session2 = sm.get_session("user2")
        session1.memory.add_message("user", "user1_data", time.time())
        session2.memory.add_message("user", "user2_data", time.time())
        assert session1.memory.get_recent_messages()[0]["content"] == "user1_data"
        assert session2.memory.get_recent_messages()[0]["content"] == "user2_data"
        assert sm.get_active_session_count() == 2

    def test_clear_session(self):
        sm = self._create_session_manager()
        sm.get_session("user1")
        assert sm.get_active_session_count() == 1
        result = sm.clear_session("user1")
        assert result is True
        assert sm.get_active_session_count() == 0

    def test_clear_nonexistent_session(self):
        sm = self._create_session_manager()
        result = sm.clear_session("nonexistent")
        assert result is False

    def test_cleanup_stale_sessions(self):
        sm = self._create_session_manager()
        sm._max_age = 1
        sm._cleanup_interval = 0
        sm.get_session("user1")
        assert sm.get_active_session_count() == 1
        time.sleep(1.5)
        sm.get_session("user2")
        assert sm.get_active_session_count() == 1

    def test_thread_safety(self):
        sm = self._create_session_manager()
        errors = []

        def create_session(user_id):
            try:
                session = sm.get_session(user_id)
                session.memory.add_message("user", f"msg_from_{user_id}", time.time())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_session, args=(f"user_{i}",)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert sm.get_active_session_count() == 50


class TestLongTermMemoryIsolation:
    def test_different_users_profiles_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_profiles.sqlite")
            ltm = LongTermMemory(db_path=db_path)

            ltm.merge_and_update("user1", {"preferences": {"style": "详细"}, "habits": {}, "constraints": []})
            ltm.merge_and_update("user2", {"preferences": {"style": "简短"}, "habits": {}, "constraints": []})

            profile1 = ltm.load_profile("user1")
            profile2 = ltm.load_profile("user2")

            assert profile1.preferences["style"] == "详细"
            assert profile2.preferences["style"] == "简短"

    def test_delete_user_does_not_affect_other(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_profiles.sqlite")
            ltm = LongTermMemory(db_path=db_path)

            ltm.merge_and_update("user1", {"preferences": {"style": "详细"}, "habits": {}, "constraints": []})
            ltm.merge_and_update("user2", {"preferences": {"style": "简短"}, "habits": {}, "constraints": []})

            ltm.delete_profile("user1")
            assert ltm.load_profile("user1") is None
            assert ltm.load_profile("user2") is not None
            assert ltm.load_profile("user2").preferences["style"] == "简短"


class TestSecurityErrorCodes:
    def test_security_error_code(self):
        error = AgentError(code=ErrorCode.SECURITY_ERROR, message="security violation")
        assert error.code == ErrorCode.SECURITY_ERROR
        d = error.to_dict()
        assert d["code"] == "SECURITY_ERROR"

    def test_file_too_large_error_code(self):
        error = AgentError(code=ErrorCode.FILE_TOO_LARGE, message="file too large")
        assert error.code == ErrorCode.FILE_TOO_LARGE

    def test_file_type_not_allowed_error_code(self):
        error = AgentError(code=ErrorCode.FILE_TYPE_NOT_ALLOWED, message="type not allowed")
        assert error.code == ErrorCode.FILE_TYPE_NOT_ALLOWED

    def test_path_traversal_error_code(self):
        error = AgentError(code=ErrorCode.PATH_TRAVERSAL_ERROR, message="path traversal")
        assert error.code == ErrorCode.PATH_TRAVERSAL_ERROR

    def test_rate_limit_error_code(self):
        error = AgentError(code=ErrorCode.RATE_LIMIT_ERROR, message="rate limited")
        assert error.code == ErrorCode.RATE_LIMIT_ERROR
