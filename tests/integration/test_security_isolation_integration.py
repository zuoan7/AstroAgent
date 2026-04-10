import os
import sys
import time
import tempfile
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.main import (
    app,
    sanitize_filename,
    validate_upload_path,
    validate_file_type,
    SessionManager,
    SessionData,
    UPLOAD_DIR,
)
from src.memory.memory import ShortTermMemory, LongTermMemory
from src.core.errors import AgentError, ErrorCode
from src.core.config import settings

from fastapi.testclient import TestClient


async def _fake_generate_events(query, **kwargs):
    yield {"type": "text", "content": "test response"}


def _create_mock_agent():
    mock_agent = MagicMock()
    mock_agent._agent_executor = MagicMock()
    mock_agent.long_term_memory = MagicMock()
    mock_agent.fallback_service = MagicMock()
    mock_agent.user_id = "test_user"
    mock_agent.speech_service = MagicMock()
    mock_agent.speech_service.build_speech_query = MagicMock(return_value="transcribed text")
    return mock_agent


def _create_mock_session():
    mock_session = MagicMock()
    mock_session.streaming_service = MagicMock()
    mock_session.streaming_service.generate_events = _fake_generate_events
    return mock_session


class TestAPIFileUploadSecurity:
    def setup_method(self):
        self.mock_agent = _create_mock_agent()

    @patch("src.api.main.agent")
    @patch("src.api.main.session_manager")
    def test_image_upload_valid_type(self, mock_sm, mock_agent_ref):
        mock_agent_ref.speech_service = self.mock_agent.speech_service
        mock_session = _create_mock_session()
        mock_sm.get_session.return_value = mock_session

        client = TestClient(app, raise_server_exceptions=False)
        img_data = BytesIO(b"fake image data" * 100)
        response = client.post(
            "/query_with_image",
            data={"query": "test", "user_id": "user1"},
            files={"image": ("photo.jpg", img_data, "image/jpeg")},
        )
        assert response.status_code != 415

    @patch("src.api.main.agent")
    @patch("src.api.main.session_manager")
    def test_image_upload_invalid_type(self, mock_sm, mock_agent_ref):
        mock_agent_ref.speech_service = self.mock_agent.speech_service
        mock_session = MagicMock()
        mock_sm.get_session.return_value = mock_session

        client = TestClient(app, raise_server_exceptions=False)
        exe_data = BytesIO(b"malicious content")
        response = client.post(
            "/query_with_image",
            data={"query": "test", "user_id": "user1"},
            files={"image": ("malware.exe", exe_data, "application/octet-stream")},
        )
        assert response.status_code == 415

    @patch("src.api.main.agent")
    @patch("src.api.main.session_manager")
    def test_image_upload_script_type(self, mock_sm, mock_agent_ref):
        mock_agent_ref.speech_service = self.mock_agent.speech_service
        mock_session = MagicMock()
        mock_sm.get_session.return_value = mock_session

        client = TestClient(app, raise_server_exceptions=False)
        script_data = BytesIO(b'<?php echo "hack"; ?>')
        response = client.post(
            "/query_with_image",
            data={"query": "test", "user_id": "user1"},
            files={"image": ("shell.php", script_data, "application/x-php")},
        )
        assert response.status_code == 415

    @patch("src.api.main.agent")
    @patch("src.api.main.session_manager")
    def test_audio_upload_valid_type(self, mock_sm, mock_agent_ref):
        mock_agent_ref.speech_service = self.mock_agent.speech_service
        mock_session = _create_mock_session()
        mock_sm.get_session.return_value = mock_session

        client = TestClient(app, raise_server_exceptions=False)
        audio_data = BytesIO(b"fake audio data" * 100)
        response = client.post(
            "/query_with_audio",
            data={"query": "test", "user_id": "user1"},
            files={"audio": ("recording.wav", audio_data, "audio/wav")},
        )
        assert response.status_code != 415

    @patch("src.api.main.agent")
    @patch("src.api.main.session_manager")
    def test_audio_upload_invalid_type(self, mock_sm, mock_agent_ref):
        mock_agent_ref.speech_service = self.mock_agent.speech_service
        mock_session = MagicMock()
        mock_sm.get_session.return_value = mock_session

        client = TestClient(app, raise_server_exceptions=False)
        bat_data = BytesIO(b"@echo off")
        response = client.post(
            "/query_with_audio",
            data={"query": "test", "user_id": "user1"},
            files={"audio": ("script.bat", bat_data, "application/bat")},
        )
        assert response.status_code == 415


class TestAPIFileSizeLimit:
    @patch("src.api.main.agent")
    @patch("src.api.main.session_manager")
    def test_upload_within_limit(self, mock_sm, mock_agent_ref):
        mock_agent_ref.speech_service = MagicMock()
        mock_session = _create_mock_session()
        mock_sm.get_session.return_value = mock_session

        client = TestClient(app, raise_server_exceptions=False)
        small_data = BytesIO(b"x" * 1024)
        response = client.post(
            "/query_with_image",
            data={"query": "test", "user_id": "user1"},
            files={"image": ("photo.jpg", small_data, "image/jpeg")},
        )
        assert response.status_code != 413

    @patch("src.api.main.agent")
    @patch("src.api.main.session_manager")
    def test_upload_exceeds_limit(self, mock_sm, mock_agent_ref):
        mock_agent_ref.speech_service = MagicMock()
        mock_session = MagicMock()
        mock_sm.get_session.return_value = mock_session

        client = TestClient(app, raise_server_exceptions=False)
        large_data = BytesIO(b"x" * (11 * 1024 * 1024))
        response = client.post(
            "/query_with_image",
            data={"query": "test", "user_id": "user1"},
            files={"image": ("large.jpg", large_data, "image/jpeg")},
        )
        assert response.status_code == 413


class TestAPIPathTraversal:
    def test_path_traversal_in_filename(self):
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "passwd" in result

    def test_path_traversal_null_byte(self):
        result = sanitize_filename("image.jpg\x00.exe")
        assert "\x00" not in result

    def test_validate_upload_path_rejects_traversal(self):
        malicious_path = os.path.abspath(os.path.join(UPLOAD_DIR, "..", "..", "etc", "passwd"))
        assert validate_upload_path(malicious_path) is False

    def test_validate_upload_path_accepts_valid(self):
        valid_path = os.path.join(UPLOAD_DIR, "abc123.jpg")
        assert validate_upload_path(valid_path) is True


class TestAPIMultiUserIsolation:
    def test_session_manager_creates_isolated_sessions(self):
        mock_agent = _create_mock_agent()
        sm = SessionManager(mock_agent, max_age=3600, cleanup_interval=300)

        session1 = sm.get_session("user_alpha")
        session2 = sm.get_session("user_beta")

        session1.memory.add_message("user", "alpha secret", time.time())
        session2.memory.add_message("user", "beta secret", time.time())

        alpha_msgs = session1.memory.get_recent_messages()
        beta_msgs = session2.memory.get_recent_messages()

        assert len(alpha_msgs) == 1
        assert alpha_msgs[0]["content"] == "alpha secret"
        assert len(beta_msgs) == 1
        assert beta_msgs[0]["content"] == "beta secret"

    def test_clear_memory_isolated_per_user(self):
        mock_agent = _create_mock_agent()
        sm = SessionManager(mock_agent, max_age=3600, cleanup_interval=300)

        session1 = sm.get_session("user1")
        session2 = sm.get_session("user2")

        session1.memory.add_message("user", "data1", time.time())
        session2.memory.add_message("user", "data2", time.time())

        sm.clear_session("user1")

        assert sm.get_session("user2").memory.get_size() == 1
        assert sm.get_session("user2").memory.get_recent_messages()[0]["content"] == "data2"

    def test_long_term_memory_user_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_isolation.sqlite")
            ltm = LongTermMemory(db_path=db_path)

            ltm.merge_and_update("user_x", {
                "preferences": {"lang": "zh"},
                "habits": {},
                "constraints": []
            })
            ltm.merge_and_update("user_y", {
                "preferences": {"lang": "en"},
                "habits": {},
                "constraints": []
            })

            profile_x = ltm.load_profile("user_x")
            profile_y = ltm.load_profile("user_y")

            assert profile_x.preferences["lang"] == "zh"
            assert profile_y.preferences["lang"] == "en"

    @patch("src.api.main.agent")
    @patch("src.api.main.session_manager")
    def test_query_endpoint_uses_user_session(self, mock_sm, mock_agent_ref):
        mock_session = MagicMock()
        mock_session.streaming_service = MagicMock()

        async def fake_events(query, **kwargs):
            yield {"type": "text", "content": "response"}

        mock_session.streaming_service.generate_events = fake_events
        mock_sm.get_session.return_value = mock_session

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/query",
            json={"query": "hello", "user_id": "specific_user"},
        )
        mock_sm.get_session.assert_called_once_with("specific_user")

    @patch("src.api.main.agent")
    @patch("src.api.main.session_manager")
    def test_clear_memory_endpoint_per_user(self, mock_sm, mock_agent_ref):
        mock_sm.clear_session.return_value = True

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/clear_memory?user_id=user_to_clear",
        )
        mock_sm.clear_session.assert_called_once_with("user_to_clear")
        assert response.json()["user_id"] == "user_to_clear"


class TestAPIProfileIsolation:
    @patch("src.api.main.agent")
    def test_get_profile_uses_user_id(self, mock_agent_ref):
        mock_profile = MagicMock()
        mock_profile.user_id = "custom_user"
        mock_profile.preferences = {"style": "详细"}
        mock_profile.habits = {}
        mock_profile.constraints = []
        mock_profile.created_at = "2026-01-01"
        mock_profile.updated_at = "2026-01-01"
        mock_agent_ref.long_term_memory.load_profile.return_value = mock_profile

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/profile?user_id=custom_user")
        data = response.json()
        assert data["user_id"] == "custom_user"
        mock_agent_ref.long_term_memory.load_profile.assert_called_once_with("custom_user")

    @patch("src.api.main.agent")
    def test_delete_profile_uses_user_id(self, mock_agent_ref):
        mock_agent_ref.long_term_memory.delete_profile.return_value = True

        client = TestClient(app, raise_server_exceptions=False)
        response = client.delete("/profile?user_id=user_to_delete")
        data = response.json()
        assert data["user_id"] == "user_to_delete"
        mock_agent_ref.long_term_memory.delete_profile.assert_called_once_with("user_to_delete")


class TestSecurityValidationIntegration:
    def test_file_type_whitelist_images(self):
        allowed = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        for ext in allowed:
            assert validate_file_type(f"test{ext}", allowed) is True

    def test_file_type_whitelist_rejects_dangerous(self):
        allowed = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        dangerous = ['.exe', '.bat', '.cmd', '.sh', '.php', '.py', '.js', '.html', '.css', '.svg']
        for ext in dangerous:
            assert validate_file_type(f"test{ext}", allowed) is False

    def test_file_type_whitelist_audio(self):
        allowed = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
        for ext in allowed:
            assert validate_file_type(f"test{ext}", allowed) is True

    def test_file_type_whitelist_rejects_audio_dangerous(self):
        allowed = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
        dangerous = ['.exe', '.bat', '.sh', '.py', '.jar', '.dll']
        for ext in dangerous:
            assert validate_file_type(f"test{ext}", allowed) is False

    def test_sanitize_removes_path_separators(self):
        for sep in ['/', '\\']:
            result = sanitize_filename(f"dir{sep}file.jpg")
            assert sep not in result

    def test_validate_path_rejects_system_paths(self):
        system_paths = [
            "/etc/passwd",
            "/var/log/syslog",
            "/root/.ssh/id_rsa",
            "C:\\Windows\\System32\\config\\SAM",
        ]
        for path in system_paths:
            assert validate_upload_path(path) is False
