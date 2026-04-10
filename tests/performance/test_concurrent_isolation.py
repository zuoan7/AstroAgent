import os
import sys
import time
import tempfile
import threading
import asyncio
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.main import SessionManager, SessionData
from src.memory.memory import ShortTermMemory, LongTermMemory
from src.core.errors import AgentError, ErrorCode


def _create_mock_agent():
    mock_agent = MagicMock()
    mock_agent._agent_executor = MagicMock()
    mock_agent.long_term_memory = MagicMock()
    mock_agent.fallback_service = MagicMock()
    return mock_agent


class TestConcurrentSessionAccess:
    def test_concurrent_session_creation(self):
        mock_agent = _create_mock_agent()
        sm = SessionManager(mock_agent, max_age=3600, cleanup_interval=300)
        errors = []
        sessions = {}

        def create_and_use(user_id):
            try:
                session = sm.get_session(user_id)
                session.memory.max_size = 200
                session.memory.add_message("user", f"msg_from_{user_id}", time.time())
                sessions[user_id] = session
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_and_use, args=(f"user_{i}",)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert sm.get_active_session_count() == 100

        for uid, session in sessions.items():
            msgs = session.memory.get_recent_messages()
            assert len(msgs) == 1
            assert msgs[0]["content"] == f"msg_from_{uid}"

    def test_concurrent_read_write_isolation(self):
        mock_agent = _create_mock_agent()
        sm = SessionManager(mock_agent, max_age=3600, cleanup_interval=300)

        user1_session = sm.get_session("user1")
        user2_session = sm.get_session("user2")

        user1_session.memory.max_size = 200
        user2_session.memory.max_size = 200

        errors = []
        msg_count = 50

        def write_messages(session, prefix, count):
            try:
                for i in range(count):
                    session.memory.add_message("user", f"{prefix}_msg_{i}", time.time())
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=write_messages, args=(user1_session, "user1", msg_count))
        t2 = threading.Thread(target=write_messages, args=(user2_session, "user2", msg_count))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
        assert user1_session.memory.get_size() == msg_count
        assert user2_session.memory.get_size() == msg_count

        user1_msgs = user1_session.memory.get_recent_messages()
        for msg in user1_msgs:
            assert msg["content"].startswith("user1_msg_")

        user2_msgs = user2_session.memory.get_recent_messages()
        for msg in user2_msgs:
            assert msg["content"].startswith("user2_msg_")

    def test_concurrent_clear_and_access(self):
        mock_agent = _create_mock_agent()
        sm = SessionManager(mock_agent, max_age=3600, cleanup_interval=300)

        errors = []

        def access_session(user_id):
            try:
                session = sm.get_session(user_id)
                session.memory.add_message("user", "data", time.time())
            except Exception as e:
                errors.append(e)

        def clear_session(user_id):
            try:
                sm.clear_session(user_id)
            except Exception as e:
                errors.append(e)

        for _ in range(10):
            sm.get_session("target_user")

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=access_session, args=("target_user",)))
            threads.append(threading.Thread(target=clear_session, args=("target_user",)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_session_cleanup(self):
        mock_agent = _create_mock_agent()
        sm = SessionManager(mock_agent, max_age=1, cleanup_interval=0)

        sm.get_session("user1")
        assert sm.get_active_session_count() == 1

        time.sleep(2)

        sm.get_session("user2")
        active = sm.get_active_session_count()
        assert active <= 2
        if active == 1:
            session2 = sm.get_session("user2")
            assert session2.memory.get_size() == 0


class TestConcurrentLongTermMemory:
    def test_concurrent_profile_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_concurrent.sqlite")
            ltm = LongTermMemory(db_path=db_path)
            errors = []

            def write_profile(user_id, style):
                try:
                    ltm.merge_and_update(user_id, {
                        "preferences": {"style": style},
                        "habits": {},
                        "constraints": []
                    })
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=write_profile, args=(f"user_{i}", f"style_{i}"))
                for i in range(20)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0

            for i in range(20):
                profile = ltm.load_profile(f"user_{i}")
                assert profile is not None
                assert profile.preferences["style"] == f"style_{i}"

    def test_concurrent_read_write_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_rw.sqlite")
            ltm = LongTermMemory(db_path=db_path)

            ltm.merge_and_update("shared_user", {
                "preferences": {"counter": 0},
                "habits": {},
                "constraints": []
            })

            errors = []

            def read_write():
                try:
                    profile = ltm.load_profile("shared_user")
                    current = profile.preferences.get("counter", 0)
                    time.sleep(0.001)
                    ltm.merge_and_update("shared_user", {
                        "preferences": {"counter": current + 1},
                        "habits": {},
                        "constraints": []
                    })
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=read_write) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            profile = ltm.load_profile("shared_user")
            assert profile is not None


class TestConcurrentUserIsolation:
    def test_no_cross_user_data_leakage(self):
        mock_agent = _create_mock_agent()
        sm = SessionManager(mock_agent, max_age=3600, cleanup_interval=300)

        num_users = 20
        messages_per_user = 10
        errors = []

        def user_operations(user_id):
            try:
                session = sm.get_session(user_id)
                session.memory.max_size = 200
                for i in range(messages_per_user):
                    session.memory.add_message("user", f"{user_id}_msg_{i}", time.time())
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=user_operations, args=(f"user_{i}",))
            for i in range(num_users)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        for i in range(num_users):
            user_id = f"user_{i}"
            session = sm.get_session(user_id)
            assert session.memory.get_size() == messages_per_user, f"user {user_id}: expected {messages_per_user}, got {session.memory.get_size()}"
            msgs = session.memory.get_recent_messages(window=messages_per_user)
            for msg in msgs:
                assert msg["content"].startswith(f"{user_id}_msg_")

    def test_clear_one_user_does_not_affect_others(self):
        mock_agent = _create_mock_agent()
        sm = SessionManager(mock_agent, max_age=3600, cleanup_interval=300)

        for i in range(5):
            session = sm.get_session(f"user_{i}")
            session.memory.add_message("user", f"persistent_data_{i}", time.time())

        sm.clear_session("user_2")

        for i in range(5):
            if i == 2:
                new_session = sm.get_session(f"user_{i}")
                assert new_session.memory.get_size() == 0
            else:
                session = sm.get_session(f"user_{i}")
                msgs = session.memory.get_recent_messages()
                assert len(msgs) == 1
                assert msgs[0]["content"] == f"persistent_data_{i}"

    def test_long_term_memory_isolation_under_concurrency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_ltm_iso.sqlite")
            ltm = LongTermMemory(db_path=db_path)
            errors = []

            def create_profile(user_id):
                try:
                    ltm.merge_and_update(user_id, {
                        "preferences": {"secret": f"secret_of_{user_id}"},
                        "habits": {"topics": [user_id]},
                        "constraints": [f"only_{user_id}"]
                    })
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=create_profile, args=(f"ltm_user_{i}",))
                for i in range(20)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0

            for i in range(20):
                uid = f"ltm_user_{i}"
                profile = ltm.load_profile(uid)
                assert profile is not None
                assert profile.preferences["secret"] == f"secret_of_{uid}"
                assert uid in profile.habits["topics"]
                assert f"only_{uid}" in profile.constraints
