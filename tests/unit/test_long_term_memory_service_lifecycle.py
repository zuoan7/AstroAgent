"""Phase 5.1 unit tests for LongTermMemoryService lifecycle management."""

import os
import time

import pytest

from src.memory.long_term_memory.service import LongTermMemoryService


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_ltm_lifecycle.sqlite")


@pytest.fixture
def service(tmp_db, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_ENABLED", True
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_LLM_EXTRACT_ENABLED", False
    )
    return LongTermMemoryService(db_path=tmp_db)


# ---------------------------------------------------------------------------
# 1. extract_and_store_async returns future and tracks pending
# ---------------------------------------------------------------------------

def test_async_returns_future_and_tracks_pending(service):
    fut = service.extract_and_store_async(
        "我喜欢简短回答", "好的助手", user_id="u1"
    )
    assert fut is not None, "应返回 future"
    with service._futures_lock:
        assert fut in service._pending_extract_futures, "future 应在 pending set 中"


# ---------------------------------------------------------------------------
# 2. Future completes and is cleaned from pending
# ---------------------------------------------------------------------------

def test_future_completed_clears_pending(service):
    fut = service.extract_and_store_async(
        "我喜欢简短回答", "好的助手回复", user_id="u1"
    )
    # Wait for completion
    try:
        fut.result(timeout=5.0)
    except Exception:
        pass

    with service._futures_lock:
        assert fut not in service._pending_extract_futures, (
            "future 完成后应从 pending set 移除"
        )


# ---------------------------------------------------------------------------
# 3. flush_extractions waits for futures
# ---------------------------------------------------------------------------

def test_flush_extractions_waits_for_completion(service):
    futures = []
    for i in range(3):
        fut = service.extract_and_store_async(
            f"记住我喜欢简短回答{i}", f"助手回复{i}", user_id="u1"
        )
        if fut:
            futures.append(fut)

    service.flush_extractions(timeout=5.0)

    for fut in futures:
        assert fut.done(), "flush 后所有 future 应完成"

    with service._futures_lock:
        assert len(service._pending_extract_futures) == 0, (
            "flush 后 pending set 应为空"
        )


# ---------------------------------------------------------------------------
# 4. shutdown with cancel_futures is callable
# ---------------------------------------------------------------------------

def test_shutdown_cancels_futures(service):
    fut = service.extract_and_store_async(
        "记住我喜欢详细", "助手回复详细", user_id="u1"
    )
    service.shutdown(wait=False, cancel_futures=True)
    # After shutdown, should not raise
    assert service._shutdown is True

    # New submissions after shutdown return None
    fut2 = service.extract_and_store_async(
        "记住我喜欢通俗", "助手回复", user_id="u1"
    )
    assert fut2 is None, "shutdown 后不应提交新 future"


# ---------------------------------------------------------------------------
# 5. shutdown called multiple times does not raise
# ---------------------------------------------------------------------------

def test_shutdown_multiple_calls_no_error(service):
    service.shutdown(wait=False, cancel_futures=True)
    service.shutdown(wait=False, cancel_futures=True)  # second call
    service.shutdown(wait=True, cancel_futures=False)  # third call
    # Should not raise


# ---------------------------------------------------------------------------
# 6. flush after shutdown does not raise
# ---------------------------------------------------------------------------

def test_flush_after_shutdown_no_error(service):
    service.shutdown(wait=False, cancel_futures=True)
    service.flush_extractions(timeout=0.5)
    # Should not raise


# ---------------------------------------------------------------------------
# 7. flush with timeout logs warning (no crash)
# ---------------------------------------------------------------------------

def test_flush_timeout_no_crash(service):
    # Submit a long-running-ish extraction
    service.extract_and_store_async(
        "我喜欢简短" * 100, "助手" * 50, user_id="u1"
    )
    # Very short timeout — should not crash
    service.flush_extractions(timeout=0.1)
    # Clean up
    service.shutdown(wait=True, cancel_futures=True)
