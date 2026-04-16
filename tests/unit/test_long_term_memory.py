import json
import os
import tempfile
import pytest

from src.memory.long_term_memory.models import (
    CandidateMemory,
    ConfirmationStatus,
    ConflictResolution,
    EventType,
    ExtractionResult,
    MemoryEvent,
    MemoryCandidate,
    MemoryConfirmation,
    MemoryItem,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    SourceType,
    UserProfile,
    _generate_id,
    _json_dumps,
    _json_loads,
    _utcnow_iso,
)
from src.memory.long_term_memory.repository import LongTermMemoryRepository
from src.memory.long_term_memory.quality import (
    ConfidenceScorer,
    ConflictDetector,
    Deduplicator,
    ExpiryManager,
    QualityAssurance,
)
from src.memory.long_term_memory.extractor import MemoryExtractor
from src.memory.long_term_memory.candidate import CandidateManager
from src.memory.long_term_memory.event_log import ConfirmationManager, EventLogger
from src.memory.long_term_memory.prompt_injector import PromptInjector
from src.memory.long_term_memory.backup import BackupManager
from src.memory.long_term_memory.manager import LongTermMemoryManager


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_ltm.sqlite")


@pytest.fixture
def repo(tmp_db):
    r = LongTermMemoryRepository(tmp_db)
    r.initialize()
    return r


@pytest.fixture
def manager(tmp_db):
    return LongTermMemoryManager(db_path=tmp_db)


class TestModels:
    def test_memory_event_model(self):
        event = MemoryEvent(
            user_id="u1",
            event_type=MemoryType.PREFERENCE,
            key="response_style",
            value="简短",
            source_text="以后都简短回答",
            confidence=0.9,
            status="active",
        )
        assert event.event_id
        assert event.status == "active"
        assert event.to_dict()["key"] == "response_style"

    def test_candidate_memory_model(self):
        candidate = CandidateMemory(
            user_id="u1",
            event_type=MemoryType.HABIT,
            key="frequent_topics",
            value=["火星"],
        )
        assert candidate.promoted is False
        assert candidate.to_dict()["event_type"] == MemoryType.HABIT

    def test_memory_item_create(self):
        item = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        assert item.user_id == "u1"
        assert item.memory_type == MemoryType.PREFERENCE
        assert item.value == "简短"
        assert item.status == MemoryStatus.ACTIVE
        assert item.id
        assert item.created_at

    def test_memory_item_serialization(self):
        item = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.HABIT,
            category="frequent_topics", key="frequent_topics",
            value=["火星", "木星"],
        )
        d = item.to_dict()
        assert d["value"] == ["火星", "木星"]
        row = item.to_db_row()
        assert isinstance(row["value"], str)

    def test_memory_item_from_db_row(self):
        item = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.CONSTRAINT,
            category="custom", key="c1", value="test",
        )
        row = item.to_db_row()
        restored = MemoryItem.from_db_row(row)
        assert restored.user_id == item.user_id
        assert restored.value == item.value

    def test_memory_candidate(self):
        c = MemoryCandidate(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="详细",
        )
        assert c.id
        assert c.occurrence_count == 1
        item = c.to_memory_item()
        assert item.value == "详细"
        assert item.confidence >= c.confidence

    def test_memory_query_where_clause(self):
        q = MemoryQuery(user_id="u1", memory_type=MemoryType.PREFERENCE, status=MemoryStatus.ACTIVE)
        where, params, order = q.to_where_clause()
        assert "user_id = ?" in where
        assert "memory_type = ?" in where
        assert "status = ?" in where
        assert len(params) == 3

    def test_extraction_result(self):
        r = ExtractionResult(
            should_extract=True, memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
            confidence=0.8, is_explicit=True,
        )
        d = r.to_dict()
        assert d["is_explicit"] is True

    def test_json_helpers(self):
        assert _json_loads(None, {}) == {}
        assert _json_loads("", "x") == "x"
        assert _json_loads('{"a":1}', {}) == {"a": 1}
        assert _json_loads("invalid", "default") == "default"

    def test_generate_id(self):
        id1 = _generate_id()
        id2 = _generate_id()
        assert id1 != id2
        assert len(id1) == 32

    def test_utcnow_iso(self):
        result = _utcnow_iso()
        assert "T" in result or "-" in result


class TestRepository:
    def test_initialize(self, repo, tmp_db):
        assert os.path.exists(tmp_db)

    def test_add_and_get_memory(self, repo):
        item = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        repo.add_memory(item)
        loaded = repo.get_memory(item.id)
        assert loaded is not None
        assert loaded.value == "简短"

    def test_update_memory(self, repo):
        item = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        repo.add_memory(item)
        item.value = "详细"
        item.confidence = 0.9
        assert repo.update_memory(item)
        loaded = repo.get_memory(item.id)
        assert loaded.value == "详细"
        assert loaded.confidence == 0.9

    def test_delete_memory(self, repo):
        item = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        repo.add_memory(item)
        assert repo.delete_memory(item.id, "u1")
        assert repo.get_memory(item.id) is None

    def test_query_memories(self, repo):
        for i in range(5):
            item = MemoryItem.create(
                user_id="u1", memory_type=MemoryType.PREFERENCE,
                category=f"cat_{i}", key=f"key_{i}", value=f"val_{i}",
            )
            repo.add_memory(item)
        query = MemoryQuery(user_id="u1", memory_type=MemoryType.PREFERENCE, limit=3)
        results = repo.query_memories(query)
        assert len(results) == 3
        total = repo.count_memories(query)
        assert total == 5

    def test_find_by_type_key(self, repo):
        item = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        repo.add_memory(item)
        found = repo.find_memory_by_type_key("u1", MemoryType.PREFERENCE, "response_style")
        assert found is not None
        assert found.value == "简短"

    def test_versions(self, repo):
        item = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        repo.add_memory(item)
        repo.add_version(item.id, 1, "简短", 0.5, "initial")
        repo.add_version(item.id, 2, "详细", 0.8, "update")
        versions = repo.get_versions(item.id)
        assert len(versions) == 2
        assert versions[0].version == 2

    def test_candidates(self, repo):
        c = MemoryCandidate(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="详细",
        )
        repo.add_candidate(c)
        found = repo.find_candidate_by_type_key("u1", MemoryType.PREFERENCE, "response_style")
        assert found is not None
        candidates = repo.list_candidates("u1")
        assert len(candidates) == 1

    def test_event_log(self, repo):
        from src.memory.long_term_memory.models import EventLogEntry
        entry = EventLogEntry(
            user_id="u1", memory_id="m1",
            event_type=EventType.CREATED, event_detail="test",
        )
        log_id = repo.add_event_log(entry)
        assert log_id > 0
        logs = repo.get_event_logs("u1")
        assert len(logs) == 1

    def test_confirmations(self, repo):
        c = MemoryConfirmation(
            user_id="u1", memory_id="m1",
            confirmation_type="update", content="test",
        )
        repo.add_confirmation(c)
        pending = repo.list_pending_confirmations("u1")
        assert len(pending) == 1
        repo.update_confirmation_status(c.id, ConfirmationStatus.CONFIRMED)
        pending = repo.list_pending_confirmations("u1")
        assert len(pending) == 0

    def test_profile(self, repo):
        repo.save_profile("u1", {"style": "简短"}, {"topic": ["火星"]}, ["no_jargon"], {"level": "入门"}, [])
        profile = repo.load_profile("u1")
        assert profile is not None
        assert profile["preferences"]["style"] == "简短"

    def test_memory_events_repository(self, repo):
        event = MemoryEvent(
            user_id="u1",
            event_type=MemoryType.PREFERENCE,
            key="response_style",
            value="详细",
            source_text="请详细一点",
            confidence=0.85,
            status="candidate",
        )
        repo.add_event(event)
        recent = repo.get_recent_events("u1", limit=5)
        assert len(recent) == 1
        assert recent[0].key == "response_style"
        assert repo.count_similar_events("u1", "response_style", "详细") == 1
        assert repo.update_event_status(event.event_id, "active")
        assert repo.update_event_confidence(event.event_id, 0.95)
        active = repo.get_active_events("u1", limit=5)
        assert len(active) == 1
        assert active[0].confidence == 0.95

    def test_user_isolation(self, repo):
        item1 = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        item2 = MemoryItem.create(
            user_id="u2", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="详细",
        )
        repo.add_memory(item1)
        repo.add_memory(item2)
        u1_items = repo.query_memories(MemoryQuery(user_id="u1"))
        u2_items = repo.query_memories(MemoryQuery(user_id="u2"))
        assert len(u1_items) == 1
        assert len(u2_items) == 1
        assert u1_items[0].value == "简短"
        assert u2_items[0].value == "详细"

    def test_memory_stats(self, repo):
        for i in range(3):
            item = MemoryItem.create(
                user_id="u1", memory_type=MemoryType.PREFERENCE,
                category=f"cat_{i}", key=f"key_{i}", value=f"val_{i}",
            )
            repo.add_memory(item)
        stats = repo.get_memory_stats("u1")
        assert "type_counts" in stats

    def test_backup_and_restore(self, repo, tmp_path):
        item = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        repo.add_memory(item)
        backup_path = str(tmp_path / "backup.sqlite")
        assert repo.backup_database(backup_path)
        assert os.path.exists(backup_path)


class TestQuality:
    def test_confidence_scorer(self):
        scorer = ConfidenceScorer()
        explicit = scorer.initial_confidence(SourceType.EXPLICIT, True)
        auto = scorer.initial_confidence(SourceType.AUTO, False)
        assert explicit > auto
        assert explicit >= 0.8
        assert auto >= 0.4

    def test_confidence_boost(self):
        scorer = ConfidenceScorer()
        conf = scorer.initial_confidence(SourceType.AUTO)
        boosted = scorer.boost_on_confirmation(conf, 2)
        assert boosted > conf

    def test_conflict_detector(self):
        detector = ConflictDetector()
        existing = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
            confidence=0.8,
        )
        conflict = detector.detect_conflict(existing, "详细", 0.9)
        assert conflict is not None
        assert conflict.conflict_type == "value_mismatch"

    def test_no_conflict_same_value(self):
        detector = ConflictDetector()
        existing = MemoryItem.create(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        assert detector.detect_conflict(existing, "简短", 0.9) is None

    def test_deduplicator_exact(self):
        dedup = Deduplicator()
        assert dedup.is_duplicate("简短", "简短")
        assert dedup.compute_similarity("简短", "详细") < 1.0

    def test_deduplicator_list(self):
        dedup = Deduplicator()
        sim = dedup.compute_similarity(["火星", "木星"], ["火星", "木星", "土星"])
        assert 0.5 < sim < 1.0

    def test_deduplicator_merge(self):
        dedup = Deduplicator()
        merged = dedup.merge_values(["火星"], ["木星"], MemoryType.HABIT)
        assert "火星" in merged
        assert "木星" in merged

    def test_expiry_manager(self):
        mgr = ExpiryManager()
        expiry = mgr.compute_expiry_date(MemoryType.PREFERENCE, SourceType.AUTO)
        assert expiry is not None
        expiry_confirmed = mgr.compute_expiry_date(MemoryType.PREFERENCE, SourceType.CONFIRMED)
        assert expiry_confirmed is None

    def test_quality_assurance_should_store(self):
        qa = QualityAssurance(min_confidence_to_store=0.3)
        assert qa.should_store(0.5, False)
        assert not qa.should_store(0.2, False)
        assert qa.should_store(0.1, True)


class TestExtractor:
    def test_should_attempt(self):
        ext = MemoryExtractor()
        assert ext.should_attempt_extraction("我喜欢简短回答")
        assert ext.should_attempt_extraction("火星什么时候观测最好？")
        assert not ext.should_attempt_extraction("")
        assert not ext.should_attempt_extraction("a")

    def test_is_explicit(self):
        ext = MemoryExtractor()
        assert ext.is_explicit_expression("我喜欢简短回答")
        assert not ext.is_explicit_expression("今天天气怎么样")

    def test_is_temporary(self):
        ext = MemoryExtractor()
        assert ext.is_temporary_request("这次简短回答就行")
        assert not ext.is_temporary_request("我喜欢简短回答")

    def test_fallback_extraction(self):
        ext = MemoryExtractor()
        results = ext._fallback_keyword_extraction("我喜欢简短回答，我是初学者", "好的")
        assert len(results) > 0
        types = [r.memory_type for r in results]
        assert MemoryType.PREFERENCE in types

    def test_legacy_format(self):
        ext = MemoryExtractor()
        result = ext.extract_legacy_format("我喜欢简短回答", "好的")
        assert "preferences" in result
        assert result["preferences"].get("response_style") == "简短"


class TestCandidateManager:
    def test_add_candidate(self, repo):
        mgr = CandidateManager(repo)
        c = mgr.add_or_update_candidate(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        assert c.occurrence_count == 1

    def test_candidate_update_increments(self, repo):
        mgr = CandidateManager(repo)
        mgr.add_or_update_candidate(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        c = mgr.add_or_update_candidate(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        assert c.occurrence_count == 2

    def test_promote_candidate(self, repo):
        mgr = CandidateManager(repo, occurrence_threshold=2)
        c = mgr.add_or_update_candidate(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        assert not mgr.should_promote(c)
        c = mgr.add_or_update_candidate(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        assert mgr.should_promote(c)
        item = mgr.promote_candidate(c.id)
        assert item is not None
        assert item.value == "简短"

    def test_explicit_bypass(self, repo):
        mgr = CandidateManager(repo, explicit_bypass=True)
        c = mgr.add_or_update_candidate(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
            source_type=SourceType.EXPLICIT,
        )
        assert mgr.should_promote(c)


class TestEventLogger:
    def test_log_event(self, repo):
        logger = EventLogger(repo)
        logger.log_created("u1", "m1", "preference", "style", "简短")
        logs = logger.get_event_logs("u1")
        assert len(logs) == 1
        assert logs[0].event_type == EventType.CREATED

    def test_log_conflict(self, repo):
        logger = EventLogger(repo)
        logger.log_conflict("u1", "m1", "value_mismatch", ConflictResolution.NEEDS_CONFIRM, "简短", "详细")
        logs = logger.get_event_logs("u1")
        assert len(logs) == 1

    def test_confirmation_manager(self, repo):
        elogger = EventLogger(repo)
        cmgr = ConfirmationManager(repo, elogger)
        c = cmgr.create_confirmation("u1", "m1", "update", "test content")
        assert c.status == ConfirmationStatus.PENDING
        pending = cmgr.list_pending_confirmations("u1")
        assert len(pending) == 1
        resolved = cmgr.resolve_confirmation(c.id, ConfirmationStatus.CONFIRMED)
        assert resolved.status == ConfirmationStatus.CONFIRMED


class TestPromptInjector:
    def test_classify_task_type(self, repo):
        injector = PromptInjector(repo)
        assert injector.classify_task_type("今晚观测什么？") == "observation"
        assert injector.classify_task_type("什么是黑洞？") == "learning"
        assert injector.classify_task_type("你好") == "general"

    def test_format_profile_empty(self, repo):
        injector = PromptInjector(repo)
        result = injector.format_profile_for_prompt("u1")
        assert "暂无" in result

    def test_format_profile_with_data(self, repo):
        repo.save_profile("u1", {"response_style": "简短"}, {}, [], {}, [])
        injector = PromptInjector(repo)
        result = injector.format_profile_for_prompt("u1")
        assert "简短" in result


class TestBackupManager:
    def test_create_backup(self, repo, tmp_path):
        mgr = BackupManager(repo, backup_dir=str(tmp_path / "backups"))
        backup_path = mgr.create_backup(tag="test")
        assert backup_path is not None
        assert os.path.exists(backup_path)

    def test_list_backups(self, repo, tmp_path):
        mgr = BackupManager(repo, backup_dir=str(tmp_path / "backups"))
        mgr.create_backup(tag="test1")
        mgr.create_backup(tag="test2")
        backups = mgr.list_backups()
        assert len(backups) == 2


class TestLongTermMemoryManager:
    def test_add_memory(self, manager):
        item = manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
            confidence=0.8, source_type=SourceType.EXPLICIT, is_explicit=True,
        )
        assert item is not None
        assert item.value == "简短"

    def test_get_memory(self, manager):
        item = manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        loaded = manager.get_memory(item.id)
        assert loaded is not None
        assert loaded.value == "简短"

    def test_update_memory(self, manager):
        item = manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        updated = manager.update_memory(item.id, "u1", value="详细")
        assert updated is not None
        assert updated.value == "详细"

    def test_delete_memory(self, manager):
        item = manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        assert manager.delete_memory(item.id, "u1")
        assert manager.get_memory(item.id) is None

    def test_query_memories(self, manager):
        for i in range(3):
            manager.add_memory(
                user_id="u1", memory_type=MemoryType.PREFERENCE,
                category=f"cat_{i}", key=f"key_{i}", value=f"val_{i}",
            )
        results = manager.query_memories(MemoryQuery(user_id="u1"))
        assert len(results) == 3

    def test_memory_versions(self, manager):
        item = manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        manager.update_memory(item.id, "u1", value="详细")
        versions = manager.get_memory_versions(item.id)
        assert len(versions) >= 1

    def test_extract_and_store(self, manager):
        results = manager.extract_and_store(
            user_message="我喜欢简短回答，我是初学者",
            assistant_message="好的，我会简洁回答",
            user_id="u1",
        )
        assert len(results) >= 1

    def test_format_profile_for_prompt(self, manager):
        manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
            source_type=SourceType.EXPLICIT, is_explicit=True,
        )
        result = manager.format_profile_for_prompt("u1")
        assert isinstance(result, str)

    def test_load_profile(self, manager):
        manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
            source_type=SourceType.EXPLICIT, is_explicit=True,
        )
        profile = manager.load_profile("u1")
        assert profile is not None

    def test_delete_profile(self, manager):
        manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        assert manager.delete_profile("u1")

    def test_candidate_flow(self, manager):
        result = manager.extract_and_store(
            user_message="我经常看火星",
            assistant_message="好的，已记录",
            user_id="u1",
        )
        candidates = manager.list_candidates("u1")
        assert len(candidates) >= 0

    def test_event_logs(self, manager):
        manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        logs = manager.get_event_logs("u1")
        assert len(logs) >= 1

    def test_maintenance(self, manager):
        manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        result = manager.run_maintenance("u1")
        assert "expired" in result
        assert "archived" in result
        assert "promoted" in result

    def test_stats(self, manager):
        manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        stats = manager.get_stats("u1")
        assert "type_counts" in stats

    def test_backup(self, manager):
        backup_path = manager.create_backup(tag="test")
        assert backup_path is not None
        backups = manager.list_backups()
        assert len(backups) >= 1

    def test_merge_and_update(self, manager):
        result = manager.merge_and_update("u1", {
            "preferences": {"response_style": "简短"},
            "habits": {"frequent_topics": ["火星"]},
            "constraints": ["避免术语"],
            "background": {"skill_level": "入门"},
            "facts": [],
        })
        assert result is not None

    def test_event_promotion_and_debug(self, manager):
        result = manager.merge_and_update("u1", {
            "preferences": {"response_style": "详细"},
            "constraints": ["避免术语"],
        })
        assert result["preferences"]["response_style"] == "详细"
        events = manager.debug_events("u1")
        assert any(event["key"] == "response_style" for event in events)
        debug = manager.debug_user_memory("u1")
        assert debug["profile"]["preferences"]["response_style"] == "详细"
        formatted = manager.format_profile_for_prompt("u1", task_type="qa")
        assert "近期记忆事件" in formatted

    def test_user_isolation(self, manager):
        manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        manager.add_memory(
            user_id="u2", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="详细",
        )
        u1 = manager.query_memories(MemoryQuery(user_id="u1"))
        u2 = manager.query_memories(MemoryQuery(user_id="u2"))
        assert len(u1) == 1
        assert len(u2) == 1
        assert u1[0].value == "简短"
        assert u2[0].value == "详细"

    def test_conflict_handling(self, manager):
        manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
            confidence=0.9, source_type=SourceType.EXPLICIT, is_explicit=True,
        )
        manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="详细",
            confidence=0.5, source_type=SourceType.AUTO,
        )
        items = manager.query_memories(MemoryQuery(user_id="u1", memory_type=MemoryType.PREFERENCE))
        assert len(items) >= 1

    def test_export_snapshot(self, manager):
        manager.add_memory(
            user_id="u1", memory_type=MemoryType.PREFERENCE,
            category="response_style", key="response_style", value="简短",
        )
        snapshot = manager.export_profile_snapshot("u1")
        assert snapshot is not None
        assert "user_id" in snapshot
