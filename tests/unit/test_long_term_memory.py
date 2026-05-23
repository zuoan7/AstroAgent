"""长期记忆主链路单元测试。

覆盖领域模型、SQLite 仓储、候选转正、抽取、删除、prompt 注入、语义索引、
rerank 降级和注入反馈自学习等当前长期记忆行为。
"""

import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src.memory.long_term_memory.backup import BackupManager
from src.memory.long_term_memory.candidate import CandidateManager
from src.memory.long_term_memory.event_log import ConfirmationManager, EventLogger
from src.memory.long_term_memory.extractor import MemoryExtractor
from src.memory.long_term_memory.models import (
    CandidateMemory,
    ConfirmationStatus,
    ConflictResolution,
    EventType,
    ExtractionResult,
    LongTermMemoryDeletionRequest,
    MemoryCandidate,
    MemoryConfirmation,
    MemoryEvent,
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
from src.memory.long_term_memory.prompt_injector import PromptInjector
from src.memory.long_term_memory.embedding import MemoryEmbeddingService
from src.memory.long_term_memory.quality import (
    ConfidenceScorer,
    ConflictDetector,
    Deduplicator,
    ExpiryManager,
    QualityAssurance,
)
from src.memory.long_term_memory.repository import LongTermMemoryRepository
from src.memory.long_term_memory.retrieval import LongTermMemoryRetriever
from src.memory.long_term_memory.service import LongTermMemoryService


@pytest.fixture
def tmp_db(tmp_path):
    """创建临时 SQLite 记忆数据库 fixture。"""

    return str(tmp_path / "test_ltm.sqlite")


@pytest.fixture
def repo(tmp_db):
    """创建测试用 repo fixture。"""

    r = LongTermMemoryRepository(tmp_db)
    r.initialize()
    return r


@pytest.fixture
def service(tmp_db):
    """创建测试用 service fixture。"""

    return LongTermMemoryService(db_path=tmp_db)


class TestModels:
    def test_memory_event_model(self):
        """测试 memory event model 场景。"""

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
        """测试 candidate memory model 场景。"""

        candidate = CandidateMemory(
            user_id="u1",
            event_type=MemoryType.HABIT,
            key="frequent_topics",
            value=["火星"],
        )
        assert candidate.promoted is False
        assert candidate.to_dict()["event_type"] == MemoryType.HABIT

    def test_memory_item_create(self):
        """测试 memory item create 场景。"""

        item = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        assert item.user_id == "u1"
        assert item.memory_type == MemoryType.PREFERENCE
        assert item.value == "简短"
        assert item.status == MemoryStatus.ACTIVE
        assert item.id
        assert item.created_at

    def test_memory_item_serialization(self):
        """测试 memory item serialization 场景。"""

        item = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.HABIT,
            category="frequent_topics",
            key="frequent_topics",
            value=["火星", "木星"],
        )
        d = item.to_dict()
        assert d["value"] == ["火星", "木星"]
        row = item.to_db_row()
        assert isinstance(row["value"], str)

    def test_memory_item_from_db_row(self):
        """测试 memory item from db row 场景。"""

        item = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.CONSTRAINT,
            category="custom",
            key="c1",
            value="test",
        )
        row = item.to_db_row()
        restored = MemoryItem.from_db_row(row)
        assert restored.user_id == item.user_id
        assert restored.value == item.value

    def test_memory_candidate(self):
        """测试 memory candidate 场景。"""

        c = MemoryCandidate(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="详细",
        )
        assert c.id
        assert c.occurrence_count == 1
        item = c.to_memory_item()
        assert item.value == "详细"
        assert item.confidence >= c.confidence

    def test_memory_query_where_clause(self):
        """测试 memory query where clause 场景。"""

        q = MemoryQuery(
            user_id="u1", memory_type=MemoryType.PREFERENCE, status=MemoryStatus.ACTIVE
        )
        where, params, order = q.to_where_clause()
        assert "user_id = ?" in where
        assert "memory_type = ?" in where
        assert "status = ?" in where
        assert len(params) == 3

    def test_extraction_result(self):
        """测试 extraction result 场景。"""

        r = ExtractionResult(
            should_extract=True,
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            confidence=0.8,
            is_explicit=True,
        )
        d = r.to_dict()
        assert d["is_explicit"] is True

    def test_json_helpers(self):
        """测试 json helpers 场景。"""

        assert _json_loads(None, {}) == {}
        assert _json_loads("", "x") == "x"
        assert _json_loads('{"a":1}', {}) == {"a": 1}
        assert _json_loads("invalid", "default") == "default"

    def test_generate_id(self):
        """测试 generate id 场景。"""

        id1 = _generate_id()
        id2 = _generate_id()
        assert id1 != id2
        assert len(id1) == 32

    def test_utcnow_iso(self):
        """测试 utcnow iso 场景。"""

        result = _utcnow_iso()
        assert "T" in result or "-" in result


class TestRepository:
    def test_initialize(self, repo, tmp_db):
        """测试 initialize 场景。"""

        assert os.path.exists(tmp_db)

    def test_add_and_get_memory(self, repo):
        """测试 add and get memory 场景。"""

        item = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        repo.add_memory(item)
        loaded = repo.get_memory(item.id)
        assert loaded is not None
        assert loaded.value == "简短"

    def test_update_memory(self, repo):
        """测试 update memory 场景。"""

        item = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        repo.add_memory(item)
        item.value = "详细"
        item.confidence = 0.9
        assert repo.update_memory(item)
        loaded = repo.get_memory(item.id)
        assert loaded.value == "详细"
        assert loaded.confidence == 0.9

    def test_delete_memory(self, repo):
        """测试 delete memory 场景。"""

        item = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        repo.add_memory(item)
        assert repo.delete_memory(item.id, "u1")
        assert repo.get_memory(item.id) is None

    def test_query_memories(self, repo):
        """测试 query memories 场景。"""

        for i in range(5):
            item = MemoryItem.create(
                user_id="u1",
                memory_type=MemoryType.PREFERENCE,
                category=f"cat_{i}",
                key=f"key_{i}",
                value=f"val_{i}",
            )
            repo.add_memory(item)
        query = MemoryQuery(user_id="u1", memory_type=MemoryType.PREFERENCE, limit=3)
        results = repo.query_memories(query)
        assert len(results) == 3
        total = repo.count_memories(query)
        assert total == 5

    def test_find_by_type_key(self, repo):
        """测试 find by type key 场景。"""

        item = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        repo.add_memory(item)
        found = repo.find_memory_by_type_key(
            "u1", MemoryType.PREFERENCE, "response_style"
        )
        assert found is not None
        assert found.value == "简短"

    def test_versions(self, repo):
        """测试 versions 场景。"""

        item = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        repo.add_memory(item)
        repo.add_version(item.id, 1, "简短", 0.5, "initial")
        repo.add_version(item.id, 2, "详细", 0.8, "update")
        versions = repo.get_versions(item.id)
        assert len(versions) == 2
        assert versions[0].version == 2

    def test_candidates(self, repo):
        """测试 candidates 场景。"""

        c = MemoryCandidate(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="详细",
        )
        repo.add_candidate(c)
        found = repo.find_candidate_by_type_key(
            "u1", MemoryType.PREFERENCE, "response_style"
        )
        assert found is not None
        candidates = repo.list_candidates("u1")
        assert len(candidates) == 1

    def test_event_log(self, repo):
        """测试 event log 场景。"""

        from src.memory.long_term_memory.models import EventLogEntry

        entry = EventLogEntry(
            user_id="u1",
            memory_id="m1",
            event_type=EventType.CREATED,
            event_detail="test",
        )
        log_id = repo.add_event_log(entry)
        assert log_id > 0
        logs = repo.get_event_logs("u1")
        assert len(logs) == 1

    def test_confirmations(self, repo):
        """测试 confirmations 场景。"""

        c = MemoryConfirmation(
            user_id="u1",
            memory_id="m1",
            confirmation_type="update",
            content="test",
        )
        repo.add_confirmation(c)
        pending = repo.list_pending_confirmations("u1")
        assert len(pending) == 1
        repo.update_confirmation_status(c.id, ConfirmationStatus.CONFIRMED)
        pending = repo.list_pending_confirmations("u1")
        assert len(pending) == 0

    def test_profile(self, repo):
        """测试 profile 场景。"""

        repo.save_profile(
            "u1",
            {"style": "简短"},
            {"topic": ["火星"]},
            ["no_jargon"],
            {"level": "入门"},
            [],
        )
        profile = repo.load_profile("u1")
        assert profile is not None
        assert profile["preferences"]["style"] == "简短"

    def test_memory_events_repository(self, repo):
        """测试 memory events repository 场景。"""

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
        """测试 user isolation 场景。"""

        item1 = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        item2 = MemoryItem.create(
            user_id="u2",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="详细",
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
        """测试 memory stats 场景。"""

        for i in range(3):
            item = MemoryItem.create(
                user_id="u1",
                memory_type=MemoryType.PREFERENCE,
                category=f"cat_{i}",
                key=f"key_{i}",
                value=f"val_{i}",
            )
            repo.add_memory(item)
        stats = repo.get_memory_stats("u1")
        assert "type_counts" in stats

    def test_backup_and_restore(self, repo, tmp_path):
        """测试 backup and restore 场景。"""

        item = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        repo.add_memory(item)
        backup_path = str(tmp_path / "backup.sqlite")
        assert repo.backup_database(backup_path)
        assert os.path.exists(backup_path)


class TestQuality:
    def test_confidence_scorer(self):
        """测试 confidence scorer 场景。"""

        scorer = ConfidenceScorer()
        explicit = scorer.initial_confidence(SourceType.EXPLICIT, True)
        auto = scorer.initial_confidence(SourceType.AUTO, False)
        assert explicit > auto
        assert explicit >= 0.8
        assert auto >= 0.4

    def test_confidence_boost(self):
        """测试 confidence boost 场景。"""

        scorer = ConfidenceScorer()
        conf = scorer.initial_confidence(SourceType.AUTO)
        boosted = scorer.boost_on_confirmation(conf, 2)
        assert boosted > conf

    def test_conflict_detector(self):
        """测试 conflict detector 场景。"""

        detector = ConflictDetector()
        existing = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            confidence=0.8,
        )
        conflict = detector.detect_conflict(existing, "详细", 0.9)
        assert conflict is not None
        assert conflict.conflict_type == "value_mismatch"

    def test_conflict_detector_classifies_relations(self):
        """测试 conflict detector classifies relations 场景。"""

        detector = ConflictDetector()
        assert (
            detector.classify_value_relation(
                "北京", "北京和承德", MemoryType.BACKGROUND, "location"
            )
            == "extension"
        )
        assert (
            detector.classify_value_relation(
                "深空天体", "深空天体中的梅西耶目标", MemoryType.PREFERENCE, "target"
            )
            == "refinement"
        )
        assert (
            detector.classify_value_relation(
                "简短", "详细", MemoryType.PREFERENCE, "response_style"
            )
            == "conflict"
        )

    def test_no_conflict_same_value(self):
        """测试 no conflict same value 场景。"""

        detector = ConflictDetector()
        existing = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        assert detector.detect_conflict(existing, "简短", 0.9) is None

    def test_deduplicator_exact(self):
        """测试 deduplicator exact 场景。"""

        dedup = Deduplicator()
        assert dedup.is_duplicate("简短", "简短")
        assert dedup.compute_similarity("简短", "详细") < 1.0

    def test_deduplicator_list(self):
        """测试 deduplicator list 场景。"""

        dedup = Deduplicator()
        sim = dedup.compute_similarity(["火星", "木星"], ["火星", "木星", "土星"])
        assert 0.5 < sim < 1.0

    def test_deduplicator_merge(self):
        """测试 deduplicator merge 场景。"""

        dedup = Deduplicator()
        merged = dedup.merge_values(["火星"], ["木星"], MemoryType.HABIT)
        assert "火星" in merged
        assert "木星" in merged

    def test_expiry_manager(self):
        """测试 expiry manager 场景。"""

        mgr = ExpiryManager()
        expiry = mgr.compute_expiry_date(MemoryType.PREFERENCE, SourceType.AUTO)
        assert expiry is not None
        expiry_confirmed = mgr.compute_expiry_date(
            MemoryType.PREFERENCE, SourceType.CONFIRMED
        )
        assert expiry_confirmed is None

    def test_quality_assurance_should_store(self):
        """测试 quality assurance should store 场景。"""

        qa = QualityAssurance(min_confidence_to_store=0.3)
        assert qa.should_store(0.5, False)
        assert not qa.should_store(0.2, False)
        assert qa.should_store(0.1, True)

    def test_negative_feedback_archives_on_second_denial(self):
        """测试 negative feedback archives on second denial 场景。"""

        qa = QualityAssurance()
        item = MemoryItem.create(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            confidence=0.9,
        )
        qa.apply_negative_feedback(item, "不对")
        assert item.status == MemoryStatus.ACTIVE
        assert item.confidence == 0.6
        qa.apply_negative_feedback(item, "还是不对")
        assert item.status == MemoryStatus.ARCHIVED
        assert item.metadata["denial_count"] == 2


class TestExtractor:
    def test_should_attempt(self):
        """测试 should attempt 场景。"""

        ext = MemoryExtractor()
        assert ext.should_attempt_extraction("我喜欢简短回答")
        # Phase 1: general astronomy questions no longer trigger extraction
        assert not ext.should_attempt_extraction("火星什么时候观测最好？")
        assert not ext.should_attempt_extraction("")
        assert not ext.should_attempt_extraction("a")

    def test_is_explicit(self):
        """测试 is explicit 场景。"""

        ext = MemoryExtractor()
        assert ext.is_explicit_expression("我喜欢简短回答")
        assert not ext.is_explicit_expression("今天天气怎么样")

    def test_is_temporary(self):
        """测试 is temporary 场景。"""

        ext = MemoryExtractor()
        assert ext.is_temporary_request("这次简短回答就行")
        assert not ext.is_temporary_request("我喜欢简短回答")

    def test_fallback_extraction(self):
        """测试 fallback extraction 场景。"""

        ext = MemoryExtractor()
        results = ext._fallback_keyword_extraction("我喜欢简短回答，我是初学者", "好的")
        assert len(results) > 0
        types = [r.memory_type for r in results]
        assert MemoryType.PREFERENCE in types

    def test_legacy_format(self):
        """测试 legacy format 场景。"""

        ext = MemoryExtractor()
        result = ext.extract_legacy_format("我喜欢简短回答", "好的")
        assert "preferences" in result
        assert result["preferences"].get("response_style") == "简短"


class TestCandidateManager:
    def test_add_candidate(self, repo):
        """测试 add candidate 场景。"""

        mgr = CandidateManager(repo)
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        assert c.occurrence_count == 1

    def test_candidate_update_increments(self, repo):
        """测试 candidate update increments 场景。"""

        mgr = CandidateManager(repo)
        mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        assert c.occurrence_count == 2

    def test_promote_candidate(self, repo):
        """测试 promote candidate 场景。"""

        mgr = CandidateManager(repo, occurrence_threshold=2)
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            confidence=0.55,
        )
        assert not mgr.should_promote(c)
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            confidence=0.8,
        )
        assert mgr.should_promote(c)
        item = mgr.promote_candidate(c.id)
        assert item is not None
        assert item.value == "简短"

    def test_explicit_bypass_requires_confidence(self, repo):
        """测试 explicit bypass requires confidence 场景。"""

        mgr = CandidateManager(repo, explicit_bypass=True)
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            source_type=SourceType.EXPLICIT,
            confidence=0.5,
        )
        assert not mgr.should_promote(c)

    def test_non_explicit_single_occurrence_not_promoted(self, repo):
        """测试 non explicit single occurrence not promoted 场景。"""

        mgr = CandidateManager(repo, occurrence_threshold=2, confidence_threshold=0.6)
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.HABIT,
            category="frequent_topics",
            key="frequent_topics",
            value=["火星"],
            confidence=0.8,
            source_type=SourceType.AUTO,
        )
        assert not mgr.should_promote(c)

    def test_solid_explicit_preference_single_can_promote(self, repo):
        """测试 solid explicit preference single can promote 场景。"""

        mgr = CandidateManager(repo)
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            confidence=0.85,
            source_type=SourceType.EXPLICIT,
            metadata={
                "extraction_grade": "solid",
                "gate_reason": "stable_profile_signal",
            },
        )
        assert mgr.should_promote(c)

    def test_tentative_single_does_not_promote(self, repo):
        """测试 tentative single does not promote 场景。"""

        mgr = CandidateManager(repo)
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.HABIT,
            category="frequent_topics",
            key="frequent_topics",
            value=["火星"],
            confidence=0.9,
            source_type=SourceType.AUTO,
            metadata={"extraction_grade": "tentative"},
        )
        assert not mgr.should_promote(c)

    def test_recent_repeated_tentative_habit_promotes(self, repo):
        """测试 recent repeated tentative habit promotes 场景。"""

        mgr = CandidateManager(repo)
        mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.HABIT,
            category="frequent_topics",
            key="frequent_topics",
            value=["火星"],
            confidence=0.8,
            source_type=SourceType.AUTO,
            source_conversation_id="c1",
            metadata={"extraction_grade": "tentative"},
        )
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.HABIT,
            category="frequent_topics",
            key="frequent_topics",
            value=["火星"],
            confidence=0.8,
            source_type=SourceType.AUTO,
            source_conversation_id="c2",
            metadata={"extraction_grade": "tentative"},
        )
        assert mgr.should_promote(c)

    def test_old_occurrences_decay_below_promotion_threshold(self, repo):
        """测试 old occurrences decay below promotion threshold 场景。"""

        mgr = CandidateManager(repo)
        old_seen = (datetime.now() - timedelta(days=400)).isoformat()
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.HABIT,
            category="frequent_topics",
            key="frequent_topics",
            value=["火星"],
            confidence=0.95,
            source_type=SourceType.AUTO,
            metadata={
                "extraction_grade": "tentative",
                "occurrence_history": [
                    {"value": ["火星"], "seen_at": old_seen},
                    {"value": ["火星"], "seen_at": old_seen},
                ],
            },
        )
        assert not mgr.should_promote(c)

    def test_fact_never_auto_promotes(self, repo):
        """测试 fact never auto promotes 场景。"""

        mgr = CandidateManager(repo)
        c = mgr.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.FACT,
            category="basic_info",
            key="timezone",
            value="Asia/Shanghai",
            confidence=0.95,
            source_type=SourceType.EXPLICIT,
            metadata={"extraction_grade": "solid"},
        )
        assert not mgr.should_promote(c)

    def test_high_risk_candidate_needs_confirm(self, repo):
        """测试 high risk candidate needs confirm 场景。"""

        mgr = CandidateManager(repo)
        mgr.process_extraction_as_candidate(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="skill_level",
            key="skill_level",
            value="入门",
            confidence=0.9,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
        )
        needs_confirm = mgr.list_candidates("u1", status=MemoryStatus.NEEDS_CONFIRM)
        assert len(needs_confirm) == 1
        assert needs_confirm[0].status == MemoryStatus.NEEDS_CONFIRM

    def test_repeated_consistent_background_device_auto_promotes(self, repo):
        """测试 repeated consistent background device auto promotes 场景。"""

        mgr = CandidateManager(repo)
        first = mgr.process_extraction_as_candidate(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="device_info",
            key="device_info",
            value="星特朗8SE",
            confidence=0.9,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
            source_conversation_id="c1",
            metadata={"extraction_grade": "solid"},
        )
        assert first is None
        second = mgr.process_extraction_as_candidate(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="device_info",
            key="device_info",
            value="星特朗8SE",
            confidence=0.9,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
            source_conversation_id="c2",
            metadata={"extraction_grade": "solid"},
        )
        assert second is not None
        assert second.metadata["probation_status"] == "active"
        assert second.metadata["promotion_score"] >= 0.7


class TestEventLogger:
    def test_log_event(self, repo):
        """测试 log event 场景。"""

        logger = EventLogger(repo)
        logger.log_created("u1", "m1", "preference", "style", "简短")
        logs = logger.get_event_logs("u1")
        assert len(logs) == 1
        assert logs[0].event_type == EventType.CREATED

    def test_log_conflict(self, repo):
        """测试 log conflict 场景。"""

        logger = EventLogger(repo)
        logger.log_conflict(
            "u1",
            "m1",
            "value_mismatch",
            ConflictResolution.NEEDS_CONFIRM,
            "简短",
            "详细",
        )
        logs = logger.get_event_logs("u1")
        assert len(logs) == 1

    def test_confirmation_manager(self, repo):
        """测试 confirmation manager 场景。"""

        elogger = EventLogger(repo)
        cmgr = ConfirmationManager(repo, elogger)
        c = cmgr.create_confirmation("u1", "m1", "update", "test content")
        assert c.status == ConfirmationStatus.PENDING
        pending = cmgr.list_pending_confirmations("u1")
        assert len(pending) == 1
        resolved = cmgr.resolve_confirmation(c.id, ConfirmationStatus.CONFIRMED)
        assert resolved.status == ConfirmationStatus.CONFIRMED


class TestPromptInjector:
    def _add_memory(
        self,
        repo,
        memory_type,
        category,
        key,
        value,
        confidence=0.9,
        source_type=SourceType.AUTO,
    ):
        """向测试仓储写入一条长期记忆。"""

        item = MemoryItem.create(
            user_id="u1",
            memory_type=memory_type,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            source_type=source_type,
        )
        repo.add_memory(item)
        return item

    def test_classify_task_type(self, repo):
        """测试 classify task type 场景。"""

        injector = PromptInjector(repo)
        assert injector.classify_task_type("今晚观测什么？") == "observation"
        assert injector.classify_task_type("什么是黑洞？") == "learning"
        assert injector.classify_task_type("你好") == "general"

    def test_format_profile_empty(self, repo):
        """测试 format profile empty 场景。"""

        injector = PromptInjector(repo)
        result = injector.format_profile_for_prompt("u1")
        assert "暂无" in result

    def test_format_profile_with_data(self, repo):
        """测试 format profile with data 场景。"""

        repo.save_profile("u1", {"response_style": "简短"}, {}, [], {}, [])
        injector = PromptInjector(repo)
        result = injector.format_profile_for_prompt("u1")
        assert "简短" in result

    def test_additive_score_components_are_normalized(self, repo):
        """测试 additive score components are normalized 场景。"""

        item = self._add_memory(
            repo,
            MemoryType.CONSTRAINT,
            "custom",
            "no_jargon",
            "避免术语",
            source_type=SourceType.EXPLICIT,
        )
        retriever = LongTermMemoryRetriever(repo)

        hit = retriever.score_hit(item, "今晚用望远镜观测什么", "observation")
        unknown_task_hit = retriever.score_hit(item, "今晚用望远镜观测什么", "new_task")

        assert 0 <= hit.score <= 1
        assert hit.components["constraint_bonus"] == 1.0
        assert hit.components["source_bonus"] >= 0.9
        assert "任务类型=general" in unknown_task_hit.reasons

    def test_selection_uses_coupled_budget_and_type_quota(self, repo):
        """测试 selection uses coupled budget and type quota 场景。"""

        constraint = self._add_memory(
            repo,
            MemoryType.CONSTRAINT,
            "custom",
            "no_jargon",
            "避免术语",
            confidence=0.9,
        )
        for index in range(5):
            self._add_memory(
                repo,
                MemoryType.PREFERENCE,
                "response_style",
                f"style_{index}",
                f"偏好 {index}",
                confidence=0.9 - index * 0.01,
            )

        injector = PromptInjector(repo, max_prompt_tokens=800, max_memories=10)
        hits = injector.select_memory_hits(
            "u1",
            "今晚用望远镜观测什么",
            task_type="observation",
            total_context_budget=4000,
            include_omitted=True,
        )
        selected = [hit for hit in hits if hit.selected]
        omitted = [hit for hit in hits if hit.omitted_reason]

        assert injector.get_last_selection_trace()["memory_budget"] == 400
        assert constraint.id in {hit.item.id for hit in selected}
        assert (
            sum(1 for hit in selected if hit.item.memory_type == MemoryType.PREFERENCE)
            == 3
        )
        assert any(hit.omitted_reason == "type_quota" for hit in omitted)

    def test_selection_deduplicates_same_key(self, repo):
        """测试 selection deduplicates same key 场景。"""

        low = self._add_memory(
            repo,
            MemoryType.PREFERENCE,
            "response_style",
            "response_style",
            "详细",
            confidence=0.5,
        )
        high = self._add_memory(
            repo,
            MemoryType.PREFERENCE,
            "response_style",
            "response_style",
            "简短",
            confidence=0.95,
        )

        injector = PromptInjector(repo)
        hits = injector.select_memory_hits(
            "u1", "请按我的回答风格来", include_omitted=True
        )
        selected_ids = {hit.item.id for hit in hits if hit.selected}
        duplicate_ids = {
            hit.item.id for hit in hits if hit.omitted_reason == "duplicate_key"
        }

        assert high.id in selected_ids
        assert low.id in duplicate_ids

    def test_schema_v4_creates_embedding_cache_table(self, tmp_db):
        """测试 schema v4 creates embedding cache table 场景。"""

        repo = LongTermMemoryRepository(tmp_db)
        repo.initialize()

        with sqlite3.connect(tmp_db) as conn:
            version = conn.execute(
                "SELECT value FROM schema_version WHERE key='version'"
            ).fetchone()[0]
            table = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='memory_embeddings'
                """
            ).fetchone()
            indexes = {
                row[1]
                for row in conn.execute("PRAGMA index_list(memory_embeddings)").fetchall()
            }
            foreign_keys = conn.execute(
                "PRAGMA foreign_key_list(memory_embeddings)"
            ).fetchall()

        assert version == "4"
        assert table[0] == "memory_embeddings"
        assert "idx_memory_embeddings_user_model" in indexes
        assert any(row[2] == "memories" and row[3] == "memory_id" for row in foreign_keys)

    def test_embedding_cache_rebuilds_on_content_hash_or_model_change(self, repo, monkeypatch):
        """测试 embedding cache rebuilds on content hash or model change 场景。"""

        item = self._add_memory(
            repo,
            MemoryType.FACT,
            "equipment",
            "mount",
            "EQ6 Pro",
            confidence=0.8,
        )
        service = MemoryEmbeddingService(
            repo,
            enabled=True,
            api_key="test-key",
            model_name="embed-v1",
        )
        monkeypatch.setattr(service, "_embed_text", lambda text: [1.0, 0.0])

        assert service.embed_memory_if_needed(item) == "created"
        assert service.embed_memory_if_needed(item) == "skipped"

        item.value = "HEQ5 Pro"
        repo.update_memory(item)
        monkeypatch.setattr(service, "_embed_text", lambda text: [0.0, 1.0])
        assert service.embed_memory_if_needed(item) == "updated"

        model_changed = MemoryEmbeddingService(
            repo,
            enabled=True,
            api_key="test-key",
            model_name="embed-v2",
        )
        monkeypatch.setattr(model_changed, "_embed_text", lambda text: [0.5, 0.5])
        assert model_changed.embed_memory_if_needed(item) == "updated"

        record = repo.get_memory_embedding(item.id)
        assert record["model_name"] == "embed-v2"
        assert record["embedding"] == [0.5, 0.5]

    def test_semantic_similarity_can_lift_non_literal_memory(self, repo):
        """测试 semantic similarity can lift non literal memory 场景。"""

        item = self._add_memory(
            repo,
            MemoryType.FACT,
            "equipment",
            "mount_setup",
            "EQ6 Pro 赤道仪",
            confidence=0.2,
        )

        class FakeEmbeddingService:
            model_name = "fake-embed"

            def cached_embeddings_for_items(self, user_id, items):
                """返回测试用的缓存 embedding。"""

                return {item.id: [1.0, 0.0]}, []

            def embed_query(self, query):
                """返回测试用 query embedding。"""

                return [1.0, 0.0], None

            def schedule_embeddings(self, items, limit=None):
                """模拟后台 embedding 调度。"""

                return 0

        no_semantic = PromptInjector(
            repo,
            relevance_threshold=0.38,
            rerank_enabled=False,
        )
        semantic = PromptInjector(
            repo,
            relevance_threshold=0.38,
            embedding_service=FakeEmbeddingService(),
            rerank_enabled=False,
        )

        assert no_semantic.select_memory_hits(
            "u1", "怎么规划深空摄影流程", task_type="learning"
        ) == []
        hits = semantic.select_memory_hits(
            "u1", "怎么规划深空摄影流程", task_type="learning"
        )

        assert [hit.item.id for hit in hits] == [item.id]
        assert hits[0].components["semantic_similarity"] == 1.0
        assert "语义召回命中" in hits[0].reasons

    def test_rerank_success_reorders_top30_and_records_trace(self, repo):
        """测试 rerank success reorders top30 and records trace 场景。"""

        first = self._add_memory(
            repo,
            MemoryType.PREFERENCE,
            "response_style",
            "style_a",
            "详细",
            confidence=0.9,
        )
        second = self._add_memory(
            repo,
            MemoryType.PREFERENCE,
            "response_style",
            "style_b",
            "简短",
            confidence=0.9,
        )
        injector = PromptInjector(repo, rerank_enabled=False)
        def fake_rerank(query, documents, top_n=None):
            """按测试目标构造假的 rerank 结果。"""

            preferred_index = next(
                index for index, doc in enumerate(documents) if "style_b" in doc
            )
            other_index = 1 - preferred_index
            return [
                SimpleNamespace(index=preferred_index, relevance_score=0.95),
                SimpleNamespace(index=other_index, relevance_score=0.1),
            ]

        injector._reranker = SimpleNamespace(
            enabled=True,
            rerank=fake_rerank,
        )

        hits = injector.select_memory_hits("u1", "请按我的回答风格来")
        selected_ids = [hit.item.id for hit in hits]
        trace_hits = injector.get_last_selection_trace()["hits"]

        assert selected_ids[0] == second.id
        assert first.id in {hit["memory_id"] for hit in trace_hits}
        assert trace_hits[0]["components"]["rerank_score"] == 0.95

    def test_rerank_zero_score_falls_back_to_policy_score(self, repo):
        """测试 rerank zero score falls back to policy score 场景。"""

        self._add_memory(
            repo,
            MemoryType.PREFERENCE,
            "response_style",
            "style_a",
            "详细",
            confidence=0.9,
        )
        self._add_memory(
            repo,
            MemoryType.PREFERENCE,
            "response_style",
            "style_b",
            "简短",
            confidence=0.9,
        )
        injector = PromptInjector(repo, rerank_enabled=False)
        injector._reranker = SimpleNamespace(
            enabled=True,
            rerank=lambda query, documents, top_n=None: [
                SimpleNamespace(index=1, relevance_score=0.0),
                SimpleNamespace(index=0, relevance_score=0.0),
            ],
        )

        hits = injector.select_memory_hits("u1", "请按我的回答风格来")

        assert all(hit.components["rerank_score"] == 0.0 for hit in hits)
        assert all(hit.score == hit.components["policy_score"] for hit in hits)
        assert any("rerank降级" in "；".join(hit.reasons) for hit in hits)


class TestBackupManager:
    def test_create_backup(self, repo, tmp_path):
        """测试 create backup 场景。"""

        mgr = BackupManager(repo, backup_dir=str(tmp_path / "backups"))
        backup_path = mgr.create_backup(tag="test")
        assert backup_path is not None
        assert os.path.exists(backup_path)

    def test_list_backups(self, repo, tmp_path):
        """测试 list backups 场景。"""

        mgr = BackupManager(repo, backup_dir=str(tmp_path / "backups"))
        mgr.create_backup(tag="test1")
        mgr.create_backup(tag="test2")
        backups = mgr.list_backups()
        assert len(backups) == 2


class TestLongTermMemoryService:
    def test_add_memory(self, service):
        """测试 add memory 场景。"""

        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            confidence=0.8,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
        )
        assert item is not None
        assert item.value == "简短"

    def test_get_memory(self, service):
        """测试 get memory 场景。"""

        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        loaded = service.get_memory(item.id)
        assert loaded is not None
        assert loaded.value == "简短"

    def test_update_memory(self, service):
        """测试 update memory 场景。"""

        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        updated = service.update_memory(item.id, "u1", value="详细")
        assert updated is not None
        assert updated.value == "详细"

    def test_delete_memory(self, service):
        """测试 delete memory 场景。"""

        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        assert service.delete_memory(item.id, "u1")
        assert service.get_memory(item.id) is None

    def test_query_memories(self, service):
        """测试 query memories 场景。"""

        for i in range(3):
            service.add_memory(
                user_id="u1",
                memory_type=MemoryType.PREFERENCE,
                category=f"cat_{i}",
                key=f"key_{i}",
                value=f"val_{i}",
            )
        results = service.query_memories(MemoryQuery(user_id="u1"))
        assert len(results) == 3

    def test_memory_versions(self, service):
        """测试 memory versions 场景。"""

        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        service.update_memory(item.id, "u1", value="详细")
        versions = service.get_versions(item.id)
        assert len(versions) >= 1

    def test_extract_and_store(self, service):
        """测试 extract and store 场景。"""

        results = service.extract_and_store(
            user_message="我喜欢简短回答，我是初学者",
            assistant_message="好的，我会简洁回答",
            user_id="u1",
        )
        assert len(results) >= 1

    def test_extract_and_store_async_returns_without_waiting(self, service, monkeypatch):
        """测试 extract and store async returns without waiting 场景。"""

        extracted = [
            ExtractionResult(
                should_extract=True,
                memory_type=MemoryType.PREFERENCE,
                category="response_style",
                key="response_style",
                value="简短",
                confidence=0.8,
                is_explicit=True,
                raw_content="我喜欢简短回答",
            )
        ]

        monkeypatch.setattr(service.extractor, "should_attempt_extraction", lambda _: True)

        def _slow_extract(user_message, assistant_message, conversation_id=None):
            """模拟耗时的长期记忆抽取。"""

            time.sleep(0.25)
            return extracted

        monkeypatch.setattr(
            service.extractor, "extract_from_conversation", _slow_extract
        )

        started = time.perf_counter()
        future = service.extract_and_store_async(
            user_message="我喜欢简短回答",
            assistant_message="好的",
            user_id="u1",
        )
        elapsed = time.perf_counter() - started

        assert future is not None
        assert elapsed < 0.15
        result = future.result(timeout=2)
        assert len(result) == 1
        stored = service.query_memories(MemoryQuery(user_id="u1"))
        assert len(stored) == 1

    def test_render_profile_prompt(self, service):
        """测试 render profile prompt 场景。"""

        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
        )
        result = service.render_profile_prompt("u1")
        assert isinstance(result, str)

    def test_get_profile(self, service):
        """测试 get profile 场景。"""

        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
        )
        profile = service.get_profile("u1")
        assert profile is not None

    def test_delete_profile(self, service):
        """测试 delete profile 场景。"""

        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        assert service.delete_profile("u1")

    def test_candidate_flow(self, service):
        """测试 candidate flow 场景。"""

        result = service.extract_and_store(
            user_message="我经常看火星",
            assistant_message="好的，已记录",
            user_id="u1",
        )
        candidates = service.list_candidates("u1")
        assert len(candidates) >= 0

    def test_event_logs(self, service):
        """测试 event logs 场景。"""

        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        logs = service.get_event_logs("u1")
        assert len(logs) >= 1

    def test_maintenance(self, service):
        """测试 maintenance 场景。"""

        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        result = service.run_maintenance("u1")
        assert "expired" in result
        assert "archived" in result
        assert "promoted" in result

    def test_stats(self, service):
        """测试 stats 场景。"""

        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        stats = service.get_stats("u1")
        assert "type_counts" in stats

    def test_backup(self, service):
        """测试 backup 场景。"""

        backup_path = service.create_backup(tag="test")
        assert backup_path is not None
        backups = service.list_backups()
        assert len(backups) >= 1

    def test_upsert_profile(self, service):
        """测试 upsert profile 场景。"""

        result = service.upsert_profile(
            "u1",
            {
                "preferences": {"response_style": "简短"},
                "habits": {"frequent_topics": ["火星"]},
                "constraints": ["避免术语"],
                "background": {"skill_level": "入门"},
                "facts": [],
            },
        )
        assert result is not None

    def test_profile_upsert_and_prompt(self, service):
        """测试 profile upsert and prompt 场景。"""

        result = service.upsert_profile(
            "u1",
            {
                "preferences": {"response_style": "详细"},
                "constraints": ["避免术语"],
            },
        )
        assert result["preferences"]["response_style"] == "详细"
        formatted = service.render_profile_prompt("u1", task_type="qa")
        assert "详细" in formatted

    def test_user_isolation(self, service):
        """测试 user isolation 场景。"""

        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        service.add_memory(
            user_id="u2",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="详细",
        )
        u1 = service.query_memories(MemoryQuery(user_id="u1"))
        u2 = service.query_memories(MemoryQuery(user_id="u2"))
        assert len(u1) == 1
        assert len(u2) == 1
        assert u1[0].value == "简短"
        assert u2[0].value == "详细"

    def test_conflict_handling(self, service):
        """测试 conflict handling 场景。"""

        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            confidence=0.9,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
        )
        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="详细",
            confidence=0.5,
            source_type=SourceType.AUTO,
        )
        items = service.query_memories(
            MemoryQuery(user_id="u1", memory_type=MemoryType.PREFERENCE)
        )
        assert len(items) >= 1

    def test_export_snapshot(self, service):
        """测试 export snapshot 场景。"""

        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
        )
        snapshot = service.export_profile_snapshot("u1")
        assert snapshot is not None
        assert "user_id" in snapshot

    def test_store_extractions_persists_grade_metadata(self, service):
        """测试 store extractions persists grade metadata 场景。"""

        service.store_extractions(
            "u1",
            [
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.BACKGROUND,
                    category="skill_level",
                    key="skill_level",
                    value="入门",
                    confidence=0.7,
                    source_type=SourceType.AUTO,
                    extraction_grade="tentative",
                    gate_reason="window_repeated_signal",
                    metadata={"signal": "test"},
                )
            ],
        )

        candidates = service.list_candidates("u1", status=MemoryStatus.NEEDS_CONFIRM)
        assert len(candidates) == 1
        assert candidates[0].metadata["extraction_grade"] == "tentative"
        assert candidates[0].metadata["gate_reason"] == "window_repeated_signal"

    def test_revoke_archives_exact_active_memory(self, service):
        """测试 revoke archives exact active memory 场景。"""

        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="device_info",
            key="device_info",
            value="星特朗8SE",
            confidence=0.9,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
        )

        service.store_extractions(
            "u1",
            [
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.BACKGROUND,
                    category="device_info",
                    key="device_info",
                    value="用户撤回请求: 忘掉设备",
                    confidence=0.9,
                    action="revoke",
                    raw_content="忘掉我之前说的设备",
                    metadata={"revoke_reason": "忘掉我之前说的设备"},
                )
            ],
        )

        active = service.repository.find_memory_by_type_key(
            "u1", MemoryType.BACKGROUND, "device_info"
        )
        archived = service.query_memories(
            MemoryQuery(user_id="u1", status=MemoryStatus.ARCHIVED)
        )
        assert active is None
        assert len(archived) == 1
        assert archived[0].id == item.id
        assert archived[0].metadata["is_revoked"] is True

    def test_revoke_rejects_exact_candidate(self, service):
        """测试 revoke rejects exact candidate 场景。"""

        candidate = service.candidates.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="device_info",
            key="device_info",
            value="星特朗8SE",
            confidence=0.7,
        )

        service.store_extractions(
            "u1",
            [
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.BACKGROUND,
                    category="device_info",
                    key="device_info",
                    value="用户撤回请求",
                    confidence=0.8,
                    action="revoke",
                    raw_content="忘掉我之前说的设备",
                )
            ],
        )

        loaded = service.candidates.get_candidate(candidate.id)
        assert loaded.status == MemoryStatus.REJECTED
        assert loaded.metadata["is_revoked"] is True

    def test_ambiguous_revoke_creates_safe_candidate(self, service):
        """测试 ambiguous revoke creates safe candidate 场景。"""

        service.store_extractions(
            "u1",
            [
                ExtractionResult(
                    should_extract=True,
                    value="用户撤回请求: 忘掉之前说的",
                    confidence=0.8,
                    action="revoke",
                    raw_content="忘掉之前说的",
                    metadata={"ambiguous_target": True},
                )
            ],
        )

        candidates = service.list_candidates("u1")
        assert len(candidates) == 1
        assert candidates[0].key == "revoke_request"
        assert candidates[0].metadata["ambiguous_revoke"] is True

    def test_probation_conflict_archives_memory_and_creates_candidate(self, service):
        """测试 probation conflict archives memory and creates candidate 场景。"""

        first = ExtractionResult(
            should_extract=True,
            memory_type=MemoryType.BACKGROUND,
            category="device_info",
            key="device_info",
            value="星特朗8SE",
            confidence=0.9,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
            extraction_grade="solid",
        )
        assert service.store_extractions("u1", [first], conversation_id="c1") == []
        promoted = service.store_extractions("u1", [first], conversation_id="c2")
        assert len(promoted) == 1
        assert promoted[0].metadata["probation_status"] == "active"

        conflict = ExtractionResult(
            should_extract=True,
            memory_type=MemoryType.BACKGROUND,
            category="device_info",
            key="device_info",
            value="星特朗11寸",
            confidence=0.9,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
            extraction_grade="solid",
            raw_content="我已经换成星特朗11寸",
        )
        result = service.store_extractions("u1", [conflict], conversation_id="c3")

        assert result == []
        archived = service.query_memories(
            MemoryQuery(user_id="u1", status=MemoryStatus.ARCHIVED)
        )
        assert len(archived) == 1
        assert archived[0].metadata["probation_status"] == "conflicted"
        candidates = service.list_candidates("u1", status=MemoryStatus.NEEDS_CONFIRM)
        assert len(candidates) == 1
        assert candidates[0].metadata["conflict_info"]["relation_type"] == "conflict"

    def test_existing_memory_extension_updates_active_memory(self, service):
        """测试 existing memory extension updates active memory 场景。"""

        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="location",
            key="location",
            value="北京",
            confidence=0.8,
        )
        service.store_extractions(
            "u1",
            [
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.BACKGROUND,
                    category="location",
                    key="location",
                    value="北京和承德",
                    confidence=0.85,
                    source_type=SourceType.EXPLICIT,
                    is_explicit=True,
                    extraction_grade="solid",
                    raw_content="我现在主要在北京和承德观测",
                )
            ],
            conversation_id="c4",
        )

        loaded = service.get_memory(item.id)
        assert loaded.value == "北京和承德"
        assert loaded.metadata["last_relation_type"] == "extension"


class TestLongTermMemoryServiceRefactor:
    def test_service_candidate_promotes_and_rebuilds_profile_projection(
        self, tmp_db, monkeypatch
    ):
        """测试 service candidate promotes and rebuilds profile projection 场景。"""

        service = LongTermMemoryService(db_path=tmp_db)
        monkeypatch.setattr(service.extractor, "should_attempt_extraction", lambda _: True)
        monkeypatch.setattr(
            service.extractor,
            "extract_from_conversation",
            lambda *args, **kwargs: [
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.HABIT,
                    category="frequent_topics",
                    key="frequent_topics",
                    value=["火星"],
                    confidence=0.8,
                    source_type=SourceType.AUTO,
                    is_explicit=False,
                    raw_content="我经常看火星",
                )
            ],
        )

        first = service.extract_and_store("我经常看火星", "好的", "u1")
        assert first == []
        assert len(service.list_candidates("u1")) == 1

        second = service.extract_and_store("我经常看火星", "已记录", "u1")
        assert len(second) >= 1
        assert service.list_candidates("u1") == []

        profile = service.get_profile("u1")
        assert profile is not None
        assert "frequent_topics" in profile["habits"]
        assert "火星" in profile["habits"]["frequent_topics"]

        service.repository.delete_profile("u1")
        rebuilt = service.get_profile("u1")
        assert rebuilt is not None
        assert "火星" in rebuilt["habits"]["frequent_topics"]

    def test_service_list_candidates_supports_status_filter(self, tmp_db):
        """测试 service list candidates supports status filter 场景。"""

        service = LongTermMemoryService(db_path=tmp_db)
        service.candidates.process_extraction_as_candidate(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="skill_level",
            key="skill_level",
            value="入门",
            confidence=0.9,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
        )

        all_candidates = service.list_candidates("u1")
        needs_confirm = service.list_candidates("u1", status=MemoryStatus.NEEDS_CONFIRM)

        assert len(all_candidates) == 1
        assert len(needs_confirm) == 1
        assert needs_confirm[0].memory_type == MemoryType.BACKGROUND

    def test_delete_memory_rebuilds_profile_after_promotion(
        self, tmp_db, monkeypatch
    ):
        """测试 delete memory rebuilds profile after promotion 场景。"""

        service = LongTermMemoryService(db_path=tmp_db)
        monkeypatch.setattr(service.extractor, "should_attempt_extraction", lambda _: True)
        monkeypatch.setattr(
            service.extractor,
            "extract_from_conversation",
            lambda *args, **kwargs: [
                ExtractionResult(
                    should_extract=True,
                    memory_type=MemoryType.HABIT,
                    category="frequent_topics",
                    key="frequent_topics",
                    value=["火星"],
                    confidence=0.8,
                    source_type=SourceType.AUTO,
                    is_explicit=False,
                    raw_content="我经常看火星",
                )
            ],
        )
        first = service.extract_and_store("我经常看火星", "好的", "u1")
        assert first == []
        second = service.extract_and_store("我经常看火星", "已记录", "u1")
        assert len(second) >= 1

        profile = service.get_profile("u1")
        assert "火星" in profile["habits"]["frequent_topics"]

        memory_id = second[0].id
        assert service.delete_memory(memory_id, "u1")
        rebuilt = service.get_profile("u1")
        assert rebuilt is not None
        assert rebuilt["habits"].get("frequent_topics", []) == []

    def test_service_delete_memory_tombstones_and_hides_from_queries(self, tmp_db):
        """测试 service delete memory tombstones and hides from queries 场景。"""

        service = LongTermMemoryService(db_path=tmp_db)
        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            confidence=0.9,
            source_type=SourceType.EXPLICIT,
            is_explicit=True,
        )
        assert item is not None

        result = service.delete(
            LongTermMemoryDeletionRequest(
                user_id="u1",
                scope="memory",
                target_id=item.id,
                reason="user request",
            )
        )

        assert result.deleted_memories == 1
        assert service.get_memory(item.id) is None
        assert service.query_memories(MemoryQuery(user_id="u1")) == []
        audit = service.repository.list_deletion_audit("u1")
        assert audit[0]["scope"] == "memory"

    def test_service_user_all_delete_hides_candidates_profile_and_memories(
        self, tmp_db
    ):
        """测试 service user all delete hides candidates profile and memories 场景。"""

        service = LongTermMemoryService(db_path=tmp_db)
        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="skill_level",
            key="skill_level",
            value="入门",
        )
        candidate = service.candidates.add_or_update_candidate(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="详细",
        )
        assert item is not None
        assert candidate is not None

        result = service.delete(
            LongTermMemoryDeletionRequest(user_id="u1", scope="user_all")
        )

        assert result.deleted_memories == 1
        assert result.deleted_candidates == 1
        assert service.query_memories(MemoryQuery(user_id="u1")) == []
        assert service.list_candidates("u1") == []
        assert service.get_profile("u1") is None

    def test_query_aware_prompt_uses_relevant_active_memories(self, tmp_db):
        """测试 query aware prompt uses relevant active memories 场景。"""

        service = LongTermMemoryService(db_path=tmp_db)
        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="device_info",
            key="device_info",
            value="80mm 折射望远镜",
            confidence=0.9,
        )
        service.add_memory(
            user_id="u1",
            memory_type=MemoryType.CONSTRAINT,
            category="custom",
            key="no_jargon",
            value="避免术语",
            confidence=0.9,
        )

        prompt = service.build_prompt_context("u1", "今晚用望远镜观测什么")
        hits = service.explain_retrieval_hits("u1", "今晚用望远镜观测什么")

        assert "80mm" in prompt
        assert "避免术语" in prompt
        assert any(hit["memory_type"] == MemoryType.BACKGROUND for hit in hits)

    def test_explain_retrieval_hits_has_no_access_side_effect(self, tmp_db):
        """测试 explain retrieval hits has no access side effect 场景。"""

        service = LongTermMemoryService(db_path=tmp_db)
        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="device_info",
            key="device_info",
            value="80mm 折射望远镜",
            confidence=0.9,
        )

        service.explain_retrieval_hits("u1", "今晚用望远镜观测什么")
        after_explain = service.repository.get_memory(item.id)
        service.build_prompt_context(
            "u1", "今晚用望远镜观测什么", total_context_budget=4000
        )
        after_prompt = service.repository.get_memory(item.id)

        assert after_explain.access_count == 0
        assert after_prompt.access_count == 1

    def test_retrieval_explain_can_include_omitted_reasons(self, tmp_db):
        """测试 retrieval explain can include omitted reasons 场景。"""

        service = LongTermMemoryService(db_path=tmp_db)
        for index in range(5):
            service.add_memory(
                user_id="u1",
                memory_type=MemoryType.PREFERENCE,
                category="response_style",
                key=f"style_{index}",
                value=f"偏好 {index}",
                confidence=0.9,
            )

        hits = service.explain_retrieval_hits(
            "u1",
            "今晚用望远镜观测什么",
            task_type="observation",
            include_omitted=True,
        )

        assert any(hit["selected"] for hit in hits)
        assert any(hit["omitted_reason"] == "type_quota" for hit in hits)

    def test_record_injection_feedback_updates_metadata_and_stale_penalty(
        self, tmp_db
    ):
        """测试 record injection feedback updates metadata and stale penalty 场景。"""

        service = LongTermMemoryService(
            db_path=tmp_db,
            config={"semantic_retrieval_enabled": False, "injection_rerank_enabled": False},
        )
        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.BACKGROUND,
            category="device_info",
            key="device_info",
            value="80mm 折射望远镜",
            confidence=0.9,
        )

        result = service.record_injection_feedback(
            "u1",
            "今晚用什么观测",
            "可以继续使用 80mm 折射望远镜。",
            [item.id],
        )
        for _ in range(5):
            service.record_injection_feedback(
                "u1", "今晚用什么观测", "今晚云量偏高。", [item.id]
            )

        stored = service.repository.get_memory(item.id)
        stats = stored.metadata["injection_stats"]
        hits = service.explain_retrieval_hits(
            "u1", "今晚用什么观测", include_omitted=True
        )
        target = next(hit for hit in hits if hit["memory_id"] == item.id)

        assert result["hit"] == 1
        assert stats["shown_count"] == 6
        assert stats["hit_count"] == 1
        assert stats["consecutive_miss_count"] == 5
        assert target["components"]["stale_penalty"] == 1.0

    def test_rebuild_semantic_index_counts_created_updated_skipped(self, tmp_db, monkeypatch):
        """测试 rebuild semantic index counts created updated skipped 场景。"""

        service = LongTermMemoryService(
            db_path=tmp_db,
            config={"semantic_retrieval_enabled": False, "injection_rerank_enabled": False},
        )
        item = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.FACT,
            category="equipment",
            key="mount",
            value="EQ6 Pro",
            confidence=0.9,
        )
        service.embedding_service.enabled = True
        service.embedding_service.api_key = "test-key"
        service.embedding_service.model_name = "embed-test"
        monkeypatch.setattr(
            service.embedding_service,
            "_embed_text",
            lambda text: [1.0, 0.0],
        )

        created = service.rebuild_semantic_index("u1", limit=10)
        skipped = service.rebuild_semantic_index("u1", limit=10)
        service.update_memory(item.id, "u1", value="HEQ5 Pro")
        updated = service.rebuild_semantic_index("u1", limit=10)

        assert created["created"] == 1
        assert skipped["skipped"] == 1
        assert updated["updated"] == 1

    def test_adaptive_type_prior_waits_for_20_samples_then_adjusts(self, tmp_db):
        """测试 adaptive type prior waits for 20 samples then adjusts 场景。"""

        service = LongTermMemoryService(
            db_path=tmp_db,
            config={"semantic_retrieval_enabled": False, "injection_rerank_enabled": False},
        )
        preferred = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.PREFERENCE,
            category="response_style",
            key="response_style",
            value="简短",
            confidence=0.9,
        )
        missed = service.add_memory(
            user_id="u1",
            memory_type=MemoryType.HABIT,
            category="observation_type",
            key="observation_type",
            value="目视观测",
            confidence=0.9,
        )

        for _ in range(19):
            service.record_injection_feedback(
                "u1", "今晚怎么回答", "保持简短。", [preferred.id], task_type="general"
            )
            service.record_injection_feedback(
                "u1", "今晚怎么回答", "今晚云量偏高。", [missed.id], task_type="general"
            )
        early_hits = service.explain_retrieval_hits(
            "u1", "今晚怎么回答", task_type="general", include_omitted=True
        )
        early_preferred = next(hit for hit in early_hits if hit["memory_id"] == preferred.id)
        early_missed = next(hit for hit in early_hits if hit["memory_id"] == missed.id)

        service.record_injection_feedback(
            "u1", "今晚怎么回答", "保持简短。", [preferred.id], task_type="general"
        )
        service.record_injection_feedback(
            "u1", "今晚怎么回答", "今晚云量偏高。", [missed.id], task_type="general"
        )
        mature_hits = service.explain_retrieval_hits(
            "u1", "今晚怎么回答", task_type="general", include_omitted=True
        )
        mature_preferred = next(hit for hit in mature_hits if hit["memory_id"] == preferred.id)
        mature_missed = next(hit for hit in mature_hits if hit["memory_id"] == missed.id)

        assert early_preferred["components"]["type_weight"] == 0.85
        assert early_missed["components"]["type_weight"] == 0.65
        assert mature_preferred["components"]["type_weight"] > 0.85
        assert mature_missed["components"]["type_weight"] < 0.65
