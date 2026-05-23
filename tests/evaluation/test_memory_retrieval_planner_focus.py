"""短期记忆检索规划器的策略级回归测试。

覆盖 select_strategy 中工具证据、焦点提取、MMR 去重、动态预算和
summary/task_state 装配等短期上下文选择策略。
"""

import json
import math
import time

from src.memory.core.models import Message, SalientFact, ToolCallRecord
from src.memory.domain.summary_snapshot import SummarySnapshot
from src.memory.domain.task_state import TaskState
from src.memory.retrieval.planner import ContextScene, RetrievalFocus, RetrievalPlanner


def _planner() -> RetrievalPlanner:
    """构造使用字符长度估算的检索规划器。"""

    return RetrievalPlanner(lambda text: len(text or ""))


def test_extract_tool_meta_parses_weather_location_and_freshness():
    """测试 extract tool meta parses weather location and freshness 场景。"""

    call = ToolCallRecord(
        tool_call_id="tool_weather",
        tool_name="weather-lookup",
        timestamp=10.0,
        input_summary='{"city":"北京","time":"22:00"}',
        output_summary=(
            'json fields=city,status; city=北京; status=晴; '
            'fresh_marker=new_weather_good; updated_at="22:00 更新"'
        ),
    )

    meta = _planner()._extract_tool_meta(call)

    assert meta.tool_call_id == "tool_weather"
    assert meta.tool_type == "weather"
    assert meta.locations == {"北京"}
    assert meta.targets == set()
    assert meta.is_fresh_marked is True
    assert meta.is_stale_marked is False
    assert meta.timestamp == 10.0


def test_extract_tool_meta_parses_photo_target_location_and_stale_marker():
    """测试 extract tool meta parses photo target location and stale marker 场景。"""

    call = ToolCallRecord(
        tool_call_id="tool_photo",
        tool_name="astrophotography-calculator",
        timestamp=20.0,
        input_summary='{"target":"M42","location":"上海","version":"old"}',
        output_summary=(
            "version=旧参数; target=M42; location=上海; "
            "iso=ISO 1600; stale_marker=old_exposure_bad"
        ),
    )

    meta = _planner()._extract_tool_meta(call)

    assert meta.tool_type == "photo"
    assert meta.locations == {"上海"}
    assert meta.targets == {"M42"}
    assert meta.is_fresh_marked is False
    assert meta.is_stale_marked is True


def test_extract_tool_meta_parses_event_aliases():
    """测试 extract tool meta parses event aliases 场景。"""

    call = ToolCallRecord(
        tool_call_id="tool_event",
        tool_name="sky-event-calendar",
        timestamp=30.0,
        input_summary='{"event":"Geminids"}',
        output_summary=(
            "event=双子座流星雨; best_time=12 月 14 日; "
            "不要混入英仙座流星雨"
        ),
    )

    meta = _planner()._extract_tool_meta(call)

    assert meta.tool_type == "event"
    assert "双子座流星雨" in meta.targets
    assert "英仙座流星雨" in meta.targets


def test_tool_metadata_catalog_is_permanent_and_latest_text_is_not_fresh_marker():
    """测试 tool metadata catalog is permanent and latest text is not fresh marker 场景。"""

    planner = _planner()
    catalog_call = ToolCallRecord(
        tool_call_id="catalog",
        tool_name="simbad-catalog",
        timestamp=1.0,
        input_summary='{"target":"M42"}',
        output_summary="catalog data",
        metadata={"tool_type": "catalog", "produced_at": 1.0, "effective_until": 0},
    )
    latest_text_call = ToolCallRecord(
        tool_call_id="latest_text",
        tool_name="weather-lookup",
        timestamp=1.0,
        input_summary='{"city":"北京"}',
        output_summary="latest weather text without structured freshness metadata",
    )

    catalog_meta = planner._extract_tool_meta(catalog_call)
    latest_text_meta = planner._extract_tool_meta(latest_text_call)

    assert catalog_meta.effective_until == 0
    assert planner._tool_is_expired(catalog_meta, reference_time=10_000_000.0) is False
    assert latest_text_meta.is_fresh_marked is False


def test_fresh_score_uses_exponential_decay():
    """测试 fresh score uses exponential decay 场景。"""

    planner = _planner()
    produced_at = 10_000.0
    meta = planner._extract_tool_meta(
        ToolCallRecord(
            tool_call_id="weather",
            tool_name="weather-lookup",
            timestamp=produced_at,
            input_summary='{"city":"北京"}',
            output_summary="city=北京",
            metadata={"produced_at": produced_at, "effective_until": produced_at + 3600},
        )
    )

    score = planner._tool_fresh_score(
        meta,
        reference_time=produced_at + planner._tool_tau_seconds("weather"),
    )

    assert math.isclose(score, math.exp(-1), rel_tol=1e-6)


def test_explicit_supersedes_metadata_derives_superseded_by():
    """测试 explicit supersedes metadata derives superseded by 场景。"""

    planner = _planner()
    old_call = ToolCallRecord(
        tool_call_id="old_weather",
        tool_name="weather-lookup",
        timestamp=1.0,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; cloud=云量 50%",
    )
    new_call = ToolCallRecord(
        tool_call_id="new_weather",
        tool_name="weather-lookup",
        timestamp=2.0,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; cloud=云量 8%",
        metadata={"supersedes_tool_call_ids": ["old_weather"]},
    )

    metas = planner._derive_tool_evidence_metas(
        [old_call, new_call],
        query="北京天气",
        focus=RetrievalFocus({"北京"}, set(), {"weather"}, "latest"),
    )

    assert metas["old_weather"].superseded_by == "new_weather"
    assert metas["new_weather"].supersedes_tool_call_ids == ["old_weather"]


def test_extract_focus_uses_query_and_task_state_without_negative_entities():
    """测试 extract focus uses query and task state without negative entities 场景。"""

    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="比较北京和上海观测条件",
        active_constraints=["当前追问优先北京", "不要混入上海"],
        next_action="回答北京结果",
    )

    focus = _planner()._extract_focus("北京那个结果呢？", state)

    assert focus.locations == {"北京"}
    assert focus.targets == set()
    assert focus.preferred_tool_types == set()
    assert focus.freshness_intent == "neutral"


def test_extract_focus_detects_target_tool_type_and_latest_intent():
    """测试 extract focus detects target tool type and latest intent 场景。"""

    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="按北京新位置判断 M42 是否适合",
        active_constraints=["旧上海结果冲突", "只问 M42", "新地点优先"],
        next_action="回答 M42 是否适合",
    )

    focus = _planner()._extract_focus("M42 那个还适合吗？", state)

    assert focus.locations == {"北京"}
    assert focus.targets == {"M42"}
    assert focus.preferred_tool_types == {"position"}
    assert focus.freshness_intent == "latest"


def test_extract_focus_detects_photo_and_event_intents():
    """测试 extract focus detects photo and event intents 场景。"""

    photo_state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="确定 M42 最新摄影参数",
        active_constraints=["新旧参数冲突", "不要混入 M31"],
        next_action="用新参数解释拍摄影响",
    )
    event_state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="回答双子座流星雨最佳观测时间",
        active_constraints=["不要混入英仙座", "最近一次事件优先"],
        next_action="给出 12 月 14 日窗口",
    )

    photo_focus = _planner()._extract_focus("参数要按哪个？", photo_state)
    event_focus = _planner()._extract_focus("刚才那个什么时候看？", event_state)

    assert photo_focus.preferred_tool_types == {"photo"}
    assert photo_focus.freshness_intent == "latest"
    assert event_focus.targets == {"双子座流星雨"}
    assert event_focus.preferred_tool_types == {"event"}
    assert event_focus.freshness_intent == "latest"


def test_rank_tools_with_focus_filters_wrong_location_and_tool_type():
    """测试 rank tools with focus filters wrong location and tool type 场景。"""

    planner = _planner()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="判断北京今晚是否还能目视观测",
        active_constraints=["需要排除上海天气", "目标是月球和亮星"],
        next_action="基于北京最新天气给出结论",
    )
    calls = [
        ToolCallRecord(
            tool_call_id="wrong_city",
            tool_name="weather-lookup",
            timestamp=1.0,
            input_summary='{"city":"上海"}',
            output_summary="city=上海; cloud=云量 72%; suitability=不适合",
        ),
        ToolCallRecord(
            tool_call_id="wrong_type",
            tool_name="celestial-position",
            timestamp=2.0,
            input_summary='{"object":"M42","location":"北京"}',
            output_summary="object=M42; location=北京; altitude=38 度",
        ),
        ToolCallRecord(
            tool_call_id="right_weather",
            tool_name="weather-lookup",
            timestamp=3.0,
            input_summary='{"city":"北京"}',
            output_summary="city=北京; status=晴; cloud=云量 8%; suitability=适合观测",
        ),
    ]

    ranked = planner._rank_tools_with_focus("那这种情况还能看吗？", state, calls)

    assert [call.tool_call_id for call in ranked] == ["right_weather"]


def test_rank_tools_with_focus_falls_back_when_all_scores_zero():
    """测试 rank tools with focus falls back when all scores zero 场景。"""

    planner = _planner()
    state = TaskState(tenant_id="tenant", session_id="session")
    calls = [
        ToolCallRecord(
            tool_call_id="old_rank_top",
            tool_name="weather-lookup",
            timestamp=2.0,
            input_summary="",
            output_summary="",
        ),
        ToolCallRecord(
            tool_call_id="old_rank_second",
            tool_name="weather-lookup",
            timestamp=1.0,
            input_summary="",
            output_summary="",
        ),
    ]

    ranked = planner._rank_tools_with_focus("", state, calls)

    assert [call.tool_call_id for call in ranked] == ["old_rank_top"]


def test_dedupe_superseded_tools_keeps_fresh_latest_result():
    """测试 dedupe superseded tools keeps fresh latest result 场景。"""

    planner = _planner()
    focus = planner._extract_focus(
        "现在还适合吗？",
        TaskState(
            tenant_id="tenant",
            session_id="session",
            current_goal="使用北京 22:00 最新天气判断观测",
            active_constraints=["新结果优先", "旧结果不应作为主要证据"],
            next_action="回答现在是否适合",
        ),
    )
    old_call = ToolCallRecord(
        tool_call_id="old_weather",
        tool_name="weather-lookup",
        timestamp=1.0,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; updated_at=18:00 旧结果; stale_marker=old_weather_bad",
    )
    new_call = ToolCallRecord(
        tool_call_id="new_weather",
        tool_name="weather-lookup",
        timestamp=2.0,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; updated_at=22:00 更新; fresh_marker=new_weather_good",
    )

    deduped = planner._dedupe_superseded_tools([old_call, new_call], focus)

    assert [call.tool_call_id for call in deduped] == ["new_weather"]


def test_dedupe_superseded_tools_allows_compare_intent():
    """测试 dedupe superseded tools allows compare intent 场景。"""

    planner = _planner()
    focus = planner._extract_focus(
        "前后为什么不一样？",
        TaskState(tenant_id="tenant", session_id="session"),
    )
    old_call = ToolCallRecord(
        tool_call_id="old_weather",
        tool_name="weather-lookup",
        timestamp=1.0,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; stale_marker=old_weather_bad",
    )
    new_call = ToolCallRecord(
        tool_call_id="new_weather",
        tool_name="weather-lookup",
        timestamp=2.0,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; fresh_marker=new_weather_good",
    )

    deduped = planner._dedupe_superseded_tools([old_call, new_call], focus)

    assert [call.tool_call_id for call in deduped] == ["old_weather", "new_weather"]


def test_rank_tools_with_focus_downranks_expired_latest_evidence():
    """测试 rank tools with focus downranks expired latest evidence 场景。"""

    planner = _planner()
    now = time.time()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="按北京最新天气判断观测",
        next_action="引用北京天气",
    )
    expired = ToolCallRecord(
        tool_call_id="expired_weather",
        tool_name="weather-lookup",
        timestamp=now,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; cloud=云量 60%",
        metadata={
            "params_hash": "same-weather",
            "produced_at": now,
            "effective_until": now - 10,
        },
    )
    valid = ToolCallRecord(
        tool_call_id="valid_weather",
        tool_name="weather-lookup",
        timestamp=now - 100,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; cloud=云量 8%",
        metadata={
            "params_hash": "same-weather",
            "produced_at": now - 100,
            "effective_until": now + 300,
        },
    )

    ranked = planner._rank_tools_with_focus("北京天气现在还能观测吗？", state, [expired, valid])

    assert [call.tool_call_id for call in ranked] == ["valid_weather"]


def test_rank_tools_with_focus_compare_intent_keeps_expired_and_current_evidence():
    """测试 rank tools with focus compare intent keeps expired and current evidence 场景。"""

    planner = _planner()
    now = time.time()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="比较北京天气前后为什么不一样",
        next_action="解释前后差异",
    )
    expired = ToolCallRecord(
        tool_call_id="expired_weather",
        tool_name="weather-lookup",
        timestamp=now - 100,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; cloud=云量 60%",
        metadata={
            "params_hash": "same-weather",
            "produced_at": now - 100,
            "effective_until": now - 10,
        },
    )
    current = ToolCallRecord(
        tool_call_id="current_weather",
        tool_name="weather-lookup",
        timestamp=now,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; cloud=云量 8%",
        metadata={
            "params_hash": "same-weather",
            "produced_at": now,
            "effective_until": now + 300,
        },
    )

    ranked = planner._rank_tools_with_focus("前后为什么不一样？", state, [expired, current])

    assert {call.tool_call_id for call in ranked} == {
        "expired_weather",
        "current_weather",
    }


def test_dedupe_superseded_tools_keeps_latest_success_and_latest_error_per_chain():
    """测试 dedupe superseded tools keeps latest success and latest error per chain 场景。"""

    planner = _planner()
    focus = planner._extract_focus(
        "现在按最新结果判断",
        TaskState(tenant_id="tenant", session_id="session", current_goal="北京最新天气"),
    )
    calls = [
        ToolCallRecord(
            tool_call_id="success_old",
            tool_name="weather-lookup",
            timestamp=1.0,
            input_summary='{"city":"北京"}',
            output_summary="city=北京; cloud=云量 30%",
        ),
        ToolCallRecord(
            tool_call_id="error_old",
            tool_name="weather-lookup",
            timestamp=2.0,
            input_summary='{"city":"北京"}',
            output_summary="timeout",
            status="error",
        ),
        ToolCallRecord(
            tool_call_id="success_new",
            tool_name="weather-lookup",
            timestamp=3.0,
            input_summary='{"city":"北京"}',
            output_summary="city=北京; cloud=云量 8%",
        ),
        ToolCallRecord(
            tool_call_id="error_new",
            tool_name="weather-lookup",
            timestamp=4.0,
            input_summary='{"city":"北京"}',
            output_summary="rate limited",
            status="error",
        ),
    ]

    deduped = planner._dedupe_superseded_tools(calls, focus)

    assert [call.tool_call_id for call in deduped] == ["success_new", "error_new"]


def test_diverse_tool_selection_caps_tool_type_and_target():
    """测试 diverse tool selection caps tool type and target 场景。"""

    planner = _planner()
    focus = RetrievalFocus(
        locations={"北京", "上海", "广州"},
        targets={"M42"},
        preferred_tool_types=set(),
        freshness_intent="neutral",
    )
    calls = [
        ToolCallRecord(
            tool_call_id="weather_beijing",
            tool_name="weather-lookup",
            timestamp=3.0,
            input_summary='{"city":"北京","target":"M42"}',
            output_summary="city=北京; target=M42",
            metadata={"tool_score": 0.9},
        ),
        ToolCallRecord(
            tool_call_id="weather_shanghai",
            tool_name="weather-lookup",
            timestamp=2.0,
            input_summary='{"city":"上海","target":"M42"}',
            output_summary="city=上海; target=M42",
            metadata={"tool_score": 0.8},
        ),
        ToolCallRecord(
            tool_call_id="weather_guangzhou",
            tool_name="weather-lookup",
            timestamp=1.0,
            input_summary='{"city":"广州","target":"M42"}',
            output_summary="city=广州; target=M42",
            metadata={"tool_score": 0.7},
        ),
        ToolCallRecord(
            tool_call_id="position_m42",
            tool_name="celestial-position",
            timestamp=4.0,
            input_summary='{"object":"M42","location":"北京"}',
            output_summary="object=M42; location=北京",
            metadata={"tool_score": 0.6},
        ),
    ]

    selected = planner._select_diverse_tool_evidence(
        calls,
        max_tools=5,
        max_per_tool_type=2,
        max_per_target=2,
        focus=focus,
    )

    weather_count = sum(
        1 for call in selected if planner._extract_tool_meta(call).tool_type == "weather"
    )
    m42_count = sum(
        1 for call in selected if "M42" in planner._extract_tool_meta(call).targets
    )
    assert weather_count <= 2
    assert m42_count <= 2


def test_non_latest_intent_injects_contrast_but_latest_does_not_force_it():
    """测试 non latest intent injects contrast but latest does not force it 场景。"""

    planner = _planner()
    fresh_one = ToolCallRecord(
        tool_call_id="fresh_one",
        tool_name="weather-lookup",
        timestamp=3.0,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; cloud=8%",
        metadata={"tool_score": 0.9},
    )
    fresh_two = ToolCallRecord(
        tool_call_id="fresh_two",
        tool_name="celestial-position",
        timestamp=2.0,
        input_summary='{"object":"M42","location":"北京"}',
        output_summary="object=M42; location=北京",
        metadata={"tool_score": 0.8},
    )
    expired = ToolCallRecord(
        tool_call_id="expired_weather",
        tool_name="weather-lookup",
        timestamp=1.0,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; cloud=70%",
        metadata={"tool_score": 0.5, "expired": True},
    )

    neutral = planner._select_diverse_tool_evidence(
        [fresh_one, fresh_two, expired],
        max_tools=2,
        max_per_tool_type=2,
        max_per_target=2,
        focus=RetrievalFocus({"北京"}, {"M42"}, set(), "neutral"),
    )
    latest = planner._select_diverse_tool_evidence(
        [fresh_one, fresh_two, expired],
        max_tools=2,
        max_per_tool_type=2,
        max_per_target=2,
        focus=RetrievalFocus({"北京"}, {"M42"}, set(), "latest"),
    )

    assert "expired_weather" in [call.tool_call_id for call in neutral]
    assert "expired_weather" not in [call.tool_call_id for call in latest]


def test_scene_weights_prioritize_freshness_or_target_by_scene():
    """测试 scene weights prioritize freshness or target by scene 场景。"""

    planner = _planner()
    now = time.time()
    focus = RetrievalFocus(set(), {"M42"}, set(), "neutral")
    target_call = ToolCallRecord(
        tool_call_id="old_target",
        tool_name="celestial-position",
        timestamp=now - 100_000,
        input_summary='{"object":"M42"}',
        output_summary="object=M42; altitude=40",
        metadata={"produced_at": now - 100_000, "effective_until": now - 10},
    )
    fresh_call = ToolCallRecord(
        tool_call_id="fresh_weather",
        tool_name="weather-lookup",
        timestamp=now,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; cloud=8%",
        metadata={"produced_at": now, "effective_until": now + 3600},
    )
    target_meta = planner._derive_tool_evidence_metas(
        [target_call],
        query="M42 参数",
        focus=focus,
    )["old_target"]
    fresh_meta = planner._derive_tool_evidence_metas(
        [fresh_call],
        query="M42 参数",
        focus=focus,
    )["fresh_weather"]

    computation_target = planner._score_tool_with_focus(
        "M42 参数", focus, target_call, target_meta, ContextScene.COMPUTATION.value
    )
    computation_fresh = planner._score_tool_with_focus(
        "M42 参数", focus, fresh_call, fresh_meta, ContextScene.COMPUTATION.value
    )
    observation_target = planner._score_tool_with_focus(
        "M42 参数", focus, target_call, target_meta, ContextScene.OBSERVATION.value
    )
    observation_fresh = planner._score_tool_with_focus(
        "M42 参数", focus, fresh_call, fresh_meta, ContextScene.OBSERVATION.value
    )

    assert computation_target > computation_fresh
    assert observation_fresh > observation_target


def test_debugging_scene_boosts_representative_error_signal():
    """测试 debugging scene boosts representative error signal 场景。"""

    planner = _planner()
    focus = RetrievalFocus({"北京"}, set(), {"weather"}, "neutral")
    success_call = ToolCallRecord(
        tool_call_id="success",
        tool_name="weather-lookup",
        timestamp=1.0,
        input_summary='{"city":"北京"}',
        output_summary="city=北京; cloud=8%",
    )
    error_call = ToolCallRecord(
        tool_call_id="error",
        tool_name="weather-lookup",
        timestamp=2.0,
        input_summary='{"city":"北京"}',
        output_summary="rate limited",
        status="error",
    )
    metas = planner._derive_tool_evidence_metas(
        [success_call, error_call],
        query="北京天气失败怎么处理",
        focus=focus,
    )

    success_score = planner._score_tool_with_focus(
        "北京天气失败怎么处理",
        focus,
        success_call,
        metas["success"],
        ContextScene.DEBUGGING.value,
    )
    error_score = planner._score_tool_with_focus(
        "北京天气失败怎么处理",
        focus,
        error_call,
        metas["error"],
        ContextScene.DEBUGGING.value,
    )

    assert metas["error"].error_signal == 1.0
    assert error_score > success_score


def test_rank_messages_with_focus_drops_zero_query_score_outside_recent_four():
    """测试 rank messages with focus drops zero query score outside recent four 场景。"""

    planner = _planner()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="回答北京 M42 观测建议",
        active_constraints=["只关注 M42"],
        next_action="基于 M42 高度回答",
    )
    messages = [
        Message(
            message_id=f"msg_{index}",
            role="user",
            content=content,
            timestamp=float(index),
        )
        for index, content in enumerate(
            [
                "很早之前聊过 M31 上海。",
                "土星冲日另说。",
                "近地小行星先放一放。",
                "现在只说 M42。",
                "M42 在北京。",
                "高度足够就用 80mm。",
                "查完一句话建议。",
            ],
            start=1,
        )
    ]

    ranked = planner._rank_messages_with_focus("建议还成立吗？", state, messages)

    assert "很早之前聊过 M31 上海。" not in [message.content for message in ranked]
    assert {message.message_id for message in ranked} <= {
        "msg_4",
        "msg_5",
        "msg_6",
        "msg_7",
    }


def test_rank_messages_with_focus_drops_task_state_covered_noise():
    """测试 rank messages with focus drops task state covered noise 场景。"""

    planner = _planner()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="回答 M42 北京拍摄是否受影响",
        active_constraints=["不要混入 M31 上海", "需要天气和曝光证据"],
        next_action="解释 M42 拍摄影响",
    )
    messages = [
        Message(
            message_id="msg_noise",
            role="user",
            content="不要把 M31 参数说成 M42。",
            timestamp=1.0,
        ),
        Message(
            message_id="msg_focus",
            role="assistant",
            content="会引用 M42 北京证据。",
            timestamp=2.0,
        ),
    ]

    ranked = planner._rank_messages_with_focus("M42 那个会不会影响拍摄？", state, messages)

    assert [message.message_id for message in ranked] == ["msg_focus"]


def test_message_candidates_recall_old_relevant_message_through_focus():
    """测试 message candidates recall old relevant message through focus 场景。"""

    planner = _planner()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="回答北京 M42 观测建议",
        active_constraints=["只关注 M42"],
    )
    focus = planner._extract_focus("建议还成立吗？", state)
    messages = [
        Message(
            message_id="old_focus",
            role="assistant",
            content="M42 在北京高度足够，可以继续用 80mm。",
            timestamp=1.0,
        ),
        *[
            Message(
                message_id=f"recent_{index}",
                role="user",
                content=f"无关近消息 {index}",
                timestamp=float(index + 2),
            )
            for index in range(5)
        ],
    ]

    candidates = planner._message_candidates("建议还成立吗？", state, messages, focus)
    old_candidate = next(
        candidate for candidate in candidates if candidate.candidate_id == "old_focus"
    )

    assert "focus" in old_candidate.metadata["recall_sources"]


def test_build_context_mmr_limits_near_duplicate_facts():
    """测试 build context mmr limits near duplicate facts 场景。"""

    planner = _planner()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="回答北京 M42 观测建议",
        active_constraints=["只关注 M42"],
    )
    facts = [
        SalientFact(
            fact_id=f"dup_{index}",
            fact_type="observation",
            content="北京 M42 云量 8%，适合继续观测。",
            timestamp=float(index),
        )
        for index in range(10)
    ]

    context = planner.build_context(
        query="北京 M42 观测建议还成立吗？",
        token_budget=1000,
        task_state=state,
        summary_snapshot=None,
        messages=[],
        facts=facts,
        tool_calls=[],
    )

    assert len(context["selected_salient_facts"]) <= 2


def test_build_context_records_scene_budgets_and_rendered_selected_ids_only():
    """测试 build context records scene budgets and rendered selected ids only 场景。"""

    planner = _planner()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="判断北京今晚天气是否适合观测",
        active_constraints=["优先北京最新结果"],
        next_action="引用北京天气证据回答",
        version=4,
    )
    summary = SummarySnapshot(
        tenant_id="tenant",
        session_id="session",
        summary_text="之前讨论过上海旧天气和北京观测目标。" * 8,
        created_at=1.0,
    )
    facts = [
        SalientFact(
            fact_id="fact_beijing",
            fact_type="observation",
            content="北京观测时要重点看云量。",
            timestamp=2.0,
        ),
        SalientFact(
            fact_id="fact_shanghai",
            fact_type="observation",
            content="上海旧天气不作为当前依据。",
            timestamp=1.0,
        ),
    ]
    tool_calls = [
        ToolCallRecord(
            tool_call_id="tool_beijing",
            tool_name="weather-lookup",
            timestamp=3.0,
            input_summary='{"city":"北京"}',
            output_summary="city=北京; cloud=云量 8%; fresh_marker=new_weather_good",
        ),
        ToolCallRecord(
            tool_call_id="tool_shanghai",
            tool_name="weather-lookup",
            timestamp=2.0,
            input_summary='{"city":"上海"}',
            output_summary="city=上海; cloud=云量 72%; stale_marker=old_weather_bad",
        ),
    ]
    messages = [
        Message(
            message_id=f"msg_{index}",
            role="user",
            content=content,
            timestamp=float(index),
        )
        for index, content in enumerate(
            [
                "很早之前查了上海。",
                "现在只看北京天气。",
                "北京云量如果低就适合。",
                "最后请给一句话结论。",
            ],
            start=1,
        )
    ]

    context = planner.build_context(
        query="北京天气现在还能观测吗？",
        token_budget=260,
        task_state=state,
        summary_snapshot=summary,
        messages=messages,
        facts=facts,
        tool_calls=tool_calls,
    )

    plan = context["retrieval_plan"]
    assert plan["context_scene"] == "observation"
    assert plan["section_budgets"]["tools"] > plan["section_budgets"]["facts"]
    assert plan["selected_task_state_version"] == 4
    assert plan["selected_message_ids"] == [
        message["message_id"] for message in context["selected_recent_messages"]
    ]
    assert plan["selected_fact_ids"] == [
        fact["fact_id"] for fact in context["selected_salient_facts"]
    ]
    assert plan["selected_tool_call_ids"] == [
        call["tool_call_id"] for call in context["selected_tool_calls"]
    ]
    for message in context["selected_recent_messages"]:
        assert message["content"] in context["context_text"]
    for fact in context["selected_salient_facts"]:
        assert fact["content"] in context["context_text"]
    for call in context["selected_tool_calls"]:
        assert call["output_summary"] in context["context_text"]


def test_build_context_selected_tool_calls_include_selection_debug_metadata():
    """测试 build context selected tool calls include selection debug metadata 场景。"""

    planner = _planner()
    now = time.time()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="判断北京最新天气是否适合观测",
        next_action="引用北京天气",
    )
    tool_calls = [
        ToolCallRecord(
            tool_call_id="old_weather",
            tool_name="weather-lookup",
            timestamp=now - 100,
            input_summary='{"city":"北京"}',
            output_summary="city=北京; cloud=云量 70%",
            metadata={
                "params_hash": "same-weather",
                "produced_at": now - 100,
                "effective_until": now - 10,
            },
        ),
        ToolCallRecord(
            tool_call_id="new_weather",
            tool_name="weather-lookup",
            timestamp=now,
            input_summary='{"city":"北京"}',
            output_summary="city=北京; cloud=云量 8%",
            metadata={
                "params_hash": "same-weather",
                "produced_at": now,
                "effective_until": now + 300,
            },
        ),
    ]

    context = planner.build_context(
        query="北京天气现在还能观测吗？",
        token_budget=1200,
        task_state=state,
        summary_snapshot=None,
        messages=[],
        facts=[],
        tool_calls=tool_calls,
    )

    selected = context["selected_tool_calls"][0]
    metadata = selected["metadata"]
    assert selected["tool_call_id"] == "new_weather"
    assert "fresh_score" in metadata
    assert "expired" in metadata
    assert "superseded_by" in metadata
    assert "query_relevance" in metadata
    assert "tool_score" in metadata
    assert "selection_reason" in metadata


def test_build_context_renders_relevant_structured_summary_fields():
    """测试 build context renders relevant structured summary fields 场景。"""

    planner = _planner()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="回答北京 M42 观测建议",
    )
    summary = SummarySnapshot(
        tenant_id="tenant",
        session_id="session",
        summary_level="l2",
        summary_text=json.dumps(
            {
                "topics": ["北京", "M42", "上海"],
                "decisions": [
                    "assistant: 结论：北京 M42 适合观测。",
                    "assistant: 上海 M31 暂不处理。",
                ],
                "open_questions": ["北京 M42 是否还适合？"],
                "established_facts": [
                    "tool weather-lookup: 北京云量 8%",
                    "tool weather-lookup: 上海云量 70%",
                ],
                "tool_results_index": [
                    {
                        "tool": "weather-lookup",
                        "tool_type": "weather",
                        "params_hash": "beijing",
                        "status": "success",
                        "key_finding": "北京云量 8%",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        created_at=1.0,
    )

    context = planner.build_context(
        query="北京 M42 现在还适合观测吗？",
        token_budget=3000,
        task_state=state,
        summary_snapshot=summary,
        messages=[],
        facts=[],
        tool_calls=[],
    )

    assert "open_questions:" in context["context_text"]
    assert "北京 M42 适合观测" in context["context_text"]
    assert "weather-lookup" in context["context_text"]
    assert '{"topics"' not in context["context_text"]


def test_build_context_keeps_task_state_pinned_with_tiny_budget():
    """测试 build context keeps task state pinned with tiny budget 场景。"""

    planner = _planner()
    state = TaskState(
        tenant_id="tenant",
        session_id="session",
        current_goal="完成北京 M42 观测判断",
        active_constraints=["不要混入上海旧结果", "保留最新工具证据"],
        next_action="给出结论",
        version=3,
    )

    context = planner.build_context(
        query="结论是什么？",
        token_budget=90,
        task_state=state,
        summary_snapshot=None,
        messages=[
            Message(role="user", content="北京 M42 还能看吗？", timestamp=1.0),
        ],
        facts=[],
        tool_calls=[],
    )

    assert "=== task state ===" in context["context_text"]
    assert "current_goal" in context["context_text"]
    assert context["retrieval_plan"]["selected_task_state_version"] == 3
    assert "compact_task_state" in context["retrieval_plan"]["downgrade_steps"]


def test_derive_focus_stack_boosts_stable_recent_focus():
    """测试 derive focus stack boosts stable recent focus 场景。"""

    planner = _planner()
    state = TaskState(tenant_id="tenant", session_id="session")
    messages = [
        Message(
            message_id=f"msg_{index}",
            role="user",
            content="继续看北京天气",
            timestamp=float(index),
        )
        for index in range(1, 4)
    ]

    focus, stack = planner._derive_focus_stack("现在还适合吗？", state, messages)

    assert focus.locations == {"北京"}
    assert focus.boosted_locations == {"北京"}
    assert stack[0]["drifted"] is False


def test_derive_focus_stack_detects_topic_drift():
    """测试 derive focus stack detects topic drift 场景。"""

    planner = _planner()
    state = TaskState(tenant_id="tenant", session_id="session")
    messages = [
        Message(
            message_id=f"msg_{index}",
            role="user",
            content="继续看北京天气",
            timestamp=float(index),
        )
        for index in range(1, 4)
    ]

    focus, stack = planner._derive_focus_stack("上海天气呢？", state, messages)

    assert focus.locations == {"上海"}
    assert focus.boosted_locations == set()
    assert stack[0]["drifted"] is True
