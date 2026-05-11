from src.memory.core.models import Message, ToolCallRecord
from src.memory.domain.task_state import TaskState
from src.memory.retrieval.planner import RetrievalPlanner


def _planner() -> RetrievalPlanner:
    return RetrievalPlanner(lambda text: len(text or ""))


def test_extract_tool_meta_parses_weather_location_and_freshness():
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


def test_extract_focus_uses_query_and_task_state_without_negative_entities():
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


def test_rank_messages_with_focus_drops_zero_query_score_outside_recent_four():
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
