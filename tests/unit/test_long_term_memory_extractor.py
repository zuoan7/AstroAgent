"""长期记忆抽取器稳定性测试。

覆盖抽取触发收敛、对话窗口聚合、LLM 开关/配置、结构化抽取结果和规则
fallback，保证普通天文问题不会被误写入用户画像。
"""

from unittest import mock

import pytest

from src.memory.long_term_memory.extractor import MemoryExtractor
from src.memory.long_term_memory.models import ExtractionResult, MemoryType, SourceType
from src.agent.prompts import get_prompt_renderer


@pytest.fixture
def extractor():
    """创建测试用 extractor fixture。"""

    return MemoryExtractor()


# ---------------------------------------------------------------------------
# 1. General astronomy questions should NOT trigger extraction
# ---------------------------------------------------------------------------

GENERAL_ASTRONOMY_QUESTIONS = [
    "今晚适合观测什么？",
    "M31 怎么看？",
    "火星什么时候升起？",
    "帮我推荐深空目标",
    "什么是黑洞？",
    "木星有多大？",
    "最近有什么流星雨？",
    "今晚的月亮好看吗",
    "土星环是怎么形成的",
    "银河系中心在哪",
    "望远镜应该怎么选？",
    "拍摄深空需要什么设备",  # general equipment question, not ownership
    "今晚能看到哪些行星？",
    "彗星什么时候来？",
    "星云是怎么分类的",
]


@pytest.mark.parametrize("question", GENERAL_ASTRONOMY_QUESTIONS)
def test_general_astronomy_does_not_trigger(extractor, question):
    """测试 general astronomy does not trigger 场景。"""

    assert extractor.should_attempt_extraction(question) is False, (
        f"普通天文问题不应触发抽取: {question}"
    )


ONE_OFF_STYLE_REQUESTS = [
    "简单介绍一下黑洞",
    "详细解释月食原理",
    "通俗讲讲光污染",
    "专业分析一下木星观测",
    "请用表格说明今晚观测目标",
    "别跟我说太专业的术语",
    "简短讲一下今晚适合看什么",
]


@pytest.mark.parametrize("question", ONE_OFF_STYLE_REQUESTS)
def test_one_off_style_request_does_not_trigger(extractor, question):
    """测试 one off style request does not trigger 场景。"""

    assert extractor.should_attempt_extraction(question) is False, (
        f"无长期信号的本轮风格请求不应触发抽取: {question}"
    )


LONG_TERM_STYLE_REQUESTS = [
    "以后请简短回答",
    "我喜欢通俗解释",
    "我偏好详细步骤",
    "请记住我不喜欢太专业",
    "下次默认用表格说明",
    "以后详细一点",
    "我希望你以后不要太专业",
    "请记住我喜欢简洁回答",
]


@pytest.mark.parametrize("question", LONG_TERM_STYLE_REQUESTS)
def test_long_term_style_request_does_trigger(extractor, question):
    """测试 long term style request does trigger 场景。"""

    assert extractor.should_attempt_extraction(question) is True, (
        f"长期风格偏好应触发抽取: {question}"
    )


@pytest.mark.parametrize("question", GENERAL_ASTRONOMY_QUESTIONS)
def test_general_astronomy_returns_empty_list(extractor, monkeypatch, question):
    """测试 general astronomy returns empty list 场景。"""

    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_ENABLED", True
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_LLM_EXTRACT_ENABLED", False
    )
    result = extractor.extract_from_conversation(question, "助手回复")
    assert result == [], f"普通天文问题 extract_from_conversation 应返回 []: {question}"


# ---------------------------------------------------------------------------
# 2. Explicit preferences SHOULD trigger extraction
# ---------------------------------------------------------------------------

EXPLICIT_PREFERENCE_MESSAGES = [
    "以后请用简短一点的方式回答我",
    "我喜欢用表格看观测建议",
    "我不想看太多专业术语",
    "记住，我喜欢详细的解释",
    "请记住，我不喜欢太长的回答",
    "以后默认用中文回答我",
    "我希望你每次都给我观测建议",
    "我偏好用通俗的方式讲解",
    "不要给我太多数学公式",
    "下次请直接给结论",
    "永远不要用英文回答",
    "我习惯在晚上观测",
    "请总是优先推荐行星观测",
    "以后都按这个格式输出",
    "以后请简短回答",
    "我偏好详细步骤",
    "请记住我不喜欢太专业",
    "我希望你以后通俗一点",
]



@pytest.mark.parametrize("question", EXPLICIT_PREFERENCE_MESSAGES)
def test_explicit_preference_triggers_extraction(extractor, question):
    """测试 explicit preference triggers extraction 场景。"""

    assert extractor.should_attempt_extraction(question) is True, (
        f"明确偏好表达应触发抽取: {question}"
    )


# ---------------------------------------------------------------------------
# 3. Explicit equipment / location SHOULD trigger extraction
# ---------------------------------------------------------------------------

EXPLICIT_EQUIPMENT_LOCATION_MESSAGES = [
    "我在北京观测",
    "我的观测地点是上海",
    "我有一台 80EQ 望远镜",
    "我用佳能相机拍深空",
    "我的望远镜是星特朗8SE",
    "我在广州拍照",
    "我主要拍摄深空天体",
    "我的观测地点在杭州",
    "我用的是信达小黑",
    "我的设备是索尼A7M3",
    "我有一架8寸道布森",
    "我是初学者",
    "我是有经验的天文爱好者",
    "我刚入门天文摄影",
    "我主要观测行星",
]


@pytest.mark.parametrize("question", EXPLICIT_EQUIPMENT_LOCATION_MESSAGES)
def test_explicit_equipment_location_triggers(extractor, question):
    """测试 explicit equipment location triggers 场景。"""

    assert extractor.should_attempt_extraction(question) is True, (
        f"明确设备/地点/技能表达应触发抽取: {question}"
    )


# ---------------------------------------------------------------------------
# 4. Disable total switch (LTM_EXTRACT_ENABLED = False)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "user_msg,assistant_msg",
    [
        ("我喜欢简短的回答", "好的，我会简短回答"),
        ("我在北京观测", "北京今晚天气不错"),
        ("我有一台望远镜", "望远镜可以帮助观测"),
    ],
)
def test_ltm_extract_disabled_returns_empty(
    extractor, monkeypatch, user_msg, assistant_msg
):
    """测试 ltm extract disabled returns empty 场景。"""

    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_ENABLED", False
    )
    result = extractor.extract_from_conversation(user_msg, assistant_msg)
    assert result == [], (
        f"LTM_EXTRACT_ENABLED=False 时应返回 []: {user_msg}"
    )


# ---------------------------------------------------------------------------
# 5. Disable LLM extraction (LTM_LLM_EXTRACT_ENABLED = False)
# ---------------------------------------------------------------------------

def test_llm_extract_disabled_does_not_call_llm(extractor, monkeypatch):
    """测试 llm extract disabled does not call llm 场景。"""

    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_ENABLED", True
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_LLM_EXTRACT_ENABLED", False
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.DASHSCOPE_API_KEY", "fake-key"
    )

    with mock.patch.object(
        extractor, "extract_with_llm", wraps=extractor.extract_with_llm
    ) as spy:
        result = extractor.extract_from_conversation(
            "我喜欢简短的回答", "好的助手回复"
        )
        spy.assert_not_called()

    # Fallback should return conservative results for explicit preference
    assert any(
        r.key == "response_style" and r.value == "简短" for r in result
    ), "LLM 禁用时 fallback 应对明确偏好返回保守结果"


def test_llm_disabled_general_astronomy_returns_empty(extractor, monkeypatch):
    """测试 llm disabled general astronomy returns empty 场景。"""

    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_ENABLED", True
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_LLM_EXTRACT_ENABLED", False
    )
    result = extractor.extract_from_conversation("今晚适合观测什么？", "助手回复")
    assert result == [], "LLM 禁用时普通天文问题 fallback 应返回 []"


# ---------------------------------------------------------------------------
# 6. LLM config: lightweight model, timeout, retries
# ---------------------------------------------------------------------------

def test_extract_with_llm_uses_ltm_config(extractor, monkeypatch):
    """测试 extract with llm uses ltm config 场景。"""

    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_MODEL_NAME",
        "qwen-plus",
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_TIMEOUT_SECONDS",
        6.0,
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_MAX_RETRIES", 0
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.DASHSCOPE_API_KEY", "test-key"
    )

    with mock.patch(
        "src.memory.long_term_memory.extractor.build_chat_model"
    ) as mock_build:
        mock_llm = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"extractions": []}'
        mock_llm.invoke.return_value = mock_response
        mock_build.return_value = mock_llm

        extractor.extract_with_llm("测试消息", "助手回复")

        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs["model"] == "qwen-plus", (
            f"应使用 LTM_EXTRACT_MODEL_NAME(qwen-plus)，实际: {call_kwargs['model']}"
        )
        assert call_kwargs["request_timeout"] == 6.0, (
            f"应使用 LTM_EXTRACT_TIMEOUT_SECONDS(6.0)，实际: {call_kwargs['request_timeout']}"
        )
        assert call_kwargs["max_retries"] == 0, (
            f"应使用 LTM_EXTRACT_MAX_RETRIES(0)，实际: {call_kwargs['max_retries']}"
        )


def test_extract_with_llm_falls_back_to_small_model(extractor, monkeypatch):
    """测试 extract with llm falls back to small model 场景。"""

    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_MODEL_NAME", ""
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.SMALL_MODEL_NAME", "qwen-plus"
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.DASHSCOPE_API_KEY", "test-key"
    )

    with mock.patch(
        "src.memory.long_term_memory.extractor.build_chat_model"
    ) as mock_build:
        mock_llm = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"extractions": []}'
        mock_llm.invoke.return_value = mock_response
        mock_build.return_value = mock_llm

        extractor.extract_with_llm("测试消息", "助手回复")

        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs["model"] == "qwen-plus", (
            f"LTM_EXTRACT_MODEL_NAME 为空时应 fallback 到 SMALL_MODEL_NAME"
        )


# ---------------------------------------------------------------------------
# 7. Fallback does NOT generate frequent_topics from general astronomy
# ---------------------------------------------------------------------------

def test_fallback_no_frequent_topics_from_astronomy(extractor):
    """普通天文主题不应在 fallback 中生成 frequent_topics habit."""
    result = extractor._fallback_keyword_extraction(
        "我想看看火星和木星", "火星和木星是太阳系行星"
    )
    frequent_topic_results = [
        r for r in result
        if r.memory_type == MemoryType.HABIT and r.key == "frequent_topics"
    ]
    assert len(frequent_topic_results) == 0, (
        "fallback 不应从普通天文主题生成 frequent_topics"
    )


def test_fallback_no_observation_type_from_keywords(extractor):
    """深空/行星/摄影等词不应自动生成 observation_type habit."""
    for msg in ["我想拍深空天体", "行星观测有什么技巧", "怎么拍摄星空"]:
        result = extractor._fallback_keyword_extraction(msg, "助手回复")
        obs_type_results = [
            r for r in result
            if r.memory_type == MemoryType.HABIT and r.key == "observation_type"
        ]
        assert len(obs_type_results) == 0, (
            f"fallback 不应从 '{msg}' 自动生成 observation_type"
        )


def test_fallback_still_extracts_explicit_signals(extractor):
    """Fallback 仍应提取明确表达的偏好/约束/技能."""
    # Explicit response style preference
    result = extractor._fallback_keyword_extraction("请以后简短回答", "")
    assert any(r.key == "response_style" and r.value == "简短" for r in result)

    # Explicit knowledge level
    result = extractor._fallback_keyword_extraction("请用专业术语解释", "")
    assert any(r.key == "knowledge_level" and r.value == "专业" for r in result)

    # Explicit constraint
    result = extractor._fallback_keyword_extraction("不要使用术语解释", "")
    assert any(r.key == "no_jargon" for r in result)

    # Explicit skill level
    result = extractor._fallback_keyword_extraction("我是初学者", "")
    assert any(r.key == "skill_level" and r.value == "入门" for r in result)


def test_fallback_extracts_device_with_ownership(extractor):
    """Fallback 应在用户明确声明拥有设备时提取."""
    result = extractor._fallback_keyword_extraction("我用星特朗8SE望远镜", "")
    assert any(r.key == "device_info" for r in result), (
        "fallback 应提取明确声明的设备信息"
    )


def test_fallback_extracts_location_with_context(extractor):
    """Fallback 应在用户声明观测地点时提取."""
    result = extractor._fallback_keyword_extraction("我在北京观测", "")
    assert any(r.key == "location_info" for r in result), (
        "fallback 应提取明确声明的观测位置"
    )


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

def test_empty_message_does_not_trigger(extractor):
    """测试 empty message does not trigger 场景。"""

    assert extractor.should_attempt_extraction("") is False
    assert extractor.should_attempt_extraction("  ") is False
    assert extractor.should_attempt_extraction("a") is False


def test_temporary_request_without_memory_signal(extractor):
    """本轮临时要求（如'这次简短回答'）不应触发长期记忆抽取."""
    assert extractor.should_attempt_extraction("这次简短回答一下") is False
    assert extractor.should_attempt_extraction("本次请详细解释") is False


def test_temporary_with_memory_signal_triggers(extractor):
    """即使本轮临时要求，如包含'以后'等记忆信号仍应触发."""
    assert extractor.should_attempt_extraction("以后都简短回答，这次也不例外") is True
    assert extractor.should_attempt_extraction("这次也请记住，我喜欢表格") is True


def test_city_without_context_does_not_trigger(extractor):
    """仅提及城市名但不涉及观测/设备上下文不应触发."""
    assert extractor.should_attempt_extraction("北京今天天气怎么样") is False


def test_short_message_with_keyword_triggers(extractor):
    """短消息但包含明确偏好的应触发."""
    assert extractor.should_attempt_extraction("记住，我喜欢简短") is True
    assert extractor.should_attempt_extraction("以后都要详细") is True


def test_window_aggregation_triggers_repeated_equipment(extractor, monkeypatch):
    """测试 window aggregation triggers repeated equipment 场景。"""

    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_ENABLED", True
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_LLM_EXTRACT_ENABLED", False
    )
    window = [
        {
            "user_message": "我用星特朗8SE看木星",
            "assistant_message": "可以",
            "conversation_id": "t1",
        },
        {
            "user_message": "星特朗8SE接目镜怎么配？",
            "assistant_message": "建议先确认焦距",
            "conversation_id": "t2",
        },
        {
            "user_message": "接着说调焦方案",
            "assistant_message": "继续说明",
            "conversation_id": "t3",
        },
    ]

    assert extractor.should_attempt_extraction(
        "接着说调焦方案", conversation_window=window
    )
    results = extractor.extract_from_conversation(
        "接着说调焦方案",
        "继续说明",
        conversation_window=window,
    )

    assert any(
        r.key == "device_info" and r.extraction_grade == "tentative"
        for r in results
    )
    assert any("window_repeated_signal" in r.gate_reason for r in results)


def test_llm_should_extract_false_writes_no_results(extractor, monkeypatch):
    """测试 llm should extract false writes no results 场景。"""

    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_EXTRACT_ENABLED", True
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.LTM_LLM_EXTRACT_ENABLED", True
    )
    monkeypatch.setattr(
        "src.memory.long_term_memory.extractor.settings.DASHSCOPE_API_KEY", "test-key"
    )

    with mock.patch(
        "src.memory.long_term_memory.extractor.build_chat_model"
    ) as mock_build:
        mock_llm = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.content = (
            '{"should_extract": false, "reason": "one-off", "extractions": []}'
        )
        mock_llm.invoke.return_value = mock_response
        mock_build.return_value = mock_llm

        result = extractor.extract_from_conversation("我喜欢简短回答", "好的")

    assert result == []
    rendered_prompt = get_prompt_renderer().render(
        "memory.long_term_extractor.user",
        {
            "user_message": "我喜欢简短回答",
            "assistant_message": "好的",
            "conversation_window_json": "[]",
            "gating_signals_json": "{}",
        },
    )
    assert "conversation_window_json" not in rendered_prompt
    assert "最近 4 轮窗口 JSON" in rendered_prompt
    assert "Gating 信号 JSON" in rendered_prompt
