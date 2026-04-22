import pytest

from src.agent.output_parser import (
    AgentFinish,
    LenientReActSingleInputOutputParser,
    OutputParserException,
)


def test_parser_accepts_unlabeled_final_answer_after_thought():
    parser = LenientReActSingleInputOutputParser()

    result = parser.parse(
        "Thought: 我现在知道最终答案了\n"
        "今晚最佳观测目标是娥眉月。建议傍晚朝西方低空观测，"
        "月光较弱，也适合顺便观测较亮恒星。"
    )

    assert isinstance(result, AgentFinish)
    assert result.return_values["output"].startswith("今晚最佳观测目标是娥眉月")


def test_parser_accepts_direct_unlabeled_final_answer():
    parser = LenientReActSingleInputOutputParser()

    result = parser.parse(
        "🌟 **你好呀！为你整理了 2026年4月的天象预报。**\n\n"
        "✨ **本月特殊天象**\n"
        "• **2026-04-22** 天琴座流星雨极大"
    )

    assert isinstance(result, AgentFinish)
    assert "2026年4月的天象预报" in result.return_values["output"]


def test_parser_keeps_tool_intent_parse_errors():
    parser = LenientReActSingleInputOutputParser()

    with pytest.raises(OutputParserException):
        parser.parse("Thought: 需要调用 ObservationPlanner 工具查询今晚最佳观测目标。")
