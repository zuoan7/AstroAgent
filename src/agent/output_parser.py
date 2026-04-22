"""Output parsers for agent ReAct responses."""

from __future__ import annotations

import re
from typing import Any, ClassVar

try:
    from langchain_classic.agents.output_parsers.react_single_input import (
        ReActSingleInputOutputParser,
    )
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.exceptions import OutputParserException
except (ImportError, ModuleNotFoundError):
    class OutputParserException(Exception):
        pass

    class AgentAction:
        def __init__(self, tool: str, tool_input: str, log: str):
            self.tool = tool
            self.tool_input = tool_input
            self.log = log

    class AgentFinish:
        def __init__(self, return_values: dict[str, Any], log: str):
            self.return_values = return_values
            self.log = log

    class ReActSingleInputOutputParser:
        def parse(self, text: str) -> AgentAction | AgentFinish:
            if "Final Answer:" in text:
                return AgentFinish(
                    {"output": text.rsplit("Final Answer:", maxsplit=1)[-1].strip()},
                    text,
                )

            action_match = re.search(
                r"Action\s*\d*\s*:[\s]*(.*?)Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)",
                text,
                re.DOTALL,
            )
            if action_match:
                return AgentAction(
                    action_match.group(1).strip(),
                    action_match.group(2).strip(" ").strip('"'),
                    text,
                )

            raise OutputParserException(f"Could not parse LLM output: `{text}`")


class LenientReActSingleInputOutputParser(ReActSingleInputOutputParser):
    """Accept final answers that forgot the explicit ``Final Answer:`` label.

    Some chat models occasionally follow a final ``Thought:`` with the user-facing
    answer directly. The stock ReAct parser treats that as a missing ``Action:``
    and burns another LLM round on format correction. This parser keeps the stock
    behavior for tool calls and malformed actions, but converts answer-like text
    into an ``AgentFinish``.
    """

    _ACTION_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"Action\s*\d*\s*:", re.IGNORECASE
    )
    _FINAL_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"Final Answer\s*:", re.IGNORECASE
    )
    _THOUGHT_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"Thought\s*:\s*", re.IGNORECASE
    )
    _INTENT_ONLY_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(需要|应当|应该|准备|先|下一步).{0,16}"
        r"(调用|使用|查询|检索|计算|选择).{0,16}(工具|技能|Action|数据)?"
    )

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            return super().parse(text)
        except OutputParserException:
            answer = self._extract_unlabeled_final_answer(text)
            if answer:
                return AgentFinish({"output": answer}, text)
            raise

    @classmethod
    def _extract_unlabeled_final_answer(cls, text: str) -> str | None:
        if cls._ACTION_RE.search(text) or cls._FINAL_RE.search(text):
            return None

        parts = cls._THOUGHT_RE.split(text)
        candidate = parts[-1].strip() if len(parts) >= 2 else text.strip()
        if not candidate:
            return None

        lines = candidate.splitlines()
        if len(lines) > 1 and cls._is_meta_thought(lines[0]):
            candidate = "\n".join(lines[1:]).strip()

        if not candidate or cls._looks_like_tool_intent(candidate):
            return None

        return candidate

    @staticmethod
    def _is_meta_thought(line: str) -> bool:
        normalized = line.strip(" 。:：")
        return normalized in {
            "我现在知道最终答案了",
            "现在我知道最终答案了",
            "我可以回答了",
            "现在可以回答了",
            "整理观测结果",
            "整理天象结果",
        }

    @classmethod
    def _looks_like_tool_intent(cls, text: str) -> bool:
        first_line = text.splitlines()[0].strip()
        if len(text) <= 80 and cls._INTENT_ONLY_RE.search(first_line):
            return True
        return False
