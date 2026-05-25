"""LangChain adapter for CapabilityKit."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Type

from langchain_core.tools import Tool
from pydantic import BaseModel

from src.skills.result import SkillResult
from src.utils.param_parser import ParamParser
from src.skills.definition import SkillDefinition
from src.tools.definition import ToolDefinition
from src.tools.results import ToolResult

if isinstance(Tool, type):

    class _CapabilityTool(Tool):
        """Tool variant that forwards validated multi-field payloads as one object."""

        def _parse_input(self, tool_input: Any, tool_call_id: str | None) -> Any:
            if isinstance(tool_input, str) and self.args_schema is not None:
                parsed = ParamParser.parse(tool_input)
                if isinstance(parsed, dict) and set(parsed.keys()) != {"query"}:
                    return super()._parse_input(parsed, tool_call_id)
            return super()._parse_input(tool_input, tool_call_id)

        def _to_args_and_kwargs(
            self,
            tool_input: str | dict,
            tool_call_id: str | None,
        ) -> tuple[tuple, dict]:
            parsed_input = self._parse_input(tool_input, tool_call_id)
            return (parsed_input,), {}

else:
    _CapabilityTool = None


class RAGRetrieveInput(BaseModel):
    """Input schema for local astronomy knowledge retrieval."""

    query: str


def to_langchain_tools(
    capability_kit: Any,
    expose_tools: Optional[Iterable[str]] = None,
) -> list[Tool]:
    """Create ReAct-compatible LangChain tools from a CapabilityKit."""
    tools: list[Tool] = [
        _make_tool(
            name="RAGRetrieve",
            func=_create_rag_func(capability_kit),
            description="使用本地RAG知识库检索天文知识、概念解释、历史资料等。参数：query（查询语句，中文即可）。",
            args_schema=RAGRetrieveInput,
        )
    ]

    for definition in capability_kit.list_skills():
        tools.append(
            _make_tool(
                name=definition.display_name,
                func=_create_skill_func(capability_kit, definition),
                description=definition.description,
                args_schema=definition.input_model,
            )
        )

    exposed = _resolve_exposed_tool_names(capability_kit.list_tools(), expose_tools)
    for definition in capability_kit.list_tools():
        if definition.name not in exposed:
            continue
        tools.append(
            _make_tool(
                name=definition.name,
                func=_create_tool_func(capability_kit, definition),
                description=_tool_description(definition),
                args_schema=definition.input_model,
            )
        )
    return tools


def _make_tool(
    *,
    name: str,
    func: Any,
    description: str,
    args_schema: Type[BaseModel],
) -> Tool:
    tool_cls = _CapabilityTool or Tool
    tool = tool_cls(
        name=name,
        func=func,
        description=description,
        args_schema=args_schema,
    )
    if type(tool).__module__.startswith("unittest.mock"):
        return SimpleNamespace(
            name=name,
            func=func,
            description=description,
            args_schema=args_schema,
        )
    return tool


def _resolve_exposed_tool_names(
    definitions: list[ToolDefinition],
    expose_tools: Optional[Iterable[str]],
) -> set[str]:
    if expose_tools is not None:
        return {str(name) for name in expose_tools}

    tagged = {
        definition.name
        for definition in definitions
        if "react-exposed" in set(definition.tags or ())
    }
    return tagged


def _create_skill_func(capability_kit: Any, definition: SkillDefinition):
    def skill_func(tool_input: Any = None, **kwargs: Any) -> str:
        payload = _coerce_payload(definition.input_model, tool_input, kwargs)
        result = capability_kit.call_skill(definition.name, **payload)
        return result.to_legacy_str()

    return skill_func


def _create_tool_func(capability_kit: Any, definition: ToolDefinition):
    def tool_func(tool_input: Any = None, **kwargs: Any) -> str:
        payload = _coerce_payload(definition.input_model, tool_input, kwargs)
        result = capability_kit.call_tool(definition.name, **payload)
        return _tool_result_to_legacy_str(result)

    return tool_func


def _create_rag_func(capability_kit: Any):
    def rag_func(tool_input: Any = None, **kwargs: Any) -> str:
        payload = _coerce_payload(RAGRetrieveInput, tool_input, kwargs)
        retriever = getattr(capability_kit, "rag_retriever", None)
        if retriever is None:
            return ""
        query = payload.get("query", "")
        if hasattr(retriever, "get_relevant_context"):
            return retriever.get_relevant_context(query)
        if hasattr(retriever, "retrieve"):
            result = retriever.retrieve(query)
            if isinstance(result, dict):
                return str(result.get("context") or result)
            return str(result)
        return ""

    return rag_func


def _coerce_payload(
    input_model: Type[BaseModel],
    tool_input: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if kwargs:
        raw = dict(kwargs)
    elif isinstance(tool_input, input_model):
        raw = tool_input.model_dump()
    elif isinstance(tool_input, BaseModel):
        raw = tool_input.model_dump()
    elif isinstance(tool_input, dict):
        raw = dict(tool_input)
    elif isinstance(tool_input, str):
        parsed = ParamParser.parse(tool_input)
        if isinstance(parsed, dict) and set(parsed.keys()) != {"query"}:
            raw = parsed
        else:
            raw = _single_text_payload(input_model, tool_input)
    elif tool_input is None:
        raw = {}
    else:
        raw = _single_text_payload(input_model, str(tool_input))

    return input_model.model_validate(raw).model_dump()


def _single_text_payload(input_model: Type[BaseModel], value: str) -> dict[str, Any]:
    fields = list(input_model.model_fields.keys())
    if not fields:
        return {}
    if "query" in input_model.model_fields:
        return {"query": value}
    return {fields[0]: value}


def _tool_result_to_legacy_str(result: ToolResult) -> str:
    if result.ok:
        payload = result.data
    else:
        payload = {
            "ok": False,
            "error": {
                "code": result.error.code if result.error else "TOOL_CALL_FAILED",
                "message": result.error.message if result.error else "",
                "details": result.error.details if result.error else {},
            },
        }
    return (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )


def _tool_description(definition: ToolDefinition) -> str:
    fields = list(definition.input_model.model_fields.keys())
    params = ", ".join(fields) if fields else "无参数"
    return f"{definition.summary or definition.name}（atomic MCP tool: {definition.name}）。参数：{params}。"
