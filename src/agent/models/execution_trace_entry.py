"""ExecutionTraceEntry — 统一执行 trace 模型（Phase 7 引入）。

三种执行路径（direct / planned / react）的步骤执行结果统一表示。
设计原则：与 StepExecutionResult 结构保持 1:1 映射，同时可从旧 dict（FinalResponse.execution_trace）
         无损恢复，实现与旧数据格式的双向兼容。

当前状态：StepExecutionResult 新增 to_trace_entry()；FinalResponse.execution_trace 仍为 list[dict]，
          不做 breaking change。
收敛计划：Phase 8 可将 FinalResponse.execution_trace 类型升级为 List[ExecutionTraceEntry]，
          届时删除 from_dict 中的兼容性注释。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionTraceEntry:
    """单步执行结果，兼容 StepExecutionResult 与旧 execution_trace dict。"""

    step_id: str
    title: str
    kind: str
    status: str
    skill: Optional[str] = None
    input_params: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    required: bool = True
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    summary: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    # react 路径额外字段（planned/direct 路径不使用）
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output_summary: Optional[str] = None
    duration_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status,
            "skill": self.skill,
            "input_params": dict(self.input_params),
            "attempts": self.attempts,
            "required": self.required,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "summary": self.summary,
            "sources": list(self.sources),
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output_summary": self.tool_output_summary,
            "duration_sec": self.duration_sec,
        }

    @classmethod
    def from_step_result(cls, step_result: Any) -> "ExecutionTraceEntry":
        """从 StepExecutionResult 构造（planned/direct 路径）。"""
        return cls(
            step_id=step_result.step_id,
            title=step_result.title,
            kind=step_result.kind,
            status=step_result.status,
            skill=step_result.skill,
            input_params=dict(step_result.input_params),
            attempts=step_result.attempts,
            required=step_result.required,
            latency_ms=step_result.latency_ms,
            error=step_result.error,
            summary=step_result.summary,
            sources=list(step_result.sources),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionTraceEntry":
        """从旧 execution_trace list[dict] 恢复（兼容层）。"""
        return cls(
            step_id=d.get("step_id", ""),
            title=d.get("title", ""),
            kind=d.get("kind", "tool"),
            status=d.get("status", ""),
            skill=d.get("skill"),
            input_params=d.get("input_params", {}),
            attempts=d.get("attempts", 0),
            required=d.get("required", True),
            latency_ms=d.get("latency_ms"),
            error=d.get("error"),
            summary=d.get("summary", ""),
            sources=d.get("sources", []),
            tool_name=d.get("tool_name"),
            tool_input=d.get("tool_input"),
            tool_output_summary=d.get("tool_output_summary"),
            duration_sec=d.get("duration_sec"),
        )

    @classmethod
    def from_react_tool(
        cls,
        *,
        step_id: str,
        tool_name: str,
        tool_input: str,
        output_summary: str,
        status: str,
        duration_sec: Optional[float] = None,
    ) -> "ExecutionTraceEntry":
        """从 react 路径的 tool_start/end 事件构造。"""
        return cls(
            step_id=step_id,
            title=tool_name,
            kind="tool",
            status=status,
            skill=tool_name,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output_summary=output_summary,
            duration_sec=duration_sec,
        )
