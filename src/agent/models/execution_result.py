"""统一执行结果模型，包装 FinalResponse 并提供旧字段兼容访问。

当前主路径仍直接返回 FinalResponse，本模型作为薄包装层保留给旁路记录、
兼容接口和后续统一输出收敛使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.models.final_response import FinalResponse


@dataclass
class ExecutionResult:
    """统一执行结果，包装 FinalResponse 并兼容旧字段访问。"""
    response: FinalResponse
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    raw_artifacts: Dict[str, Any] = field(default_factory=dict)
    execution_path: str = "unknown"

    # ── 便捷属性，减少调用层的字段访问深度 ─────────────────────────────────
    @property
    def answer(self) -> str:
        """返回最终答案文本。"""
        return self.response.answer

    @property
    def sources(self) -> List[Dict[str, Any]]:
        """返回最终答案使用的来源列表。"""
        return self.response.sources

    @property
    def tools_used(self) -> List[Dict[str, Any]]:
        """返回最终答案使用的工具时间线。"""
        return self.response.tools_used

    def to_dict(self) -> Dict[str, Any]:
        """将当前模型转换为可序列化字典。"""
        return {
            "response": self.response.to_dict(),
            "execution_trace": list(self.execution_trace),
            "raw_artifacts": dict(self.raw_artifacts),
            "execution_path": self.execution_path,
        }

    @classmethod
    def from_final_response(
        cls,
        response: FinalResponse,
        *,
        execution_path: str = "unknown",
        raw_artifacts: Optional[Dict[str, Any]] = None,
    ) -> "ExecutionResult":
        """[Legacy adapter] 从现有 FinalResponse 构造 ExecutionResult。

        当前仅被 _run_orchestrated_path() 旁路记录块调用（写 latency meta，不驱动执行）。
        待 UnifiedExecutionEngine 实现后，主路径将原生返回 ExecutionResult，本方法降为可选。
        """
        return cls(
            response=response,
            execution_trace=list(response.execution_trace),
            raw_artifacts=raw_artifacts or {},
            execution_path=execution_path,
        )
