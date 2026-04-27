"""
ExecutionResult — 统一执行结果，Phase 2 引入。

薄包装层：持有 FinalResponse + 解包后的 execution_trace + raw_artifacts（可选）。

当前状态：模型已稳定，但主路径各执行分支仍直接返回 FinalResponse。
          from_final_response() 目前仅在 _run_orchestrated_path() 旁路记录块中调用，
          不参与主执行逻辑。
收敛计划：待 UnifiedExecutionEngine 实现后，成为所有执行路径（direct/planned/react）
          的统一输出，替代各路径直接返回 FinalResponse 的方式。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.models.final_response import FinalResponse


@dataclass
class ExecutionResult:
    response: FinalResponse
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    raw_artifacts: Dict[str, Any] = field(default_factory=dict)
    execution_path: str = "unknown"

    # ── 便捷属性，减少调用层的字段访问深度 ─────────────────────────────────
    @property
    def answer(self) -> str:
        return self.response.answer

    @property
    def sources(self) -> List[Dict[str, Any]]:
        return self.response.sources

    @property
    def tools_used(self) -> List[Dict[str, Any]]:
        return self.response.tools_used

    def to_dict(self) -> Dict[str, Any]:
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
