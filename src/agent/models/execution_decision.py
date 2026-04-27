"""
ExecutionDecision — 最小可用执行模式决策，Phase 3 引入。

从 TaskProfile/ExecutionContext 推断执行模式，目标替代 choose_path(route) 的 route 字符串传参。

MVP 字段：mode / reason / fallback_modes / legacy_execution_path

当前状态：模型已稳定，decide() 可通过 ENABLE_EXECUTION_DECISION flag 开启。
          主路径仍调用 choose_path(route)；flag 关闭时 decide() 内部委托 choose_path()，
          结果可观测但不驱动执行。
收敛计划：待 UnifiedExecutionEngine 实现后，decide() 作为主路径入口，
          替代 choose_path(route) 调用；choose_path() 降为兼容别名。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


VALID_EXECUTION_MODES = {"direct", "planned", "react"}


@dataclass(frozen=True)
class ExecutionDecision:
    mode: str                           # direct | planned | react
    reason: str                         # 决策原因（可读字符串）
    fallback_modes: List[str] = field(default_factory=list)
    legacy_execution_path: str = "unknown"  # 与 choose_path() 保持可比较

    def __post_init__(self):
        if self.mode not in VALID_EXECUTION_MODES:
            raise ValueError(f"Invalid mode: {self.mode!r}, must be one of {VALID_EXECUTION_MODES}")

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "fallback_modes": list(self.fallback_modes),
            "legacy_execution_path": self.legacy_execution_path,
        }
