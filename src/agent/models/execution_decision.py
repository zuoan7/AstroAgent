"""执行模式决策模型，描述 direct、planned、react 的选择原因和 fallback 模式。

该模型由策略层从 TaskProfile/ExecutionContext 推断得到，用于替代旧的
choose_path(route) 字符串分支；legacy_execution_path 只保留兼容对比用途。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


VALID_EXECUTION_MODES = {"direct", "planned", "react"}


@dataclass(frozen=True)
class ExecutionDecision:
    """执行模式决策，表示本轮走 direct、planned 还是 react。"""
    mode: str                           # direct | planned | react
    reason: str                         # 决策原因（可读字符串）
    fallback_modes: List[str] = field(default_factory=list)
    legacy_execution_path: str = "unknown"  # 与 choose_path() 保持可比较

    def __post_init__(self):
        """在 dataclass 初始化后校验和规范化字段。"""
        if self.mode not in VALID_EXECUTION_MODES:
            raise ValueError(f"Invalid mode: {self.mode!r}, must be one of {VALID_EXECUTION_MODES}")

    def to_dict(self) -> dict:
        """将当前模型转换为可序列化字典。"""
        return {
            "mode": self.mode,
            "reason": self.reason,
            "fallback_modes": list(self.fallback_modes),
            "legacy_execution_path": self.legacy_execution_path,
        }
