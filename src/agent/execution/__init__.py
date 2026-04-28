"""统一执行引擎包（Phase 4 引入）。

当前状态：ExecutionEngine 已是默认主执行入口，统一 direct / planned / react。
          TaskOrchestrator 降级为兼容回退门面。
"""
from src.agent.execution.engine import ExecutionEngine
from src.agent.execution.direct_executor import DirectExecutor
from src.agent.execution.planned_executor import PlannedExecutor
from src.agent.execution.react_executor import ReactExecutor

__all__ = [
    "ExecutionEngine",
    "DirectExecutor",
    "PlannedExecutor",
    "ReactExecutor",
]
