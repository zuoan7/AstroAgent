"""统一执行引擎包（Phase 4 引入）。

当前状态：ExecutionEngine 及三个 Executor 已存在，但 ENABLE_UNIFIED_EXECUTION_ENGINE 默认关闭。
          主路径仍由 StreamingService -> TaskOrchestrator 驱动；本包为旁路结构统一层。
收敛计划：待 ENABLE_UNIFIED_EXECUTION_ENGINE 开启后，StreamingService 改为调用
          ExecutionEngine.run()，TaskOrchestrator 降级为兼容门面。
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
