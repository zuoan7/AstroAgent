"""ReactExecutor — ReAct 任务执行器（Phase 4 引入）。

React 执行逻辑的独立入口，解耦其与 StreamingService 的强绑定。
初版作为薄封装：持有 agent_executor 引用，暴露 astream_events() 接口。

当前状态：ReactExecutor 已独立存在，但 ENABLE_UNIFIED_EXECUTION_ENGINE=False 时
          StreamingService 仍直接调用 agent_executor，本类不接入主路径。
收敛计划：待 flag 开启后，StreamingService 改为通过 ExecutionEngine -> ReactExecutor
          执行 react 路径；agent_executor 的直接引用从 StreamingService 移除。
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Callable, Dict, Optional


class ReactExecutor:
    """React 执行器薄封装，持有 agent_executor 引用并代理其流式接口。"""

    def __init__(
        self,
        agent_executor: Optional[Any] = None,
        agent_executor_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._agent_executor = agent_executor
        self._agent_executor_factory = agent_executor_factory

    def ensure_executor(self) -> Any:
        if self._agent_executor is not None:
            return self._agent_executor
        if self._agent_executor_factory is None:
            raise ValueError("react agent executor is not configured")
        self._agent_executor = self._agent_executor_factory()
        return self._agent_executor

    async def astream_events(
        self,
        agent_input: Dict[str, Any],
        *,
        version: str = "v1",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """代理 agent_executor.astream_events()，使 react 拥有独立入口。"""
        executor = self.ensure_executor()
        async for event in executor.astream_events(agent_input, version=version):
            yield event
