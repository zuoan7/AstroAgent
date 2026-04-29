"""ExecutionEvent — 统一执行事件模型。

封装 router / policy / engine / executor 内部产生的结构化事件，
作为 StreamingService 的上游协议。旧前端事件名继续保留，
由适配层完成 ExecutionEvent -> StreamEvent 映射。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# 内部事件类型集合（与前端事件名解耦，映射由前端事件 adapter 完成）
EXECUTION_EVENT_TYPES = {
    "task_profile",       # Router 画像已生成
    "route_decided",      # 路由决策完成
    "execution_decision", # Policy/Engine 执行决策完成
    "plan_built",         # 执行计划已生成
    "plan_created",       # plan_built 兼容别名
    "step_started",       # 步骤开始执行
    "step_finished",      # 步骤执行结束
    "tool_result",        # react 路径工具返回
    "fallback_triggered", # 触发降级/兜底
    "answer_ready",       # 最终答案已生成
    "final_answer",       # answer_ready 兼容别名
    "tool_called",        # react 路径工具调用（on_tool_start）
    "tool_returned",      # tool_result 兼容别名
}


@dataclass
class ExecutionEvent:
    """执行过程中产生的内部事件，承载结构化数据。

    type: 内部事件类型（见 EXECUTION_EVENT_TYPES）
    payload: 事件数据（类型由 type 决定）
    source: 产生该事件的执行器名称（"direct", "planned", "react"）
    """

    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_frontend_type(self) -> Optional[str]:
        """兼容接口：返回旧前端事件名。

        主映射逻辑已迁入 FrontendExecutionEventAdapter；
        此方法仅保留给旧测试和外部兼容调用。
        """
        from src.agent.frontend_event_adapter import FrontendExecutionEventAdapter

        return FrontendExecutionEventAdapter.FRONTEND_EVENT_TYPE_MAP.get(self.type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "payload": dict(self.payload),
            "source": self.source,
        }
