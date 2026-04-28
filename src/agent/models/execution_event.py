"""ExecutionEvent — 统一执行事件模型。

封装 router / policy / engine / executor 内部产生的结构化事件，
作为 StreamingService 的上游协议。旧前端事件名继续保留，
由适配层完成 ExecutionEvent -> StreamEvent 映射。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# 内部事件类型集合（与前端事件名解耦，映射在 StreamingService 中完成）
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
        """将内部事件类型映射为旧前端事件类型（保持兼容）。

        收敛计划：Phase 8 可将此映射移入 FrontendJsonEventAdapter，
                  届时 ExecutionEvent 与 StreamEvent 解耦更彻底。
        """
        _MAP = {
            "route_decided": "route_decision",
            "plan_built": "plan_update",
            "plan_created": "plan_update",
            "step_started": "step_start",
            "step_finished": "step_end",
            "answer_ready": "final_answer",
            "final_answer": "final_answer",
            "tool_called": "tool_start",
            "tool_result": "tool_end",
            "tool_returned": "tool_end",
        }
        return _MAP.get(self.type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "payload": dict(self.payload),
            "source": self.source,
        }
