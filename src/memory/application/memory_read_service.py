"""短期记忆读取服务。

从事件投影、任务状态和 summary snapshot 中组装 prompt 上下文，
同时提供调试、导出和兼容旧接口的读取能力。
"""

from typing import Any, Dict

from src.memory.api.dto import BuildContextRequest
from src.memory.application.event_projection_reader import EventProjectionReader
from src.memory.application.summary_snapshot_manager import SummarySnapshotManager
from src.memory.application.task_state_manager import TaskStateManager
from src.memory.domain.task_state import TaskState
from src.memory.infrastructure.repositories.event_store import EventStore
from src.memory.retrieval import RetrievalPlanner


class MemoryReadService:
    """Reads short-term memory context from stable projections."""

    def __init__(
        self,
        event_store: EventStore,
        task_state_manager: TaskStateManager,
        summary_snapshot_manager: SummarySnapshotManager,
        retrieval_planner: RetrievalPlanner,
    ):
        self.event_store = event_store
        self.task_state_manager = task_state_manager
        self.summary_snapshot_manager = summary_snapshot_manager
        self.retrieval_planner = retrieval_planner
        self.projection_reader = EventProjectionReader(event_store)

    def get_task_state(self, session_id: str, tenant_id: str) -> TaskState:
        """读取会话任务状态；不存在时由 manager 返回默认投影。"""

        return self.task_state_manager.get_state(tenant_id, session_id)

    def build_context(self, request: BuildContextRequest) -> Dict[str, Any]:
        """收集稳定投影并交给 RetrievalPlanner 做预算内上下文选择。"""

        token_budget = request.max_tokens or 4000
        task_state = self.get_task_state(request.session_id, request.tenant_id)
        summary_snapshot = self.summary_snapshot_manager.get_latest(request.session_id)
        messages = self.projection_reader.list_messages(request.session_id)
        facts = self.projection_reader.list_salient_facts(request.session_id)
        tool_calls = self.projection_reader.list_tool_calls(request.session_id)
        return self.retrieval_planner.build_context(
            query=request.query,
            token_budget=token_budget,
            task_state=task_state,
            summary_snapshot=summary_snapshot,
            messages=messages,
            facts=facts,
            tool_calls=tool_calls,
        )

    def get_debug_info(self, session_id: str) -> Dict[str, Any]:
        """统计当前会话事件、消息、工具调用、事实和摘要可用性。"""

        messages = self.projection_reader.list_messages(session_id, limit=1000)
        tool_calls = self.projection_reader.list_tool_calls(session_id, limit=1000)
        facts = self.projection_reader.list_salient_facts(session_id, limit=1000)
        return {
            "session_id": session_id,
            "event_count": len(self.event_store.list_by_session(session_id, limit=5000)),
            "message_count": len(messages),
            "tool_call_count": len(tool_calls),
            "fact_count": len(facts),
            "summary_available": self.summary_snapshot_manager.get_latest(session_id)
            is not None,
        }

    def get_all_messages(self, session_id: str) -> list[Dict[str, Any]]:
        return [
            item.to_dict()
            for item in self.projection_reader.list_messages(session_id, limit=1000)
        ]

    def get_tool_calls(self, session_id: str) -> list[Dict[str, Any]]:
        return [
            item.to_dict()
            for item in self.projection_reader.list_tool_calls(session_id, limit=1000)
        ]

    def get_salient_facts(self, session_id: str) -> list[Dict[str, Any]]:
        return [
            item.to_dict()
            for item in self.projection_reader.list_salient_facts(session_id, limit=1000)
        ]

    def get_summary(self, session_id: str) -> str:
        latest = self.summary_snapshot_manager.get_latest(session_id)
        return latest.summary_text if latest else ""

    def export_memory(self, session_id: str, tenant_id: str) -> Dict[str, Any]:
        """导出会话事件流、任务状态和最新 summary snapshot。"""

        latest = self.summary_snapshot_manager.get_latest(session_id)
        return {
            "session_id": session_id,
            "events": [
                event.to_dict()
                for event in self.event_store.list_by_session(session_id, limit=5000)
            ],
            "task_state": self.get_task_state(session_id, tenant_id).to_dict(),
            "summary_snapshot": latest.to_dict() if latest else None,
        }
