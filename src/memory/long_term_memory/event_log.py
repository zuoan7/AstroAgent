from typing import Any, Dict, List, Optional

from src.core.logger import logger
from src.memory.long_term_memory.models import (
    EventLogEntry,
    EventType,
    MemoryConfirmation,
    ConfirmationStatus,
    _utcnow_iso,
)
from src.memory.long_term_memory.repository import LongTermMemoryRepository


class EventLogger:
    def __init__(self, repository: LongTermMemoryRepository):
        self._repo = repository

    def log_event(
        self,
        user_id: str,
        event_type: str,
        event_detail: str,
        memory_id: Optional[str] = None,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        entry = EventLogEntry(
            user_id=user_id,
            memory_id=memory_id,
            event_type=event_type,
            event_detail=event_detail,
            old_value=old_value,
            new_value=new_value,
            metadata=metadata or {},
        )
        log_id = self._repo.add_event_log(entry)
        logger.debug(f"记忆事件: [{event_type}] {event_detail[:100]}")
        return log_id

    def log_created(self, user_id: str, memory_id: str, memory_type: str, key: str, value: Any):
        self.log_event(
            user_id=user_id,
            event_type=EventType.CREATED,
            event_detail=f"创建记忆: {memory_type}.{key}",
            memory_id=memory_id,
            new_value=str(value),
            metadata={"memory_type": memory_type, "key": key},
        )

    def log_updated(self, user_id: str, memory_id: str, key: str, old_value: Any, new_value: Any, reason: str = ""):
        self.log_event(
            user_id=user_id,
            event_type=EventType.UPDATED,
            event_detail=f"更新记忆: {key}" + (f" ({reason})" if reason else ""),
            memory_id=memory_id,
            old_value=str(old_value) if old_value is not None else None,
            new_value=str(new_value) if new_value is not None else None,
            metadata={"reason": reason},
        )

    def log_deleted(self, user_id: str, memory_id: str, key: str, value: Any):
        self.log_event(
            user_id=user_id,
            event_type=EventType.DELETED,
            event_detail=f"删除记忆: {key}",
            memory_id=memory_id,
            old_value=str(value),
        )

    def log_accessed(self, user_id: str, memory_id: str, key: str):
        self.log_event(
            user_id=user_id,
            event_type=EventType.ACCESSED,
            event_detail=f"访问记忆: {key}",
            memory_id=memory_id,
        )

    def log_expired(self, user_id: str, memory_id: str, key: str):
        self.log_event(
            user_id=user_id,
            event_type=EventType.EXPIRED,
            event_detail=f"记忆过期: {key}",
            memory_id=memory_id,
        )

    def log_archived(self, user_id: str, memory_id: str, key: str):
        self.log_event(
            user_id=user_id,
            event_type=EventType.ARCHIVED,
            event_detail=f"记忆归档: {key}",
            memory_id=memory_id,
        )

    def log_conflict(self, user_id: str, memory_id: str, conflict_type: str, resolution: str, old_value: Any = None, new_value: Any = None):
        self.log_event(
            user_id=user_id,
            event_type=EventType.CONFLICT_DETECTED,
            event_detail=f"冲突检测: {conflict_type}, 解决策略: {resolution}",
            memory_id=memory_id,
            old_value=str(old_value) if old_value is not None else None,
            new_value=str(new_value) if new_value is not None else None,
            metadata={"conflict_type": conflict_type, "resolution": resolution},
        )

    def log_conflict_resolved(self, user_id: str, memory_id: str, resolution: str, detail: str = ""):
        self.log_event(
            user_id=user_id,
            event_type=EventType.CONFLICT_RESOLVED,
            event_detail=f"冲突解决: {resolution}" + (f" - {detail}" if detail else ""),
            memory_id=memory_id,
            metadata={"resolution": resolution},
        )

    def log_merged(self, user_id: str, memory_id: str, key: str, old_value: Any, new_value: Any):
        self.log_event(
            user_id=user_id,
            event_type=EventType.MERGED,
            event_detail=f"合并记忆: {key}",
            memory_id=memory_id,
            old_value=str(old_value) if old_value is not None else None,
            new_value=str(new_value) if new_value is not None else None,
        )

    def log_deduplicated(self, user_id: str, memory_id: str, key: str, detail: str = ""):
        self.log_event(
            user_id=user_id,
            event_type=EventType.DEDUPLICATED,
            event_detail=f"去重处理: {key}" + (f" - {detail}" if detail else ""),
            memory_id=memory_id,
        )

    def get_event_logs(
        self, user_id: str, memory_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[EventLogEntry]:
        return self._repo.get_event_logs(user_id, memory_id=memory_id, limit=limit, offset=offset)

    def get_memory_trace(self, user_id: str, memory_id: str) -> List[EventLogEntry]:
        return self._repo.get_event_logs(user_id, memory_id=memory_id, limit=100)


class ConfirmationManager:
    def __init__(self, repository: LongTermMemoryRepository, event_logger: EventLogger):
        self._repo = repository
        self._event_logger = event_logger

    def create_confirmation(
        self,
        user_id: str,
        memory_id: str,
        confirmation_type: str,
        content: str,
    ) -> MemoryConfirmation:
        confirmation = MemoryConfirmation(
            user_id=user_id,
            memory_id=memory_id,
            confirmation_type=confirmation_type,
            content=content,
        )
        self._repo.add_confirmation(confirmation)
        self._event_logger.log_event(
            user_id=user_id,
            event_type=EventType.CONFIRMATION_REQUESTED,
            event_detail=f"请求确认: {confirmation_type}",
            memory_id=memory_id,
            metadata={"confirmation_id": confirmation.id},
        )
        return confirmation

    def resolve_confirmation(
        self, confirmation_id: str, status: str
    ) -> Optional[MemoryConfirmation]:
        confirmation = self._repo.get_confirmation(confirmation_id)
        if not confirmation:
            return None
        if confirmation.status != ConfirmationStatus.PENDING:
            return confirmation

        self._repo.update_confirmation_status(confirmation_id, status)
        confirmation.status = status
        confirmation.resolved_at = _utcnow_iso()

        self._event_logger.log_event(
            user_id=confirmation.user_id,
            event_type=EventType.CONFIRMATION_RESOLVED,
            event_detail=f"确认结果: {status}",
            memory_id=confirmation.memory_id,
            metadata={"confirmation_id": confirmation_id, "status": status},
        )
        return confirmation

    def list_pending_confirmations(self, user_id: str, limit: int = 20) -> List[MemoryConfirmation]:
        return self._repo.list_pending_confirmations(user_id, limit=limit)

    def batch_confirm(self, user_id: str, confirmation_ids: List[str], status: str) -> List[MemoryConfirmation]:
        results = []
        for cid in confirmation_ids:
            result = self.resolve_confirmation(cid, status)
            if result:
                results.append(result)
        return results
