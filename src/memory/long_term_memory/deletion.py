from src.memory.long_term_memory.models import (
    EventLogEntry,
    EventType,
    LongTermMemoryDeletionRequest,
    LongTermMemoryDeletionResult,
)
from src.memory.long_term_memory.profile_projection import ProfileProjection
from src.memory.long_term_memory.repository import LongTermMemoryRepository


class LongTermMemoryDeletionService:
    """Applies tombstone deletes and records audit entries for long-term memory."""

    SUPPORTED_SCOPES = {"memory", "candidate", "profile", "user_all"}

    def __init__(self, repository: LongTermMemoryRepository, projection: ProfileProjection):
        self._repo = repository
        self._projection = projection

    def delete(self, request: LongTermMemoryDeletionRequest) -> LongTermMemoryDeletionResult:
        if request.scope not in self.SUPPORTED_SCOPES:
            raise ValueError(f"Unsupported long-term memory deletion scope: {request.scope}")

        result = LongTermMemoryDeletionResult(request=request)
        self._repo.add_event_log(EventLogEntry(
            user_id=request.user_id,
            memory_id=request.target_id,
            event_type=EventType.DELETION_REQUESTED,
            event_detail=f"长期记忆删除请求: {request.scope}",
            metadata={"reason": request.reason, "requested_by": request.requested_by},
        ))

        if request.scope == "memory":
            if not request.target_id:
                raise ValueError("memory deletion requires target_id")
            result.deleted_memories = self._repo.tombstone_memory(
                request.user_id, request.target_id, request.reason
            )
            self._projection.rebuild(request.user_id)
        elif request.scope == "candidate":
            if not request.target_id:
                raise ValueError("candidate deletion requires target_id")
            result.deleted_candidates = self._repo.tombstone_candidate(
                request.user_id, request.target_id, request.reason
            )
        elif request.scope == "profile":
            result.deleted_profiles = 1 if self._repo.delete_profile(request.user_id) else 0
        elif request.scope == "user_all":
            result.deleted_memories = self._repo.tombstone_user_memories(request.user_id, request.reason)
            result.deleted_candidates = self._repo.tombstone_user_candidates(request.user_id, request.reason)
            result.deleted_profiles = 1 if self._repo.delete_profile(request.user_id) else 0
            self._repo.mark_legacy_events_deleted(request.user_id, request.reason)

        result.audit_id = self._repo.add_deletion_audit(
            user_id=request.user_id,
            scope=request.scope,
            target_id=request.target_id,
            reason=request.reason,
            requested_by=request.requested_by,
            deleted_memories=result.deleted_memories,
            deleted_candidates=result.deleted_candidates,
            deleted_profiles=result.deleted_profiles,
            metadata=request.metadata,
        )
        self._repo.add_event_log(EventLogEntry(
            user_id=request.user_id,
            memory_id=request.target_id,
            event_type=EventType.DELETION_APPLIED,
            event_detail=f"长期记忆删除完成: {request.scope}",
            metadata=result.to_dict(),
        ))
        return result
