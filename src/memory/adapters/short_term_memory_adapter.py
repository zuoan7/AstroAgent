import time
from typing import Any, Dict, List, Optional

from src.memory.api.dto import AppendMessageRequest, AppendToolCallRequest, BuildContextRequest, DeleteMemoryRequest
from src.memory.api.memory_service import MemoryService
from src.memory.short_term_memory.config import ShortTermMemoryConfig
from src.memory.short_term_memory.manager import ShortTermMemory


class ShortTermMemoryAdapter:
    """Compatibility adapter exposing the legacy ShortTermMemory surface.

    Existing callers can use this class like ShortTermMemory while the adapter
    writes through MemoryService into raw event/artifact stores.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: str = "default",
        service: Optional[MemoryService] = None,
    ):
        self._legacy = ShortTermMemory(session_id=session_id, user_id=user_id)
        self.session_id = self._legacy.session_id
        self.user_id = self._legacy.user_id
        self.tenant_id = tenant_id
        self.config = self._legacy.config
        self._service = service or MemoryService(
            db_path=self.config.persistence_path,
            tenant_id=tenant_id,
            short_term_memory=self._legacy,
        )

    @property
    def messages(self):
        return self._legacy.messages

    @property
    def tool_calls(self):
        return self._legacy.tool_calls

    @property
    def salient_facts(self):
        return self._legacy.salient_facts

    @property
    def summary(self) -> str:
        return self._legacy.summary

    @summary.setter
    def summary(self, value: str) -> None:
        self._legacy.summary = value

    def add_message(
        self,
        role: str,
        content: str,
        timestamp: Optional[float] = None,
        importance: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        return self._service.append_message(
            AppendMessageRequest(
                tenant_id=self.tenant_id,
                session_id=self.session_id,
                user_id=self.user_id,
                role=role,
                content=content,
                timestamp=timestamp or time.time(),
                importance=importance,
                metadata=metadata or {},
            )
        )

    def add_tool_call(
        self,
        tool_name: str,
        tool_input: str,
        result: str,
        timestamp: Optional[float] = None,
        success: bool = True,
    ):
        return self._service.append_tool_call(
            AppendToolCallRequest(
                tenant_id=self.tenant_id,
                session_id=self.session_id,
                user_id=self.user_id,
                tool_name=tool_name,
                tool_input=tool_input,
                raw_output=result,
                timestamp=timestamp or time.time(),
                success=success,
            )
        )

    def add_salient_fact(
        self,
        fact_type: str,
        content: str,
        source: str = "",
        timestamp: Optional[float] = None,
    ):
        return self._legacy.add_salient_fact(fact_type, content, source=source, timestamp=timestamp)

    def build_context(self, max_tokens: Optional[int] = None, query: str = "") -> Dict[str, Any]:
        return self._service.build_context(
            BuildContextRequest(
                tenant_id=self.tenant_id,
                session_id=self.session_id,
                query=query,
                max_tokens=max_tokens,
            )
        )

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        return self.build_context(max_tokens=max_tokens)["context_text"]

    def get_recent_messages(self, window: Optional[int] = None) -> List[Dict[str, str]]:
        return self._legacy.get_recent_messages(window=window)

    def get_all_messages(self) -> List[Dict[str, Any]]:
        return self._legacy.get_all_messages()

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        return self._legacy.get_tool_calls()

    def get_salient_facts(self) -> List[Dict[str, Any]]:
        return self._legacy.get_salient_facts()

    def get_summary(self) -> str:
        return self._legacy.get_summary()

    def get_debug_info(self) -> Dict[str, Any]:
        info = self._legacy.get_debug_info()
        info["adapter"] = "ShortTermMemoryAdapter"
        return info

    def get_context_debug_info(self) -> Dict[str, Any]:
        return self._legacy.get_context_debug_info()

    def clear(self) -> None:
        self._service.delete_memory(
            DeleteMemoryRequest(
                tenant_id=self.tenant_id,
                scope="session",
                selector={"session_id": self.session_id},
                requested_by=self.user_id,
            )
        )
        self._legacy.clear()

    def clear_session(self) -> None:
        self.clear()

    def get_size(self) -> int:
        return self._legacy.get_size()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return ShortTermMemory._estimate_tokens(text)

    @classmethod
    def restore_session(cls, session_id: str) -> Optional["ShortTermMemoryAdapter"]:
        config = ShortTermMemoryConfig.from_settings()
        if not config.enable_persistence:
            return None
        legacy = ShortTermMemory.restore_session(session_id)
        if not legacy:
            return None
        adapter = cls(session_id=session_id, user_id=legacy.user_id)
        adapter._legacy = legacy
        adapter._service.short_term_memory = legacy
        return adapter
