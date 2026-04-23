from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config import resolve_path, settings


class RequestAuditLogger:
    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        output_path: Optional[str] = None,
    ) -> None:
        self._enabled = (
            bool(getattr(settings, "AGENT_AUDIT_ENABLED", True))
            if enabled is None
            else bool(enabled)
        )
        self._output_path = resolve_path(
            output_path or getattr(settings, "AGENT_AUDIT_LOG_PATH", "logs/agent_audit/requests.jsonl")
        )
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def append(self, payload: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        path = Path(self._output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"logged_at": time.time(), **payload}
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
