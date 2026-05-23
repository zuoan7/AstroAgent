"""长期记忆 embedding 缓存服务。

本文件负责把 active 长期记忆转换为统一文本、生成 DashScope embedding、
写入 SQLite 缓存，并为语义召回提供只读缓存访问。模型调用只用于生成或
回填缓存；检索主链路只消费已有缓存，任何异常都会降级到规则召回。
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.core.config import settings
from src.core.logger import logger
from src.memory.long_term_memory.models import MemoryItem, MemoryStatus, _utcnow_iso
from src.memory.long_term_memory.repository import LongTermMemoryRepository


class MemoryEmbeddingService:
    """维护 active 长期记忆的 SQLite embedding 缓存。"""

    def __init__(
        self,
        repository: LongTermMemoryRepository,
        enabled: bool = True,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 0.8,
        backfill_limit: int = 50,
        max_workers: int = 1,
    ):
        """初始化 embedding 配置、降级原因和可选后台回填线程池。"""

        self._repo = repository
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.api_key = api_key if api_key is not None else settings.DASHSCOPE_API_KEY
        self.timeout_seconds = max(float(timeout_seconds or 0.8), 0.1)
        self.backfill_limit = max(int(backfill_limit or 50), 0)
        self.enabled = bool(enabled) and bool(self.api_key) and bool(self.model_name)
        if not self.enabled:
            if not enabled:
                self.disabled_reason = "semantic_retrieval_disabled"
            elif not self.api_key:
                self.disabled_reason = "missing_dashscope_api_key"
            else:
                self.disabled_reason = "missing_embedding_model"
        else:
            self.disabled_reason = ""

        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending_futures: set[Future] = set()
        self._scheduled_ids: set[str] = set()
        self._lock = Lock()
        self._shutdown = False
        if self.enabled and max_workers > 0:
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="ltm-embed",
            )

    def embedding_text(self, item: MemoryItem) -> str:
        """构造统一的 embedding 文本：type/category/key/value。"""

        return "/".join(
            [
                str(item.memory_type or ""),
                str(item.category or ""),
                str(item.key or ""),
                self._value_text(item.value),
            ]
        )

    def content_hash(self, item: MemoryItem) -> str:
        """计算 embedding 文本的稳定 hash，用于判断缓存是否过期。"""

        return hashlib.sha256(self.embedding_text(item).encode("utf-8")).hexdigest()

    def cached_embeddings_for_items(
        self, user_id: str, items: Iterable[MemoryItem]
    ) -> Tuple[Dict[str, List[float]], List[MemoryItem]]:
        """返回当前模型且 hash 未过期的缓存；缺失或过期项单独返回。"""

        item_list = list(items)
        if not item_list:
            return {}, []
        records = self._repo.get_memory_embeddings(
            user_id,
            memory_ids=[item.id for item in item_list],
            model_name=self.model_name,
        )
        current: Dict[str, List[float]] = {}
        stale: List[MemoryItem] = []
        for item in item_list:
            record = records.get(item.id)
            if self._record_current(item, record):
                current[item.id] = [float(value) for value in record["embedding"]]
            else:
                stale.append(item)
        return current, stale

    def embed_query(self, query: str) -> Tuple[Optional[List[float]], Optional[str]]:
        """生成查询 embedding；失败时返回 fallback reason。"""

        if not self.enabled:
            return None, self.disabled_reason
        try:
            vector = self._embed_text(str(query or ""))
            if not vector:
                return None, "empty_embedding_response"
            return vector, None
        except Exception as exc:
            logger.debug("长期记忆查询 embedding 失败: %s", exc, exc_info=True)
            return None, type(exc).__name__

    def schedule_embedding(self, item: MemoryItem) -> Optional[Future]:
        """后台补齐单条 active 记忆 embedding，不阻塞主链路。"""

        if not self.enabled or not self._executor or self._shutdown:
            return None
        if item.status != MemoryStatus.ACTIVE:
            return None
        with self._lock:
            if item.id in self._scheduled_ids:
                return None
            self._scheduled_ids.add(item.id)
        future = self._executor.submit(self.embed_memory_if_needed, item)
        with self._lock:
            self._pending_futures.add(future)
        future.add_done_callback(lambda fut, memory_id=item.id: self._cleanup_future(fut, memory_id))
        return future

    def schedule_embeddings(self, items: Iterable[MemoryItem], limit: Optional[int] = None) -> int:
        """批量调度后台补齐，返回实际入队数量。"""

        queued = 0
        max_items = limit if limit is not None else self.backfill_limit
        for item in items:
            if max_items is not None and queued >= max_items:
                break
            if self.schedule_embedding(item):
                queued += 1
        return queued

    def embed_memory_if_needed(self, item: MemoryItem) -> str:
        """同步构建单条缓存，返回 created/updated/skipped/failed。"""

        if item.status != MemoryStatus.ACTIVE:
            return "skipped"
        existing = self._repo.get_memory_embedding(item.id)
        if self._record_current(item, existing):
            return "skipped"
        if not self.enabled:
            return "skipped"
        try:
            vector = self._embed_text(self.embedding_text(item))
            if not vector:
                return "failed"
            self._repo.upsert_memory_embedding(
                memory_id=item.id,
                user_id=item.user_id,
                content_hash=self.content_hash(item),
                model_name=self.model_name,
                embedding=vector,
                updated_at=_utcnow_iso(),
            )
            return "updated" if existing else "created"
        except Exception:
            logger.debug("长期记忆 embedding 构建失败: %s", item.id, exc_info=True)
            return "failed"

    def rebuild_index(
        self, user_id: Optional[str] = None, limit: int = 500
    ) -> Dict[str, int]:
        """同步回填缺失或过期的 active memories embedding。"""

        result = {"created": 0, "updated": 0, "skipped": 0, "failed": 0}
        items = self._repo.list_active_memories(user_id=user_id, limit=max(int(limit), 0))
        for item in items:
            status = self.embed_memory_if_needed(item)
            result[status] = result.get(status, 0) + 1
        return result

    def backfill_missing_or_stale(
        self, user_id: Optional[str] = None, limit: Optional[int] = None
    ) -> Dict[str, int]:
        """维护任务使用的小批量同步回填。"""

        max_items = self.backfill_limit if limit is None else max(int(limit), 0)
        if max_items <= 0 or not self.enabled:
            return {"created": 0, "updated": 0, "skipped": 0, "failed": 0}
        candidates = self._repo.list_active_memories(
            user_id=user_id,
            limit=max(max_items * 5, max_items),
        )
        stale = []
        for item in candidates:
            if len(stale) >= max_items:
                break
            if not self._record_current(item, self._repo.get_memory_embedding(item.id)):
                stale.append(item)
        result = {"created": 0, "updated": 0, "skipped": 0, "failed": 0}
        for item in stale:
            status = self.embed_memory_if_needed(item)
            result[status] = result.get(status, 0) + 1
        return result

    def flush(self, timeout: float = 5.0) -> None:
        """等待已调度的 embedding 任务完成，主要用于测试和收尾。"""

        import time

        deadline = time.monotonic() + timeout
        with self._lock:
            pending = list(self._pending_futures)
        for future in pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                future.result(timeout=max(0.1, remaining))
            except Exception:
                pass

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        """关闭后台 embedding 线程池并清空待处理任务记录。"""

        if self._shutdown:
            return
        self._shutdown = True
        if self._executor:
            try:
                self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            except TypeError:
                self._executor.shutdown(wait=wait)
        with self._lock:
            self._pending_futures.clear()
            self._scheduled_ids.clear()

    def _embed_text(self, text: str) -> List[float]:
        """调用 DashScope OpenAI-compatible embeddings 接口生成向量。"""

        import requests

        base_url = getattr(settings, "OPENAI_COMPATIBLE_BASE_URL", "").rstrip("/")
        if not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        payload = {"model": self.model_name, "input": text}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        vector = self._parse_embedding_response(data)
        return [float(value) for value in vector]

    def _parse_embedding_response(self, data: Dict[str, Any]) -> List[float]:
        """兼容解析 OpenAI 风格和 DashScope 原生风格的 embedding 响应。"""

        if isinstance(data.get("data"), list) and data["data"]:
            embedding = data["data"][0].get("embedding")
            if isinstance(embedding, list):
                return embedding
        output = data.get("output")
        if isinstance(output, dict):
            embeddings = output.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                embedding = embeddings[0].get("embedding")
                if isinstance(embedding, list):
                    return embedding
        embedding = data.get("embedding")
        if isinstance(embedding, list):
            return embedding
        return []

    def _record_current(
        self, item: MemoryItem, record: Optional[Dict[str, Any]]
    ) -> bool:
        """判断缓存记录是否匹配当前模型、内容 hash 和向量维度。"""

        if not record:
            return False
        return (
            record.get("model_name") == self.model_name
            and record.get("content_hash") == self.content_hash(item)
            and bool(record.get("embedding"))
            and int(record.get("dimensions") or 0) == len(record.get("embedding") or [])
        )

    def _cleanup_future(self, future: Future, memory_id: str) -> None:
        """后台任务结束后移除 pending 标记，并吞掉非关键异常。"""

        with self._lock:
            self._pending_futures.discard(future)
            self._scheduled_ids.discard(memory_id)
        try:
            future.result()
        except Exception:
            logger.debug("长期记忆 embedding 后台任务异常: %s", memory_id, exc_info=True)

    def _value_text(self, value: Any) -> str:
        """把任意 memory value 规范化为 embedding 文本片段。"""

        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)
