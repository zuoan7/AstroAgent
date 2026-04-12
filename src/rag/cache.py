"""
多级缓存系统

三级缓存架构:
  L1 - 内存热缓存 (LRU, 短TTL): 高频查询即时响应
  L2 - 内存温缓存 (大容量, 长TTL): 中频查询快速响应
  L3 - 磁盘持久缓存 (SQLite): 跨重启持久化

缓存键: query + top_k 的 MD5 哈希
缓存值: 格式化后的检索结果字符串
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from cachetools import LRUCache, TTLCache

from src.core.config import settings
from src.core.logger import logger


class MultiLevelCache:
    """多级缓存系统"""

    def __init__(
        self,
        l1_maxsize: int = 64,
        l1_ttl: int = 60,
        l2_maxsize: int = 256,
        l2_ttl: int = 300,
        l3_path: Optional[str] = None,
        l3_enabled: bool = True,
    ):
        self.l1 = TTLCache(maxsize=l1_maxsize, ttl=l1_ttl)
        self.l2 = TTLCache(maxsize=l2_maxsize, ttl=l2_ttl)

        self.l3_enabled = l3_enabled
        self.l3_path = l3_path or os.path.join(settings.VECTOR_DB_PATH, "rag_cache.sqlite")
        self._l3_conn: Optional[sqlite3.Connection] = None

        if self.l3_enabled:
            self._init_l3()

        self._stats = {"l1_hits": 0, "l2_hits": 0, "l3_hits": 0, "misses": 0}

    def _init_l3(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.l3_path), exist_ok=True)
            self._l3_conn = sqlite3.connect(self.l3_path, check_same_thread=False)
            self._l3_conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_text TEXT NOT NULL,
                    query_text TEXT,
                    created_at REAL NOT NULL,
                    ttl_seconds INTEGER DEFAULT 3600,
                    access_count INTEGER DEFAULT 1
                )
            """)
            self._l3_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON rag_cache(created_at)
            """)
            self._l3_conn.commit()
            logger.info(f"✅ L3 磁盘缓存初始化: {self.l3_path}")
        except Exception as e:
            logger.warning(f"⚠️  L3 缓存初始化失败: {e}")
            self.l3_enabled = False

    @staticmethod
    def _make_key(query: str, top_k: int) -> str:
        raw = f"{query}::{top_k}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, query: str, top_k: int) -> Optional[str]:
        key = self._make_key(query, top_k)

        result = self.l1.get(key)
        if result is not None:
            self._stats["l1_hits"] += 1
            return result

        result = self.l2.get(key)
        if result is not None:
            self._stats["l2_hits"] += 1
            self.l1[key] = result
            return result

        if self.l3_enabled and self._l3_conn:
            result = self._get_l3(key)
            if result is not None:
                self._stats["l3_hits"] += 1
                self.l2[key] = result
                self.l1[key] = result
                return result

        self._stats["misses"] += 1
        return None

    def _get_l3(self, key: str) -> Optional[str]:
        try:
            cursor = self._l3_conn.execute(
                "SELECT result_text, created_at, ttl_seconds FROM rag_cache WHERE cache_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            result_text, created_at, ttl_seconds = row
            if time.time() - created_at > ttl_seconds:
                self._l3_conn.execute("DELETE FROM rag_cache WHERE cache_key = ?", (key,))
                self._l3_conn.commit()
                return None

            self._l3_conn.execute(
                "UPDATE rag_cache SET access_count = access_count + 1 WHERE cache_key = ?",
                (key,),
            )
            self._l3_conn.commit()
            return result_text
        except Exception as e:
            logger.warning(f"⚠️  L3 缓存读取失败: {e}")
            return None

    def set(self, query: str, top_k: int, result: str, ttl: int = 3600) -> None:
        if not result:
            return

        key = self._make_key(query, top_k)

        self.l1[key] = result
        self.l2[key] = result

        if self.l3_enabled and self._l3_conn:
            self._set_l3(key, result, query, ttl)

    def _set_l3(self, key: str, result: str, query: str, ttl: int) -> None:
        try:
            self._l3_conn.execute(
                """INSERT OR REPLACE INTO rag_cache
                   (cache_key, result_text, query_text, created_at, ttl_seconds, access_count)
                   VALUES (?, ?, ?, ?, ?, 1)""",
                (key, result, query[:200], time.time(), ttl),
            )
            self._l3_conn.commit()
        except Exception as e:
            logger.warning(f"⚠️  L3 缓存写入失败: {e}")

    def invalidate(self, query: str, top_k: int) -> None:
        key = self._make_key(query, top_k)
        self.l1.pop(key, None)
        self.l2.pop(key, None)
        if self.l3_enabled and self._l3_conn:
            try:
                self._l3_conn.execute("DELETE FROM rag_cache WHERE cache_key = ?", (key,))
                self._l3_conn.commit()
            except Exception:
                pass

    def clear(self) -> None:
        self.l1.clear()
        self.l2.clear()
        if self.l3_enabled and self._l3_conn:
            try:
                self._l3_conn.execute("DELETE FROM rag_cache")
                self._l3_conn.commit()
            except Exception:
                pass

    def cleanup_expired(self) -> int:
        count = 0
        if self.l3_enabled and self._l3_conn:
            try:
                cutoff = time.time() - 86400
                cursor = self._l3_conn.execute(
                    "DELETE FROM rag_cache WHERE created_at < ?", (cutoff,)
                )
                self._l3_conn.commit()
                count = cursor.rowcount
            except Exception:
                pass
        return count

    def get_stats(self) -> Dict[str, Any]:
        total = sum(self._stats.values())
        hit_rate = 0.0
        if total > 0:
            hits = self._stats["l1_hits"] + self._stats["l2_hits"] + self._stats["l3_hits"]
            hit_rate = hits / total

        return {
            "l1_size": len(self.l1),
            "l2_size": len(self.l2),
            "l1_hits": self._stats["l1_hits"],
            "l2_hits": self._stats["l2_hits"],
            "l3_hits": self._stats["l3_hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "l3_enabled": self.l3_enabled,
        }

    def close(self) -> None:
        if self._l3_conn:
            try:
                self._l3_conn.close()
            except Exception:
                pass
