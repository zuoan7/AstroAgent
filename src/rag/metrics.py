"""
检索质量监控与用户反馈机制

功能:
  - 检索质量指标收集（延迟、命中率、相关性分数分布）
  - 用户反馈收集（相关性评分、问题标记）
  - 指标聚合与报告生成
  - 持久化存储（SQLite）
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger


@dataclass
class RetrievalMetrics:
    query: str
    latency_ms: float
    num_results: int
    top_score: float
    avg_score: float
    cache_hit: bool
    pipeline_stages: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class UserFeedback:
    query: str
    relevance_rating: int  # 1-5
    is_accurate: bool
    comment: str = ""
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """检索质量指标收集器"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(
            settings.VECTOR_DB_PATH, "rag_metrics.sqlite"
        )
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    num_results INTEGER NOT NULL,
                    top_score REAL NOT NULL,
                    avg_score REAL NOT NULL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    pipeline_stages TEXT,
                    timestamp REAL NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    relevance_rating INTEGER NOT NULL,
                    is_accurate INTEGER NOT NULL,
                    comment TEXT DEFAULT '',
                    timestamp REAL NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_ts ON retrieval_metrics(timestamp)
            """)
            self._conn.commit()
        except Exception as e:
            logger.warning(f"⚠️  指标数据库初始化失败: {e}")
            self._conn = None

    def record_retrieval(self, metrics: RetrievalMetrics) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute(
                """INSERT INTO retrieval_metrics
                   (query, latency_ms, num_results, top_score, avg_score, cache_hit, pipeline_stages, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    metrics.query[:500],
                    metrics.latency_ms,
                    metrics.num_results,
                    metrics.top_score,
                    metrics.avg_score,
                    1 if metrics.cache_hit else 0,
                    json.dumps(metrics.pipeline_stages),
                    metrics.timestamp,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"⚠️  记录检索指标失败: {e}")

    def record_feedback(self, feedback: UserFeedback) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute(
                """INSERT INTO user_feedback
                   (query, relevance_rating, is_accurate, comment, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    feedback.query[:500],
                    feedback.relevance_rating,
                    1 if feedback.is_accurate else 0,
                    feedback.comment,
                    feedback.timestamp,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"⚠️  记录用户反馈失败: {e}")

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        if not self._conn:
            return {}

        cutoff = time.time() - hours * 3600
        try:
            cursor = self._conn.execute(
                "SELECT COUNT(*), AVG(latency_ms), AVG(top_score), AVG(avg_score), "
                "SUM(CASE WHEN cache_hit=1 THEN 1 ELSE 0 END) "
                "FROM retrieval_metrics WHERE timestamp > ?",
                (cutoff,),
            )
            row = cursor.fetchone()
            total = row[0] if row else 0
            avg_latency = row[1] if row else 0
            avg_top = row[2] if row else 0
            avg_avg = row[3] if row else 0
            cache_hits = row[4] if row else 0

            cursor = self._conn.execute(
                "SELECT AVG(relevance_rating), SUM(CASE WHEN is_accurate=1 THEN 1 ELSE 0 END), COUNT(*) "
                "FROM user_feedback WHERE timestamp > ?",
                (cutoff,),
            )
            fb_row = cursor.fetchone()
            avg_rating = fb_row[0] if fb_row and fb_row[0] else 0.0
            accurate_count = fb_row[1] if fb_row and fb_row[1] else 0
            feedback_total = fb_row[2] if fb_row and fb_row[2] else 0

            return {
                "period_hours": hours,
                "total_queries": total or 0,
                "avg_latency_ms": round(avg_latency or 0, 2),
                "avg_top_score": round(avg_top or 0, 4),
                "avg_avg_score": round(avg_avg or 0, 4),
                "cache_hit_rate": round((cache_hits or 0) / max(total or 1, 1), 4),
                "avg_user_rating": round(avg_rating, 2),
                "accuracy_rate": round((accurate_count or 0) / max(feedback_total or 1, 1), 4),
                "total_feedback": feedback_total or 0,
            }
        except Exception as e:
            logger.warning(f"⚠️  获取指标摘要失败: {e}")
            return {}

    def get_latency_percentiles(self, hours: int = 24) -> Dict[str, float]:
        if not self._conn:
            return {}
        cutoff = time.time() - hours * 3600
        try:
            cursor = self._conn.execute(
                "SELECT latency_ms FROM retrieval_metrics WHERE timestamp > ? ORDER BY latency_ms",
                (cutoff,),
            )
            latencies = [row[0] for row in cursor.fetchall()]
            if not latencies:
                return {}
            n = len(latencies)
            return {
                "p50": latencies[int(n * 0.5)],
                "p90": latencies[int(n * 0.9)],
                "p95": latencies[int(n * 0.95)],
                "p99": latencies[min(int(n * 0.99), n - 1)],
                "min": latencies[0],
                "max": latencies[-1],
            }
        except Exception:
            return {}

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
