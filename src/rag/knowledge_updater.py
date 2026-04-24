"""
知识更新机制

支持天文数据的增量更新:
  - 检测数据源变更（文件哈希对比）
  - 增量索引更新（仅处理新增/变更文档）
  - 定时更新调度
  - 在线数据源接入（NASA APOD, NEO 等）
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger
from src.rag.bm25_retriever import BM25Retriever


@dataclass
class UpdateRecord:
    source: str
    record_id: str
    content_hash: str
    updated_at: float
    status: str = "added"  # "added", "updated", "unchanged", "deleted"


class KnowledgeUpdateManager:
    """知识更新管理器"""

    def __init__(self, vector_db_path: Optional[str] = None):
        self.vector_db_path = vector_db_path or settings.VECTOR_DB_PATH
        self.registry_path = os.path.join(self.vector_db_path, "update_registry.json")
        self.bm25_index_path = BM25Retriever.default_index_path(self.vector_db_path)
        self.bm25_corpus_path = BM25Retriever.default_corpus_path(self.vector_db_path)
        self.registry: Dict[str, UpdateRecord] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        if not os.path.exists(self.registry_path):
            return
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, rec in data.items():
                self.registry[key] = UpdateRecord(
                    source=rec.get("source", ""),
                    record_id=rec.get("record_id", ""),
                    content_hash=rec.get("content_hash", ""),
                    updated_at=rec.get("updated_at", 0),
                    status=rec.get("status", "added"),
                )
            logger.info(f"✅ 更新注册表已加载: {len(self.registry)} 条记录")
        except Exception as e:
            logger.warning(f"⚠️  加载更新注册表失败: {e}")

    def _save_registry(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        data = {}
        for key, rec in self.registry.items():
            data[key] = {
                "source": rec.source,
                "record_id": rec.record_id,
                "content_hash": rec.content_hash,
                "updated_at": rec.updated_at,
                "status": rec.status,
            }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    def check_updates(
        self,
        documents: List[Dict[str, Any]],
        source: str = "unknown",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        检测文档变更

        Returns:
            {"added": [...], "updated": [...], "unchanged": [...]}
        """
        result = {"added": [], "updated": [], "unchanged": []}

        for doc in documents:
            content = doc.get("content", "") or doc.get("page_content", "")
            record_id = doc.get("record_id", "") or doc.get("metadata", {}).get("record_id", "")
            content_hash = self._compute_hash(content)
            registry_key = f"{source}::{record_id}" if record_id else f"{source}::{content_hash[:16]}"

            if registry_key not in self.registry:
                result["added"].append(doc)
                self.registry[registry_key] = UpdateRecord(
                    source=source,
                    record_id=record_id,
                    content_hash=content_hash,
                    updated_at=time.time(),
                    status="added",
                )
            elif self.registry[registry_key].content_hash != content_hash:
                result["updated"].append(doc)
                self.registry[registry_key] = UpdateRecord(
                    source=source,
                    record_id=record_id,
                    content_hash=content_hash,
                    updated_at=time.time(),
                    status="updated",
                )
            else:
                result["unchanged"].append(doc)

        self._save_registry()

        logger.info(
            f"📋 知识更新检测: 新增={len(result['added'])}, "
            f"更新={len(result['updated'])}, 未变={len(result['unchanged'])}"
        )
        return result

    @staticmethod
    def _normalize_document(
        doc: Dict[str, Any],
        *,
        source: str,
    ) -> Dict[str, Any]:
        content = doc.get("content", "") or doc.get("page_content", "")
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()

        metadata = dict(doc.get("metadata", {}) or {})
        if doc.get("record_id") and not metadata.get("record_id"):
            metadata["record_id"] = doc.get("record_id")
        if doc.get("source") and not metadata.get("source"):
            metadata["source"] = doc.get("source")
        metadata.setdefault("source", source)
        return {
            "content": content,
            "metadata": metadata,
            "record_id": metadata.get("record_id", ""),
        }

    @staticmethod
    def _vector_id(source: str, record_id: str, content_hash: str) -> str:
        raw = f"{source}::{record_id or content_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _load_bm25_corpus_records(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        documents, metadata = BM25Retriever.load_corpus(self.bm25_corpus_path)
        for index, content in enumerate(documents):
            meta = metadata[index] if index < len(metadata) else {}
            vector_id = meta.get("vector_id")
            if not vector_id:
                content_hash = self._compute_hash(content)
                vector_id = self._vector_id(
                    meta.get("source", "unknown"),
                    meta.get("record_id", ""),
                    content_hash,
                )
                meta = {**meta, "vector_id": vector_id, "content_hash": content_hash}
            records[vector_id] = {"content": content, "metadata": meta}
        return records

    def _save_bm25_corpus_records(self, records: Dict[str, Dict[str, Any]]) -> int:
        ordered = sorted(
            records.values(),
            key=lambda item: (
                str(item.get("metadata", {}).get("source", "")),
                str(item.get("metadata", {}).get("record_id", "")),
                str(item.get("metadata", {}).get("vector_id", "")),
            ),
        )
        documents = [item["content"] for item in ordered]
        metadata = [item.get("metadata", {}) for item in ordered]
        BM25Retriever.save_corpus(documents, metadata, self.bm25_corpus_path)
        retriever = BM25Retriever(
            index_path=self.bm25_index_path,
            corpus_path=self.bm25_corpus_path,
        )
        retriever.build_index(documents, metadata)
        return len(documents)

    def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        *,
        source: str = "unknown",
        vector_store: Any = None,
    ) -> Dict[str, Any]:
        normalized = []
        for doc in documents:
            item = self._normalize_document(doc, source=source)
            if item["content"]:
                normalized.append(item)
        if not normalized:
            return {
                "added_count": 0,
                "updated_count": 0,
                "unchanged_count": 0,
                "stored_count": 0,
                "bm25_doc_count": 0,
            }

        changes = self.check_updates(normalized, source=source)
        changed_docs = list(changes["added"]) + list(changes["updated"])
        if not changed_docs:
            return {
                "added_count": 0,
                "updated_count": 0,
                "unchanged_count": len(changes["unchanged"]),
                "stored_count": 0,
                "bm25_doc_count": len(self._load_bm25_corpus_records()),
            }

        if vector_store is None:
            raise ValueError("vector_store is required for document ingestion")

        vector_payloads = []
        now = time.time()
        for doc in changed_docs:
            content = doc["content"]
            metadata = dict(doc.get("metadata", {}) or {})
            record_id = doc.get("record_id", "") or metadata.get("record_id", "")
            content_hash = self._compute_hash(content)
            vector_id = self._vector_id(source, record_id, content_hash)
            metadata.update(
                {
                    "source": metadata.get("source", source),
                    "record_id": record_id,
                    "content_hash": content_hash,
                    "vector_id": vector_id,
                    "updated_at_ts": now,
                }
            )
            vector_payloads.append(
                {
                    "id": vector_id,
                    "content": content,
                    "metadata": metadata,
                }
            )

        ids = [item["id"] for item in vector_payloads]
        try:
            vector_store.delete(ids=ids)
        except Exception:
            pass

        vector_store.add_texts(
            [item["content"] for item in vector_payloads],
            [item["metadata"] for item in vector_payloads],
            ids=ids,
        )

        corpus_records = self._load_bm25_corpus_records()
        for item in vector_payloads:
            corpus_records[item["id"]] = {
                "content": item["content"],
                "metadata": item["metadata"],
            }
        bm25_doc_count = self._save_bm25_corpus_records(corpus_records)

        logger.info(
            "✅ 知识入库完成: added=%s, updated=%s, unchanged=%s, stored=%s, bm25_docs=%s",
            len(changes["added"]),
            len(changes["updated"]),
            len(changes["unchanged"]),
            len(vector_payloads),
            bm25_doc_count,
        )
        return {
            "added_count": len(changes["added"]),
            "updated_count": len(changes["updated"]),
            "unchanged_count": len(changes["unchanged"]),
            "stored_count": len(vector_payloads),
            "bm25_doc_count": bm25_doc_count,
        }

    def fetch_nasa_apod(self, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取 NASA APOD（每日天文图片）数据"""
        key = api_key or settings.NASA_API_KEY
        if not key:
            logger.warning("⚠️  NASA_API_KEY 未配置")
            return []

        try:
            import requests
            resp = requests.get(
                settings.NASA_APOD_URL,
                params={"api_key": key, "count": 5},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            documents = []
            for item in data if isinstance(data, list) else [data]:
                content_parts = []
                if item.get("title"):
                    content_parts.append(f"标题: {item['title']}")
                if item.get("explanation"):
                    content_parts.append(f"说明: {item['explanation']}")
                if item.get("date"):
                    content_parts.append(f"日期: {item['date']}")
                if item.get("media_type"):
                    content_parts.append(f"类型: {item['media_type']}")

                if content_parts:
                    documents.append({
                        "content": "\n".join(content_parts),
                        "metadata": {
                            "source": "nasa_apod",
                            "record_id": f"apod_{item.get('date', 'unknown')}",
                            "data_source": "nasa",
                            "credibility": 1.0,
                            "observation_date": item.get("date"),
                            "is_time_sensitive": True,
                        },
                    })

            logger.info(f"✅ NASA APOD 获取成功: {len(documents)} 条")
            return documents

        except Exception as e:
            logger.error(f"❌ NASA APOD 获取失败: {e}")
            return []

    def fetch_nasa_neo(self, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取 NASA NEO（近地天体）数据"""
        key = api_key or settings.NASA_API_KEY
        if not key:
            return []

        try:
            import requests
            resp = requests.get(
                settings.NASA_NEO_URL,
                params={"api_key": key, "count": 5},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            documents = []
            near_earth_objects = data.get("near_earth_objects", {})
            for date, objects in near_earth_objects.items():
                for obj in objects[:3]:
                    content_parts = [f"近地天体: {obj.get('name', 'Unknown')}"]
                    if obj.get("estimated_diameter"):
                        km = obj["estimated_diameter"].get("kilometers", {})
                        min_d = km.get("estimated_diameter_min", 0)
                        max_d = km.get("estimated_diameter_max", 0)
                        content_parts.append(f"估计直径: {min_d:.3f} - {max_d:.3f} km")
                    if obj.get("is_potentially_hazardous_asteroid"):
                        content_parts.append("⚠️ 潜在危险小行星")
                    content_parts.append(f"日期: {date}")
                    content_parts.append(f"NASA ID: {obj.get('neo_reference_id', '')}")

                    documents.append({
                        "content": "\n".join(content_parts),
                        "metadata": {
                            "source": "nasa_neo",
                            "record_id": obj.get("neo_reference_id", ""),
                            "data_source": "nasa",
                            "credibility": 1.0,
                            "observation_date": date,
                            "is_time_sensitive": True,
                            "doc_type": "observation_record",
                        },
                    })

            logger.info(f"✅ NASA NEO 获取成功: {len(documents)} 条")
            return documents

        except Exception as e:
            logger.error(f"❌ NASA NEO 获取失败: {e}")
            return []

    def get_update_stats(self) -> Dict[str, Any]:
        stats = {"total_records": len(self.registry)}
        for status in ["added", "updated", "unchanged", "deleted"]:
            stats[f"{status}_count"] = sum(
                1 for r in self.registry.values() if r.status == status
            )
        return stats
