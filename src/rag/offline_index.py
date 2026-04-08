#!/usr/bin/env python3
"""
离线索引构建：从 data/ 加载多种格式文档，分割、向量化并落盘到 Chroma。

支持格式：
- .txt / .md / .pdf：作为文本类文档加载
- .json：支持 dict 或 list[dict]，每条记录转为一个 Document
- .jsonl：每行一个 JSON 对象，每条记录转为一个 Document
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.core.logger import logger


def _safe_read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_to_text(obj: Any) -> str:
    """
    将 JSON 对象转为更利于检索的文本。
    优先将 dict 以 key: value 的方式展开；否则 fallback 为 pretty json。
    """
    if isinstance(obj, dict):
        parts: list[str] = []
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                vv = ", ".join([str(x) for x in v])
            else:
                vv = str(v)
            parts.append(f"{k}: {vv}")
        return "\n".join(parts) if parts else json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _iter_json_records(path: str) -> Iterable[tuple[str, Any]]:
    """
    返回 (record_id, record_obj)。
    - json: dict => 单记录；list => 多记录
    - jsonl: 每行一个 dict
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        data = json.loads(_safe_read_text(path))
        if isinstance(data, list):
            for i, item in enumerate(data):
                rid = None
                if isinstance(item, dict):
                    rid = item.get("id") or item.get("name")
                yield (str(rid) if rid is not None else str(i), item)
        else:
            rid = data.get("id") if isinstance(data, dict) else None
            yield (str(rid) if rid is not None else "0", data)
        return

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rid = None
                if isinstance(obj, dict):
                    rid = obj.get("id") or obj.get("name")
                yield (str(rid) if rid is not None else str(i), obj)
        return

    raise ValueError(f"Unsupported json format: {path}")


@dataclass
class OfflineIndexConfig:
    data_dir: str = "./data"
    vector_db_path: str = settings.VECTOR_DB_PATH
    collection_name: str = "astronomy_rag"
    chunk_size: int = 800
    chunk_overlap: int = 120
    batch_size: int = 64


class OfflineIndexer:
    def __init__(self, cfg: OfflineIndexConfig):
        self.cfg = cfg
        self.hash_file = os.path.join(cfg.vector_db_path, "document_hashes.txt")
        self.document_hashes: set[str] = set()

        self.embeddings = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

        os.makedirs(cfg.vector_db_path, exist_ok=True)
        self.db = Chroma(
            embedding_function=self.embeddings,
            collection_name=cfg.collection_name,
            persist_directory=cfg.vector_db_path,
        )

    def _load_hashes(self) -> None:
        if not os.path.exists(self.hash_file):
            return
        with open(self.hash_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.document_hashes.add(line)

    def _save_hashes(self) -> None:
        with open(self.hash_file, "w", encoding="utf-8") as f:
            for h in sorted(self.document_hashes):
                f.write(h + "\n")

    def _doc_hash(self, source: str, record_id: Optional[str], text: str) -> str:
        return _sha256(f"{source}\n{record_id or ''}\n{text}")

    def _load_file_documents(self, file_path: str) -> list[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        # 文本类：txt / md
        if ext in (".txt", ".md"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata = {**(d.metadata or {}), "source": filename, "format": ext[1:]}
            return docs

        # PDF
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for d in docs:
                d.metadata = {**(d.metadata or {}), "source": filename, "format": "pdf"}
            return docs

        # JSON / JSONL：每条记录一个 Document
        if ext in (".json", ".jsonl"):
            docs: list[Document] = []
            for rid, obj in _iter_json_records(file_path):
                text = _json_to_text(obj)
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "format": ext[1:],
                            "record_id": rid,
                        },
                    )
                )
            return docs

        return []

    def build_or_update(self) -> None:
        if not settings.RAG_ENABLED:
            logger.warning("⚠️  RAG_ENABLED=False，跳过离线索引构建")
            return

        self._load_hashes()

        data_dir = self.cfg.data_dir
        if not os.path.exists(data_dir):
            logger.warning(f"⚠️  data 目录不存在：{data_dir}")
            return

        supported = {".txt", ".md", ".pdf", ".json", ".jsonl"}
        files = [
            os.path.join(data_dir, fn)
            for fn in os.listdir(data_dir)
            if os.path.splitext(fn)[1].lower() in supported
        ]

        logger.info(f"=== 离线索引开始：扫描到 {len(files)} 个文件 ===")

        new_chunks: list[str] = []
        new_metas: list[dict[str, Any]] = []

        for fp in sorted(files):
            filename = os.path.basename(fp)
            try:
                docs = self._load_file_documents(fp)
                if not docs:
                    logger.warning(f"⚠️  未加载到文档：{filename}")
                    continue

                # 为每个原始 doc 计算 hash（基于内容 + record_id + source）
                # 若已存在则跳过该 doc
                filtered: list[Document] = []
                for d in docs:
                    source = d.metadata.get("source", filename)
                    record_id = d.metadata.get("record_id")
                    h = self._doc_hash(source, record_id, d.page_content)
                    if h in self.document_hashes:
                        continue
                    d.metadata = {**(d.metadata or {}), "doc_hash": h}
                    filtered.append(d)

                if not filtered:
                    logger.info(f"⏭️  跳过已索引文件：{filename}")
                    continue

                # 分割
                chunks = self.splitter.split_documents(filtered)
                logger.info(f"✅ 载入 {filename}：原始 {len(filtered)} 条，分割 {len(chunks)} 片段")

                for c in chunks:
                    new_chunks.append(c.page_content)
                    new_metas.append(c.metadata or {})

                # 记录 hash（按原始 doc hash 记）
                for d in filtered:
                    self.document_hashes.add(d.metadata["doc_hash"])

            except Exception as e:
                logger.error(f"❌ 处理文件失败：{filename} | {e}")
                continue

        if not new_chunks:
            logger.info("=== 离线索引结束：无新增文档 ===")
            self._save_hashes()
            return

        # 分批写入向量库
        logger.info(f"=== 写入向量库：新增 {len(new_chunks)} 个片段 ===")
        bs = self.cfg.batch_size
        for i in range(0, len(new_chunks), bs):
            batch_texts = new_chunks[i : i + bs]
            batch_metas = new_metas[i : i + bs]
            self.db.add_texts(batch_texts, batch_metas)
            logger.info(f"✅ 写入 batch {i//bs + 1}（{len(batch_texts)}）")

        self._save_hashes()
        logger.info("=== 离线索引完成 ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--vector-db-path", default=settings.VECTOR_DB_PATH)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    cfg = OfflineIndexConfig(
        data_dir=args.data_dir,
        vector_db_path=args.vector_db_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    OfflineIndexer(cfg).build_or_update()


if __name__ == "__main__":
    main()

