#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 BM25 索引脚本
优先从本地 BM25 语料文件构建，缺失时再从 Chroma 一次性迁移
"""
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from src.rag.bm25_retriever import BM25Retriever, build_bm25_index_from_chroma
from src.core.config import settings
from src.core.logger import logger


def build_bm25_index(
    vector_db_path: str = None,
    collection_name: str = "astronomy_rag",
    output_path: str = None
):
    """
    构建 BM25 索引

    Args:
        vector_db_path: Chroma 向量库路径
        collection_name: collection 名称
        output_path: BM25 索引输出路径
    """
    if vector_db_path is None:
        vector_db_path = settings.VECTOR_DB_PATH

    if output_path is None:
        output_path = BM25Retriever.default_index_path(vector_db_path)
    corpus_path = BM25Retriever.default_corpus_path(vector_db_path)

    logger.info("=" * 60)
    logger.info("开始构建 BM25 索引...")
    logger.info("=" * 60)

    try:
        if os.path.exists(corpus_path):
            documents, metadata = BM25Retriever.load_corpus(corpus_path)
            if documents:
                logger.info(f"从本地 BM25 语料文件获取到 {len(documents)} 个文档")
                retriever = BM25Retriever(index_path=output_path, corpus_path=corpus_path)
                retriever.build_index(documents, metadata)
                logger.info("=" * 60)
                logger.info(f"✅ BM25 索引构建成功: {len(documents)} 个文档")
                logger.info(f"📁 索引保存位置: {output_path}")
                logger.info("=" * 60)
                return

            logger.warning("⚠️  BM25 语料文件存在但为空，将回退到 Chroma 迁移")

        # 连接 Chroma
        embeddings = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )

        chroma_db = Chroma(
            embedding_function=embeddings,
            collection_name=collection_name,
            persist_directory=vector_db_path,
        )

        build_bm25_index_from_chroma(
            chroma_db,
            output_path=output_path,
            corpus_path=corpus_path,
        )

        logger.info("=" * 60)
        logger.info("✅ BM25 索引构建成功")
        logger.info(f"📁 索引保存位置: {output_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ 构建 BM25 索引失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建 BM25 索引")
    parser.add_argument("--vector-db-path", default=settings.VECTOR_DB_PATH,
                        help="Chroma 向量库路径")
    parser.add_argument("--collection-name", default="astronomy_rag",
                        help="Collection 名称")
    parser.add_argument("--output-path", default=None,
                        help="BM25 索引输出路径")

    args = parser.parse_args()

    build_bm25_index(
        vector_db_path=args.vector_db_path,
        collection_name=args.collection_name,
        output_path=args.output_path
    )
