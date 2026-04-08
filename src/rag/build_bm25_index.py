#!/usr/bin/env python3
"""
构建 BM25 索引脚本
从 Chroma 向量库中提取文档并构建 BM25 索引
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from rag.bm25_retriever import BM25Retriever
from config import settings
from logger import logger


def build_bm25_index(
    vector_db_path: str = None,
    collection_name: str = "astronomy_rag",
    output_path: str = None
):
    """
    从 Chroma 向量库构建 BM25 索引

    Args:
        vector_db_path: Chroma 向量库路径
        collection_name: collection 名称
        output_path: BM25 索引输出路径
    """
    if vector_db_path is None:
        vector_db_path = settings.VECTOR_DB_PATH

    if output_path is None:
        output_path = os.path.join(vector_db_path, "bm25_index.pkl")

    logger.info("=" * 60)
    logger.info("开始构建 BM25 索引...")
    logger.info("=" * 60)

    try:
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

        # 获取所有文档
        results = chroma_db.get()

        documents = []
        metadata = []

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for i, doc in enumerate(docs):
            if doc:  # 跳过空文档
                documents.append(doc)
                metadata.append(metas[i] if metas and i < len(metas) else {})

        if not documents:
            logger.warning("⚠️  Chroma 中没有文档")
            return

        logger.info(f"从 Chroma 获取到 {len(documents)} 个文档")

        # 构建 BM25 索引
        retriever = BM25Retriever(index_path=output_path)
        retriever.build_index(documents, metadata)

        logger.info("=" * 60)
        logger.info(f"✅ BM25 索引构建成功: {len(documents)} 个文档")
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
