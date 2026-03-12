#!/usr/bin/env python3
"""
混合检索测试脚本
测试 RAG 的混合检索功能（向量检索 + BM25）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import logger


def test_bm25_retriever():
    """测试 BM25 检索器"""
    from rag.bm25_retriever import BM25Retriever

    print("\n" + "=" * 60)
    print("  测试 1: BM25 检索器")
    print("=" * 60)

    # 创建测试数据
    documents = [
        '什么是黑洞？黑洞是恒星坍缩形成的致密天体',
        '如何观测流星雨？选择光污染少的地方，避开月光',
        '望远镜有哪些类型？折射式、反射式、折返式',
        '月相对观测有什么影响？满月光害严重，不适合深空',
        '火星冲日是什么？火星与太阳黄经相差180度的现象',
        '春季星空有哪些星座？猎户座、金牛座、双子座',
        '如何拍摄星空？需要三脚架、长曝光、高ISO'
    ]

    metadata = [{'source': f'test{i}.txt', 'record_id': str(i)} for i in range(len(documents))]

    # 创建临时索引
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_index = os.path.join(temp_dir, 'test_bm25.pkl')

    retriever = BM25Retriever(index_path=temp_index)
    retriever.build_index(documents, metadata)

    # 测试查询
    test_queries = [
        ('黑洞是什么', '黑洞'),
        ('如何观测流星', '流星'),
        ('望远镜类型', '望远镜'),
        ('拍摄星空', '星空'),
    ]

    for query, keyword in test_queries:
        results = retriever.search(query, top_k=2)
        print(f"\n查询: {query}")
        if results:
            for r in results[:2]:
                print(f"  Score: {r['score']:.2f}, Doc: {r['document'][:40]}...")
        else:
            print("  无结果")

    print("\n✅ BM25 检索器测试完成")


def test_hybrid_retriever():
    """测试混合检索器"""
    print("\n" + "=" * 60)
    print("  测试 2: 混合检索器")
    print("=" * 60)

    # 检查向量库
    vector_db_path = "./vector_db"
    bm25_index_path = os.path.join(vector_db_path, "bm25_index.pkl")

    if not os.path.exists(vector_db_path):
        print(f"⚠️  向量库不存在: {vector_db_path}")
        print("请先运行: python -m rag.offline_index 构建向量库")
        return

    if not os.path.exists(bm25_index_path):
        print(f"⚠️  BM25 索引不存在: {bm25_index_path}")
        print("请先运行: python -m rag.build_bm25_index 构建 BM25 索引")
        return

    try:
        from rag.online_retriever import OnlineRetriever

        retriever = OnlineRetriever(
            use_hybrid=True,
            vector_weight=0.5,
            bm25_weight=0.5
        )

        test_queries = [
            "如何观测流星雨",
            "黑洞是什么",
            "望远镜类型",
            "春季星空"
        ]

        for query in test_queries:
            print(f"\n查询: {query}")
            context = retriever.get_relevant_context(query, top_k=3)
            if context:
                print(f"  返回上下文长度: {len(context)} 字符")
                print(f"  前100字: {context[:100]}...")
            else:
                print("  无结果")

        print("\n✅ 混合检索器测试完成")

    except Exception as e:
        print(f"❌ 混合检索器测试失败: {e}")
        import traceback
        traceback.print_exc()


def build_bm25_index():
    """构建 BM25 索引"""
    print("\n" + "=" * 60)
    print("  构建 BM25 索引")
    print("=" * 60)

    try:
        from rag.build_bm25_index import build_bm25_index
        build_bm25_index()
        print("\n✅ BM25 索引构建完成")
    except Exception as e:
        print(f"❌ BM25 索引构建失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("\n" + "🔍" * 30)
    print("  AstroAgent 混合检索测试")
    print("🔍" * 30)

    # 测试 BM25 检索器
    test_bm25_retriever()

    # 检查是否需要构建索引
    vector_db_path = "./vector_db"
    bm25_index_path = os.path.join(vector_db_path, "bm25_index.pkl")

    if not os.path.exists(bm25_index_path):
        print("\n⚠️  BM25 索引不存在，是否构建？")
        print("运行: python -m rag.build_bm25_index")

    # 测试混合检索
    test_hybrid_retriever()

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)
    print("""
📝 使用说明:
1. 首先确保已构建向量库: python -m rag.offline_index
2. 然后构建 BM25 索引: python -m rag.build_bm25_index
3. 混合检索将自动启用，默认权重: 向量 50% + BM25 50%
4. 可通过参数调整权重:
   retriever = OnlineRetriever(
       use_hybrid=True,
       vector_weight=0.6,  # 向量权重
       bm25_weight=0.4    # BM25 权重
   )
""")


if __name__ == "__main__":
    main()
