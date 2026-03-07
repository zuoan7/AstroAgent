from agent import AstroAgent
import time
import traceback  # 新增：打印详细异常


def test_basic_query():
    """测试基本查询功能"""
    print("=== 测试基本查询功能 ===")
    try:
        agent = AstroAgent()

        # 测试基本天文问题
        query = "什么是黑洞？"
        print(f"用户: {query}")
        print("助手:", end=" ", flush=True)  # 新增flush=True，确保实时输出

        start_time = time.time()
        full_response = ""
        # 迭代生成器实现流式输出（核心逻辑）
        for chunk in agent.generate_response(query):
            print(chunk, end="", flush=True)  # flush=True是流式输出的关键
            full_response += chunk
        print()  # 回答结束后换行
        end_time = time.time()

        print(f"✅ 基本查询测试完成 | 响应时间: {end_time - start_time:.2f}秒")
        print("-" * 50)  # 分隔线，增强可读性
    except Exception as e:
        print(f"\n❌ 基本查询测试失败：{str(e)}")
        traceback.print_exc()
        print("-" * 50)


def test_memory_function():
    """测试记忆功能"""
    print("=== 测试记忆功能 ===")
    try:
        agent = AstroAgent()

        # 第一个问题
        query1 = "太阳系有哪些行星？"
        print(f"用户: {query1}")
        print("助手:", end=" ", flush=True)
        full_response1 = ""
        for chunk in agent.generate_response(query1):
            print(chunk, end="", flush=True)
            full_response1 += chunk
        print()

        # 模拟等待（避免请求过快触发限流）
        time.sleep(2)

        # 第二个问题，参考前一个问题的上下文
        query2 = "最大的那个行星有多大？"
        print(f"用户: {query2}")
        print("助手:", end=" ", flush=True)
        full_response2 = ""
        for chunk in agent.generate_response(query2):
            print(chunk, end="", flush=True)
            full_response2 += chunk
        print()

        print("✅ 记忆功能测试完成")
        print("-" * 50)
    except Exception as e:
        print(f"\n❌ 记忆功能测试失败：{str(e)}")
        traceback.print_exc()
        print("-" * 50)


def test_rag_function():
    """测试RAG功能（适配新版RAGSystem）"""
    print("=== 测试RAG功能 ===")
    try:
        # 初始化Agent时，会自动初始化RAGSystem并加载data文件夹中的文档
        agent = AstroAgent()
        print("✅ AstroAgent初始化完成，RAG系统已加载data文件夹中的文档")

        # 测试与RAG知识库相关的问题（匹配data文件夹中的天文文档）
        query = "银河系的结构是什么样的？"
        print(f"用户: {query}")
        print("助手:", end=" ", flush=True)

        start_time = time.time()
        full_response = ""
        for chunk in agent.generate_response(query):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()
        end_time = time.time()

        print(f"✅ RAG功能测试完成 | 响应时间: {end_time - start_time:.2f}秒")
        print("-" * 50)
    except Exception as e:
        print(f"\n❌ RAG功能测试失败：{str(e)}")
        traceback.print_exc()
        print("-" * 50)


# ========== 移除全局的answer方法（它属于AstroAgent类，不该出现在测试脚本中） ==========

if __name__ == "__main__":
    print("===== 开始执行AstroAgent全量测试 =====")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    # 按顺序执行测试（可注释掉不需要的测试）
    try:
        # 优先测试RAG功能（核心）
        test_rag_function()

        # 测试基本查询
        test_basic_query()

        # 测试记忆功能
        test_memory_function()

        print("\n===== 所有测试执行完成 =====")
    except Exception as e:
        print(f"\n===== 测试执行异常终止 =====")
        print(f"错误原因: {str(e)}")
        traceback.print_exc()