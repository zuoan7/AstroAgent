#!/usr/bin/env python3
"""
测试基于LangChain的AstroAgent
"""
import time
from agent_langchain import AstroAgent


def test_basic_query():
    """测试基本查询功能"""
    print("=== 测试基本查询功能 ===")
    try:
        agent = AstroAgent()

        # 测试基本天文问题
        query = "什么是黑洞？"
        print(f"用户: {query}")
        print("助手:", end=" ", flush=True)

        start_time = time.time()
        full_response = ""
        for chunk in agent.generate_response(query):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()
        end_time = time.time()

        print(f"✅ 基本查询测试完成 | 响应时间: {end_time - start_time:.2f}秒")
        print("-" * 50)
    except Exception as e:
        print(f"\n❌ 基本查询测试失败：{e}")
        import traceback
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

        # 模拟等待
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
        print(f"\n❌ 记忆功能测试失败：{e}")
        import traceback
        traceback.print_exc()
        print("-" * 50)


def test_rag_function():
    """测试RAG功能"""
    print("=== 测试RAG功能 ===")
    try:
        agent = AstroAgent()

        # 测试与文档相关的问题
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
        print(f"\n❌ RAG功能测试失败：{e}")
        import traceback
        traceback.print_exc()
        print("-" * 50)


def main():
    """主函数"""
    print("===== 开始执行基于LangChain的AstroAgent测试 =====")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("--------------------------------------------------")

    test_rag_function()
    test_basic_query()
    test_memory_function()

    print("===== 所有测试执行完成 =====")


if __name__ == "__main__":
    main()
