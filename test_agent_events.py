#!/usr/bin/env python3
"""
测试agent的天象预测工具
"""

from agent_langchain import AstroAgent

def test_agent_events():
    """测试agent的天象预测工具"""
    print("=== 测试agent的天象预测工具 ===")
    
    try:
        # 初始化agent
        agent = AstroAgent()
        print("✅ Agent初始化成功")
        
        # 测试获取今晚最佳观测目标
        test_query = "请告诉我今晚最佳观测目标"
        print(f"\n测试查询：{test_query}")
        print("\n响应：")
        
        # 生成响应
        response = agent.generate_response(test_query)
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
        
        # 测试获取未来一周天象
        test_query = "请告诉我未来一周的天象"
        print(f"\n测试查询：{test_query}")
        print("\n响应：")
        
        response = agent.generate_response(test_query)
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
        
        # 测试获取本月天象
        test_query = "请告诉我本月的天象"
        print(f"\n测试查询：{test_query}")
        print("\n响应：")
        
        response = agent.generate_response(test_query)
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
        
        # 测试获取指定月份天象
        test_query = "请告诉我2026年8月的天象"
        print(f"\n测试查询：{test_query}")
        print("\n响应：")
        
        response = agent.generate_response(test_query)
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
        
        print("\n✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_events()
