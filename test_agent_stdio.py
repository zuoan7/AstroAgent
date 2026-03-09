#!/usr/bin/env python3
# 测试使用stdio模式调用MCP服务器工具的功能

from agent_langchain import AstroAgent

def test_agent_stdio():
    """测试Agent使用stdio模式调用MCP服务器工具"""
    print("=== 测试Agent使用stdio模式调用MCP服务器工具 ===")
    
    try:
        agent = AstroAgent()
        print("✅ Agent初始化成功")
        
        # 测试一个简单的查询
        print("\n=== 测试简单查询 ===")
        query = "今晚能看到什么？"
        print(f"用户查询: {query}")
        
        response = agent.generate_response(query)
        print("助手回答:")
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
        
        # 测试行星位置查询
        print("\n=== 测试行星位置查询 ===")
        query = "火星现在的位置"
        print(f"用户查询: {query}")
        
        response = agent.generate_response(query)
        print("助手回答:")
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
        
        print("\n✅ 测试完成")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_stdio()
