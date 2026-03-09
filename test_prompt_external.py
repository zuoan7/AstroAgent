#!/usr/bin/env python3
# 测试外部prompt文件是否正常工作

from agent_langchain import AstroAgent

def test_agent_initialization():
    """测试Agent初始化，验证外部prompt文件是否正常加载"""
    print("=== 测试Agent初始化 ===")
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
        
        print("✅ 测试完成")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_initialization()
