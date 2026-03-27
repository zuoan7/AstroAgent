#!/usr/bin/env python3
"""
测试agent的提示词模板是否正确工作
"""

from agent import AstroAgent

def test_agent_prompt():

    agent = AstroAgent()
    print("✅ Agent初始化成功")
    
    while True:
        # 获取用户输入
        test_query = input("\n请输入测试查询（输入exit退出）：")
        if test_query.lower() == 'exit':
            break
        
        # 生成响应
        response = agent.generate_response(test_query)
        for chunk in response:
            print(chunk, end="", flush=True)
        print("="*20)
        
       

if __name__ == "__main__":
    test_agent_prompt()
