from agent import AstroAgent
import time


def main():
    """简单的对话模型测试"""
    print("=== 天文知识助手 ===")
    print("输入 '退出' 或 'exit' 结束对话")
    print("=" * 50)
    
    # 创建Agent实例
    agent = AstroAgent()
    
    while True:
        # 获取用户输入
        user_input = input("用户: ")
        
        # 检查是否退出
        if user_input.lower() in ['退出', 'exit']:
            print("助手: 再见！")
            break
        
        # 生成响应
        print("助手: ", end="", flush=True)
        start_time = time.time()
        
        # 流式输出响应
        full_response = ""
        for chunk in agent.generate_response(user_input):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        end_time = time.time()
        print()
        print(f"响应时间: {end_time - start_time:.2f}秒")
        print("=" * 50)


if __name__ == "__main__":
    main()
