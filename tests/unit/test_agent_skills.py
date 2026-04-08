#!/usr/bin/env python3
"""
天文Agent自然语言查询测试样例
通过自然语言直接测试各个技能的使用效果
"""

from src.agent import AstroAgent


def test_agent_skills():
    """测试Agent的所有技能"""
    print("🚀 天文Agent自然语言查询测试样例")
    print("="*60)
    
    try:
        print("\n正在初始化Agent...")
        agent = AstroAgent()
        print("✅ Agent初始化成功")
        
        test_queries = [
            # 1. 天气查询相关
            "请查询北京今天的天气，适合观测吗？",
            "上海现在的天气怎么样？",
            "深圳明天的天气如何？",
            
            # 2. 观测计划生成相关
            "请帮我生成今天北京的观测计划",
            "明天上海前半夜的观测计划是什么？",
            "2026年3月20日广州的观测计划",
            
            # 3. 天象事件预报相关
            "未来一周有什么天象？",
            "这个月有什么重要的天文事件？",
            "2026年4月的天象预报",
            "最近有流星雨吗？",
            
            # 4. 深空天体观测指导相关
            "我想观测M31仙女座星系，需要注意什么？",
            "猎户座大星云怎么观测？使用双筒望远镜",
            "M42星云的观测建议，我在北京",
            "M33三角座星系观测指南",
            
            # 5. 近地天体追踪相关
            "最近有近地小行星飞掠吗？",
            "未来30天有哪些近地天体？",
            "本月直径超过100米的近地天体",
            "未来有哪些可观测的近地天体？",
            
            # 6. 天文摄影参数计算相关
            "我想用Sony A7R4拍摄M31，参数怎么设置？",
            "拍摄银河需要什么参数？用Canon EOS R5",
            "如何用Nikon Z7拍摄猎户座大星云？",
            "月球摄影参数建议",
            
            # 7. 天体位置计算相关
            "火星现在在什么位置？我在北京",
            "2026年3月13日晚上10点，上海能看到木星吗？",
            "土星在2026年4月1日晚上8点的位置",
            "金星现在的位置",
            
            # 8. 综合查询
            "今天晚上适合观测什么？",
            "我是新手，想入门天文观测，有什么建议？",
            "这个周末想出去观星，帮我规划一下",
        ]
        
        print(f"\n📋 准备了 {len(test_queries)} 个测试查询")
        print("="*60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n\n{'='*60}")
            print(f"测试 {i}/{len(test_queries)}")
            print(f"查询: {query}")
            print(f"{'='*60}")
            print("\n响应:")
            
            try:
                response = agent.generate_response(query)
                for chunk in response:
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"❌ 查询失败: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*60}")
            
            if i < len(test_queries):
                choice = input("\n按 Enter 继续下一个测试，输入 'q' 退出: ").strip()
                if choice.lower() == 'q':
                    print("\n👋 测试提前结束")
                    break
        
        print("\n✅ 所有测试完成!")
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def interactive_test():
    """交互式测试模式"""
    print("🚀 天文Agent交互式测试")
    print("="*60)
    
    try:
        print("\n正在初始化Agent...")
        agent = AstroAgent()
        print("✅ Agent初始化成功")
        
        print("\n💡 使用提示:")
        print("  - 输入任意天文问题进行测试")
        print("  - 输入数字 1-7 选择示例问题")
        print("  - 输入 'exit' 或 'quit' 退出")
        print("  - 输入 'help' 查看示例问题")
        print("="*60)
        
        examples = [
            "查询北京今天的天气",
            "生成今晚的观测计划",
            "未来一周有什么天象？",
            "M31仙女座星系怎么观测？",
            "最近有近地小行星吗？",
            "拍摄银河需要什么参数？",
            "木星现在的位置在哪里？",
        ]
        
        while True:
            query = input("\n请输入你的问题: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("\n👋 测试结束")
                break
            
            if query.lower() == 'help':
                print("\n📋 示例问题:")
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. {example}")
                continue
            
            if query.isdigit():
                choice = int(query)
                if 1 <= choice <= len(examples):
                    query = examples[choice - 1]
                    print(f"\n📋 已选择示例问题: {query}")
                else:
                    print(f"\n❌ 无效的数字，请输入 1-{len(examples)} 之间的数字，或输入 'help' 查看示例")
                    continue
            
            if not query:
                continue
            
            print("\n🤖 Agent响应:")
            try:
                response = agent.generate_response(query)
                for chunk in response:
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"❌ 查询失败: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("🚀 天文Agent测试工具")
    print("="*60)
    print("\n请选择测试模式:")
    print("  1. 自动化测试 - 运行预设的所有测试查询")
    print("  2. 交互式测试 - 自由输入问题进行测试")
    
    choice = input("\n请输入选项 (1或2，默认2): ").strip()
    
    if choice == "1":
        test_agent_skills()
    else:
        interactive_test()


if __name__ == "__main__":
    main()
