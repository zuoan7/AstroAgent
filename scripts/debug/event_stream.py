#!/usr/bin/env python3
"""调试事件流 - 查看API返回的实际事件格式"""

from src.agent import AstroAgent
from pathlib import Path
import asyncio

async def main():
    print("="*70)
    print("调试API事件流 - 查看完整事件格式")
    print("="*70)
    
    # 初始化agent
    agent = AstroAgent()
    
    # 测试图片路径
    image_path = str(Path(__file__).parent / "image.jpg")
    
    query = "这张图片里有什么天文现象？"
    print(f"\n查询: {query}")
    print(f"图片: {image_path}")
    print("-"*70)
    
    event_count = 0
    async for event in agent.generate_events(query, image_path=image_path):
        event_count += 1
        print(f"\n事件 #{event_count}:")
        print(f"  完整内容: {event}")
        print(f"  类型: {event.get('type')}")
        print(f"  所有键: {list(event.keys())}")
    
    print("\n" + "="*70)
    print(f"总共接收 {event_count} 个事件")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
