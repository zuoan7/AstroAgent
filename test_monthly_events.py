#!/usr/bin/env python3
# 测试月预测功能，验证不同月份返回不同的行星可见性信息

from astronomy_tools import AstronomyEventsPredictor

def test_monthly_events():
    """测试不同月份的月预测"""
    print("=== 测试月预测功能 ===")
    
    try:
        predictor = AstronomyEventsPredictor()
        print("✅ 天象预测器初始化成功")
        
        # 测试几个不同的月份
        test_months = [3, 6, 9, 12]  # 3月、6月、9月、12月
        
        for month in test_months:
            print(f"\n=== 测试 {month} 月预测 ===")
            result = predictor.get_monthly_events(2026, month)
            print(result)
            
        print("\n✅ 测试完成")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_monthly_events()
