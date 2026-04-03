#!/usr/bin/env python3
"""
天文Agent技能测试样例
测试所有7个技能：
1. weather-lookup - 天气查询
2. observation-planner - 观测计划生成
3. celestial-events-forecast - 天象事件预报
4. deep-sky-observing-guide - 深空天体观测指导
5. neo-tracker - 近地天体追踪
6. astrophotography-calculator - 天文摄影参数计算
7. celestial-position-calculator - 天体位置计算
"""

from agent.skill_manager import SkillManager


def test_weather_lookup(skill_manager):
    """测试天气查询技能"""
    print("\n" + "="*60)
    print("测试 1: weather-lookup - 天气查询")
    print("="*60)
    
    test_cases = [
        ("北京", "查询北京天气"),
        ("上海", "查询上海天气"),
        ("深圳", "查询深圳天气"),
    ]
    
    for city, description in test_cases:
        print(f"\n📋 {description}")
        result = skill_manager.call_skill("weather-lookup", city=city)
        print(result)


def test_observation_planner(skill_manager):
    """测试观测计划生成技能"""
    print("\n" + "="*60)
    print("测试 2: observation-planner - 观测计划生成")
    print("="*60)
    
    test_cases = [
        {"date": "今天", "location": "北京", "duration": "整晚", "description": "今天北京整晚观测计划"},
        {"date": "明天", "location": "上海", "duration": "前半夜", "description": "明天上海前半夜观测计划"},
        {"date": "2026-03-20", "location": "广州", "duration": "后半夜", "description": "2026-03-20广州后半夜观测计划"},
        {"date": "今天", "location": None, "duration": None, "description": "今天通用观测计划"},
    ]
    
    for test_case in test_cases:
        print(f"\n📋 {test_case['description']}")
        result = skill_manager.call_skill(
            "observation-planner",
            date=test_case["date"],
            location=test_case["location"],
            duration=test_case["duration"]
        )
        print(result)


def test_celestial_events_forecast(skill_manager):
    """测试天象事件预报技能"""
    print("\n" + "="*60)
    print("测试 3: celestial-events-forecast - 天象事件预报")
    print("="*60)
    
    test_cases = [
        {"start_date": None, "end_date": None, "event_type": None, "description": "未来一周天象预报"},
        {"start_date": "2026-03-13", "end_date": "2026-03-20", "event_type": None, "description": "2026-03-13至2026-03-20天象预报"},
        {"start_date": "2026-04-01", "end_date": "2026-04-30", "event_type": None, "description": "2026年4月天象预报"},
        {"start_date": None, "end_date": None, "event_type": "流星雨", "description": "未来一周流星雨相关预报"},
    ]
    
    for test_case in test_cases:
        print(f"\n📋 {test_case['description']}")
        result = skill_manager.call_skill(
            "celestial-events-forecast",
            start_date=test_case["start_date"],
            end_date=test_case["end_date"],
            event_type=test_case["event_type"]
        )
        print(result)


def test_deep_sky_observing_guide(skill_manager):
    """测试深空天体观测指导技能"""
    print("\n" + "="*60)
    print("测试 4: deep-sky-observing-guide - 深空天体观测指导")
    print("="*60)
    
    test_cases = [
        {"target": "M31", "observer_location": "北京", "date": "今天", "equipment": "8寸望远镜", "description": "M31仙女座星系观测指导"},
        {"target": "猎户座大星云", "observer_location": "上海", "date": "明天", "equipment": "双筒望远镜", "description": "猎户座大星云观测指导"},
        {"target": "M42", "observer_location": "深圳", "date": "2026-03-20", "equipment": "裸眼", "description": "M42猎户座星云观测指导"},
        {"target": "M33", "observer_location": None, "date": None, "equipment": None, "description": "M33三角座星系通用观测指导"},
    ]
    
    for test_case in test_cases:
        print(f"\n📋 {test_case['description']}")
        result = skill_manager.call_skill(
            "deep-sky-observing-guide",
            target=test_case["target"],
            observer_location=test_case["observer_location"],
            date=test_case["date"],
            equipment=test_case["equipment"]
        )
        print(result)


def test_neo_tracker(skill_manager):
    """测试近地天体追踪技能"""
    print("\n" + "="*60)
    print("测试 5: neo-tracker - 近地天体追踪")
    print("="*60)
    
    test_cases = [
        {"time_range": "未来30天", "min_size": None, "max_distance": None, "observable_only": None, "description": "未来30天近地天体"},
        {"time_range": "本月", "min_size": 100, "max_distance": None, "observable_only": None, "description": "本月直径大于100米的近地天体"},
        {"time_range": "未来30天", "min_size": None, "max_distance": 5, "observable_only": None, "description": "未来30天距离小于5个地月距离的近地天体"},
        {"time_range": "未来30天", "min_size": None, "max_distance": None, "observable_only": True, "description": "未来30天可观测的近地天体"},
    ]
    
    for test_case in test_cases:
        print(f"\n📋 {test_case['description']}")
        result = skill_manager.call_skill(
            "neo-tracker",
            time_range=test_case["time_range"],
            min_size=test_case["min_size"],
            max_distance=test_case["max_distance"],
            observable_only=test_case["observable_only"]
        )
        print(result)


def test_astrophotography_calculator(skill_manager):
    """测试天文摄影参数计算技能"""
    print("\n" + "="*60)
    print("测试 6: astrophotography-calculator - 天文摄影参数计算")
    print("="*60)
    
    test_cases = [
        {"target": "M31仙女座星系", "camera": "Sony A7R4", "telescope": "8寸施卡望远镜", "mount": "EQ6-R赤道仪", "location": "北京", "date": "今天", "description": "M31摄影参数计算"},
        {"target": "银河", "camera": "Canon EOS R5", "telescope": "24mm广角镜头", "mount": "三脚架", "location": "上海", "date": "明天", "description": "银河摄影参数计算"},
        {"target": "猎户座大星云", "camera": "Nikon Z7", "telescope": "135mm定焦镜头", "mount": "星特朗AVX", "location": "深圳", "date": "2026-03-20", "description": "猎户座大星云摄影参数计算"},
        {"target": "月球", "camera": "Sony A7S3", "telescope": None, "mount": None, "location": None, "date": None, "description": "月球摄影参数计算"},
    ]
    
    for test_case in test_cases:
        print(f"\n📋 {test_case['description']}")
        result = skill_manager.call_skill(
            "astrophotography-calculator",
            target=test_case["target"],
            camera=test_case["camera"],
            telescope=test_case["telescope"],
            mount=test_case["mount"],
            location=test_case["location"],
            date=test_case["date"]
        )
        print(result)


def test_celestial_position_calculator(skill_manager):
    """测试天体位置计算技能"""
    print("\n" + "="*60)
    print("测试 7: celestial-position-calculator - 天体位置计算")
    print("="*60)
    
    test_cases = [
        {"target": "mars", "datetime": "2026-03-13 22:00", "location": "39.9,116.4", "output_format": "radec", "description": "2026-03-13 22:00北京火星位置"},
        {"target": "jupiter", "datetime": "2026-03-14 21:00", "location": "31.2,121.5", "output_format": "altaz", "description": "2026-03-14 21:00上海木星位置"},
        {"target": "saturn", "datetime": "2026-04-01 20:00", "location": "22.5,114.0", "output_format": "radec", "description": "2026-04-01 20:00深圳土星位置"},
        {"target": "venus", "datetime": None, "location": None, "output_format": None, "description": "当前时间北京金星位置"},
    ]
    
    for test_case in test_cases:
        print(f"\n📋 {test_case['description']}")
        result = skill_manager.call_skill(
            "celestial-position-calculator",
            target=test_case["target"],
            datetime=test_case["datetime"],
            location=test_case["location"],
            output_format=test_case["output_format"]
        )
        print(result)


def main():
    """主测试函数"""
    print("🚀 天文Agent技能测试样例")
    print("="*60)
    
    try:
        print("\n正在初始化技能管理器...")
        skill_manager = SkillManager()
        print("✅ 技能管理器初始化成功")
        
        print(f"\n📋 可用技能列表:")
        skills = skill_manager.list_skills()
        for skill_name, skill_desc in skills.items():
            print(f"  - {skill_name}: {skill_desc}")
        
        while True:
            print("\n" + "="*60)
            print("选择要测试的技能:")
            print("  1. weather-lookup - 天气查询")
            print("  2. observation-planner - 观测计划生成")
            print("  3. celestial-events-forecast - 天象事件预报")
            print("  4. deep-sky-observing-guide - 深空天体观测指导")
            print("  5. neo-tracker - 近地天体追踪")
            print("  6. astrophotography-calculator - 天文摄影参数计算")
            print("  7. celestial-position-calculator - 天体位置计算")
            print("  8. 运行所有测试")
            print("  0. 退出")
            
            choice = input("\n请输入选项 (0-8): ").strip()
            
            if choice == "0":
                print("\n👋 测试结束")
                break
            elif choice == "1":
                test_weather_lookup(skill_manager)
            elif choice == "2":
                test_observation_planner(skill_manager)
            elif choice == "3":
                test_celestial_events_forecast(skill_manager)
            elif choice == "4":
                test_deep_sky_observing_guide(skill_manager)
            elif choice == "5":
                test_neo_tracker(skill_manager)
            elif choice == "6":
                test_astrophotography_calculator(skill_manager)
            elif choice == "7":
                test_celestial_position_calculator(skill_manager)
            elif choice == "8":
                test_weather_lookup(skill_manager)
                test_observation_planner(skill_manager)
                test_celestial_events_forecast(skill_manager)
                test_deep_sky_observing_guide(skill_manager)
                test_neo_tracker(skill_manager)
                test_astrophotography_calculator(skill_manager)
                test_celestial_position_calculator(skill_manager)
            else:
                print("\n❌ 无效选项，请重新选择")
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
