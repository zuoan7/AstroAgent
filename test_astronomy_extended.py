#!/usr/bin/env python3
"""
测试天文工具类的扩展功能
"""

from datetime import datetime
from astronomy_tools import AstronomyTools

def test_astrophysical_object_info():
    """测试查询天体基本信息"""
    print("测试查询天体基本信息...")
    tools = AstronomyTools()
    
    # 测试查询仙女座星系
    andromeda_info = tools.get_astrophysical_object_info('Andromeda Galaxy')
    print(f"仙女座星系信息: {andromeda_info}")
    
    # 测试查询天狼星
    sirius_info = tools.get_astrophysical_object_info('Sirius')
    print(f"天狼星信息: {sirius_info}")
    
    # 测试查询猎户座大星云
    orion_info = tools.get_astrophysical_object_info('Orion Nebula')
    print(f"猎户座大星云信息: {orion_info}")
    
    print("天体基本信息查询测试完成\n")

def test_galaxy_data():
    """测试星系数据查询"""
    print("测试星系数据查询...")
    tools = AstronomyTools()
    
    # 测试查询仙女座星系
    andromeda_data = tools.get_galaxy_data('Andromeda Galaxy')
    print(f"仙女座星系数据: {andromeda_data}")
    
    # 测试查询银河系
    milky_way_data = tools.get_galaxy_data('Milky Way')
    print(f"银河系数据: {milky_way_data}")
    
    # 测试查询三角座星系
    triangulum_data = tools.get_galaxy_data('Triangulum Galaxy')
    print(f"三角座星系数据: {triangulum_data}")
    
    print("星系数据查询测试完成\n")

def test_nasa_apod():
    """测试获取NASA每日天文图"""
    print("测试获取NASA每日天文图...")
    tools = AstronomyTools()
    
    # 测试获取今天的天文图
    today = datetime.now().strftime('%Y-%m-%d')
    apod_today = tools.get_nasa_apod()
    print(f"今天的NASA天文图标题: {apod_today.get('title', '未知')}")
    print(f"今天的NASA天文图URL: {apod_today.get('url', '未知')}")
    
    # 测试获取指定日期的天文图
    apod_specific = tools.get_nasa_apod(date='2024-01-01')
    print(f"2024年1月1日的NASA天文图标题: {apod_specific.get('title', '未知')}")
    print(f"2024年1月1日的NASA天文图URL: {apod_specific.get('url', '未知')}")
    
    print("NASA每日天文图测试完成\n")

def test_neo_data():
    """测试获取近地天体数据"""
    print("测试获取近地天体数据...")
    tools = AstronomyTools()
    
    # 测试获取近地天体数据
    neo_data = tools.get_neo_data()
    if 'error' in neo_data:
        print(f"获取近地天体数据错误: {neo_data['error']}")
    else:
        # 统计近地天体数量
        total_neo = 0
        for date, neos in neo_data.get('near_earth_objects', {}).items():
            total_neo += len(neos)
        print(f"近地天体总数: {total_neo}")
        print(f"数据日期范围: {neo_data.get('links', {}).get('self', '未知')}")
    
    print("近地天体数据测试完成\n")

if __name__ == "__main__":
    print("开始测试天文工具类的扩展功能...\n")
    
    try:
        test_astrophysical_object_info()
    except Exception as e:
        print(f"测试天体基本信息查询时出错: {e}\n")
    
    try:
        test_galaxy_data()
    except Exception as e:
        print(f"测试星系数据查询时出错: {e}\n")
    
    try:
        test_nasa_apod()
    except Exception as e:
        print(f"测试NASA每日天文图时出错: {e}\n")
    
    try:
        test_neo_data()
    except Exception as e:
        print(f"测试近地天体数据时出错: {e}\n")
    
    print("天文工具类扩展功能测试完成！")