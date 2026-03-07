#!/usr/bin/env python3
"""
测试天文工具类的功能
"""

from datetime import datetime
from astronomy_tools import AstronomyTools

def test_planet_position():
    """测试行星位置计算"""
    print("测试行星位置计算...")
    tools = AstronomyTools()
    
    # 测试当前时间的火星位置
    mars_position = tools.get_planet_position('mars')
    print(f"火星当前位置: 赤经={mars_position['ra']:.4f}小时, 赤纬={mars_position['dec']:.4f}度, 距离={mars_position['distance_au']:.4f}天文单位")
    
    # 测试指定时间的木星位置
    test_time = datetime(2024, 1, 1, 0, 0, 0)
    jupiter_position = tools.get_planet_position('jupiter', test_time)
    print(f"2024年1月1日木星位置: 赤经={jupiter_position['ra']:.4f}小时, 赤纬={jupiter_position['dec']:.4f}度, 距离={jupiter_position['distance_au']:.4f}天文单位")
    
    # 测试指定地点的金星位置
    venus_position = tools.get_planet_position('venus', latitude=39.9, longitude=116.4)
    print(f"北京观测金星位置: 赤经={venus_position['ra']:.4f}小时, 赤纬={venus_position['dec']:.4f}度, 距离={venus_position['distance_au']:.4f}天文单位")
    
    print("行星位置计算测试完成\n")

def test_coordinate_transformation():
    """测试天体坐标转换"""
    print("测试天体坐标转换...")
    tools = AstronomyTools()
    
    # 测试ICRS到FK5的转换
    ra = 10.68458
    dec = 41.26917
    transformed = tools.coordinate_transformation(ra, dec, target_system='fk5')
    print(f"ICRS到FK5转换: 原始坐标 (RA={ra}小时, Dec={dec}度) -> 转换后 (RA={transformed['ra']:.5f}小时, Dec={transformed['dec']:.5f}度)")
    
    # 测试FK5到ICRS的转换（实际上是保持不变）
    transformed_back = tools.coordinate_transformation(transformed['ra'], transformed['dec'], target_system='icrs')
    print(f"FK5到ICRS转换: 原始坐标 (RA={transformed['ra']:.5f}小时, Dec={transformed['dec']:.5f}度) -> 转换后 (RA={transformed_back['ra']:.5f}小时, Dec={transformed_back['dec']:.5f}度)")
    
    print("坐标转换测试完成\n")

def test_rise_set_times():
    """测试天体升起落下时间"""
    print("测试天体升起落下时间...")
    tools = AstronomyTools()
    
    # 测试北京地区的太阳升起落下时间
    beijing_lat = 39.9
    beijing_lon = 116.4
    sun_times = tools.get_rise_set_times('sun', beijing_lat, beijing_lon)
    if 'error' in sun_times:
        print(f"太阳升起落下时间错误: {sun_times['error']}")
    else:
        print(f"北京太阳升起时间: {sun_times['rise_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"北京太阳落下时间: {sun_times['set_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试北京地区的月亮升起落下时间
    moon_times = tools.get_rise_set_times('moon', beijing_lat, beijing_lon)
    if 'error' in moon_times:
        print(f"月亮升起落下时间错误: {moon_times['error']}")
    else:
        print(f"北京月亮升起时间: {moon_times['rise_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"北京月亮落下时间: {moon_times['set_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试北京地区的火星升起落下时间
    mars_times = tools.get_rise_set_times('mars', beijing_lat, beijing_lon)
    if 'error' in mars_times:
        print(f"火星升起落下时间错误: {mars_times['error']}")
    else:
        print(f"北京火星升起时间: {mars_times['rise_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"北京火星落下时间: {mars_times['set_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("升起落下时间测试完成\n")

def test_current_sky_objects():
    """测试当前天空中的主要天体"""
    print("测试当前天空中的主要天体...")
    tools = AstronomyTools()
    
    # 测试北京地区的当前天空天体
    beijing_lat = 39.9
    beijing_lon = 116.4
    sky_objects = tools.get_current_sky_objects(beijing_lat, beijing_lon)
    
    for obj_name, obj_data in sky_objects.items():
        print(f"{obj_name}:")
        if 'error' in obj_data:
            print(f"  错误: {obj_data['error']}")
        elif 'ra' in obj_data:
            print(f"  赤经: {obj_data['ra']:.4f}小时")
            print(f"  赤纬: {obj_data['dec']:.4f}度")
            print(f"  距离: {obj_data['distance_au']:.4f}天文单位")
        elif 'rise_time' in obj_data:
            print(f"  升起时间: {obj_data['rise_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  落下时间: {obj_data['set_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("当前天空天体测试完成\n")

if __name__ == "__main__":
    print("开始测试天文工具类...\n")
    
    try:
        test_planet_position()
    except Exception as e:
        print(f"测试行星位置计算时出错: {e}\n")
    
    try:
        test_coordinate_transformation()
    except Exception as e:
        print(f"测试坐标转换时出错: {e}\n")
    
    try:
        test_rise_set_times()
    except Exception as e:
        print(f"测试升起落下时间时出错: {e}\n")
    
    try:
        test_current_sky_objects()
    except Exception as e:
        print(f"测试当前天空天体时出错: {e}\n")
    
    print("天文工具类测试完成！")