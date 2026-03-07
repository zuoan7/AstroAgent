#!/usr/bin/env python3
"""
简化的天文工具类测试，只测试不需要网络下载的功能
"""

from datetime import datetime
from astronomy_tools import AstronomyTools

def test_coordinate_transformation():
    """测试天体坐标转换"""
    print("测试天体坐标转换...")
    tools = AstronomyTools()
    
    # 测试ICRS到FK5的转换
    ra = 10.68458
    dec = 41.26917
    try:
        transformed = tools.coordinate_transformation(ra, dec, target_system='fk5')
        print(f"ICRS到FK5转换: 原始坐标 (RA={ra}小时, Dec={dec}度) -> 转换后 (RA={transformed['ra']:.5f}小时, Dec={transformed['dec']:.5f}度)")
    except Exception as e:
        print(f"坐标转换测试失败: {e}")
    
    print("坐标转换测试完成\n")

def test_rise_set_times():
    """测试天体升起落下时间"""
    print("测试天体升起落下时间...")
    tools = AstronomyTools()
    
    # 测试北京地区的太阳升起落下时间
    beijing_lat = 39.9
    beijing_lon = 116.4
    try:
        sun_times = tools.get_rise_set_times('sun', beijing_lat, beijing_lon)
        if 'error' in sun_times:
            print(f"太阳升起落下时间错误: {sun_times['error']}")
        else:
            print(f"北京太阳升起时间: {sun_times['rise_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"北京太阳落下时间: {sun_times['set_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"太阳升起落下时间测试失败: {e}")
    
    # 测试北京地区的月亮升起落下时间
    try:
        moon_times = tools.get_rise_set_times('moon', beijing_lat, beijing_lon)
        if 'error' in moon_times:
            print(f"月亮升起落下时间错误: {moon_times['error']}")
        else:
            print(f"北京月亮升起时间: {moon_times['rise_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"北京月亮落下时间: {moon_times['set_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"月亮升起落下时间测试失败: {e}")
    
    print("升起落下时间测试完成\n")

def test_current_sky_objects():
    """测试当前天空中的主要天体"""
    print("测试当前天空中的主要天体...")
    tools = AstronomyTools()
    
    # 测试北京地区的当前天空天体
    beijing_lat = 39.9
    beijing_lon = 116.4
    try:
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
    except Exception as e:
        print(f"当前天空天体测试失败: {e}")
    
    print("当前天空天体测试完成\n")

if __name__ == "__main__":
    print("开始测试天文工具类...\n")
    
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