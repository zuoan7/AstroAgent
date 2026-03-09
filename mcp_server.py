#!/usr/bin/env python3
# MCP服务器，封装天文工具

import json
from astronomy_tools import AstronomyTools, AstronomyEventsPredictor

# 初始化工具类
tools = AstronomyTools()
events_predictor = AstronomyEventsPredictor()

# 工具字典
tool_map = {}

# 行星位置计算工具
def get_planet_position(planet_name: str, observation_time: str = None, latitude: float = None, longitude: float = None) -> str:
    """
    获取行星位置
    
    Args:
        planet_name: 行星名称，如 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'
        observation_time: 观测时间，可选
        latitude: 观测点纬度，可选
        longitude: 观测点经度，可选
    
    Returns:
        行星位置信息
    """
    try:
        return tools.get_planet_position(planet_name, observation_time, latitude, longitude)
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["get_planet_position"] = get_planet_position

# 天体坐标转换工具
def coordinate_transformation(ra: float, dec: float, epoch: str = "J2000", target_system: str = "fk5") -> str:
    """
    天体坐标转换
    
    Args:
        ra: 赤经，小时
        dec: 赤纬，度
        epoch: 历元，默认为J2000
        target_system: 目标坐标系，默认为fk5
    
    Returns:
        转换后的坐标信息
    """
    try:
        return tools.coordinate_transformation(ra, dec, epoch, target_system)
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["coordinate_transformation"] = coordinate_transformation

# 升起落下时间工具
def get_rise_set_times(body_name: str, latitude: float, longitude: float, date: str = None) -> str:
    """
    获取天体升起和落下时间
    
    Args:
        body_name: 天体名称，如 'sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn'
        latitude: 观测点纬度
        longitude: 观测点经度
        date: 日期，可选
    
    Returns:
        升起和落下时间信息
    """
    try:
        return tools.get_rise_set_times(body_name, latitude, longitude, date)
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["get_rise_set_times"] = get_rise_set_times

# 当前天空天体工具
def get_current_sky_objects(latitude: float, longitude: float, date: str = None) -> str:
    """
    获取当前天空中的主要天体
    
    Args:
        latitude: 观测点纬度
        longitude: 观测点经度
        date: 日期，可选
    
    Returns:
        当前天空可见天体信息
    """
    try:
        return tools.get_current_sky_objects(latitude, longitude, date)
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["get_current_sky_objects"] = get_current_sky_objects

# 天体基本信息工具
def get_astrophysical_object_info(object_name: str) -> str:
    """
    查询天体基本信息
    
    Args:
        object_name: 天体名称
    
    Returns:
        天体基本信息
    """
    try:
        return tools.get_astrophysical_object_info(object_name)
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["get_astrophysical_object_info"] = get_astrophysical_object_info

# 星系数据查询工具
def get_galaxy_data(galaxy_name: str) -> str:
    """
    星系数据查询
    
    Args:
        galaxy_name: 星系名称
    
    Returns:
        星系详细数据
    """
    try:
        return tools.get_galaxy_data(galaxy_name)
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["get_galaxy_data"] = get_galaxy_data

# NASA每日天文图工具
def get_nasa_apod(date: str = None, hd: bool = False) -> str:
    """
    获取NASA每日天文图
    
    Args:
        date: 日期，格式为YYYY-MM-DD，可选
        hd: 是否获取高清图像，可选
    
    Returns:
        NASA每日天文图信息
    """
    try:
        return tools.get_nasa_apod(date, hd)
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["get_nasa_apod"] = get_nasa_apod

# 近地天体数据工具
def get_neo_data(start_date: str = None, end_date: str = None, limit: int = 10) -> str:
    """
    获取近地天体数据
    
    Args:
        start_date: 开始日期，格式为YYYY-MM-DD，可选
        end_date: 结束日期，格式为YYYY-MM-DD，可选
        limit: 返回结果数量限制，可选
    
    Returns:
        近地天体数据
    """
    try:
        return tools.get_neo_data(start_date, end_date, limit)
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["get_neo_data"] = get_neo_data

# 今晚最佳观测目标工具
def get_tonight_best() -> str:
    """
    获取今晚最佳观测目标
    
    Returns:
        今晚最佳观测目标信息
    """
    try:
        return events_predictor.get_tonight_best()
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["get_tonight_best"] = get_tonight_best

# 未来一周天象工具
def get_weekly_events(start_date: str = None) -> str:
    """
    获取未来一周的天象
    
    Args:
        start_date: 起始日期，可选，默认为今天
    
    Returns:
        未来一周天象信息
    """
    try:
        return events_predictor.get_weekly_events(start_date)
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["get_weekly_events"] = get_weekly_events

# 未来一个月天象工具
def get_monthly_events(year: int = None, month: int = None) -> str:
    """
    获取未来一个月的天象
    
    Args:
        year: 年份，可选，默认为当前年
        month: 月份，可选，默认为下个月
    
    Returns:
        未来一个月天象信息
    """
    try:
        return events_predictor.get_monthly_events(year, month)
    except Exception as e:
        return f"错误：{str(e)}"
tool_map["get_monthly_events"] = get_monthly_events

if __name__ == "__main__":
    # 直接处理stdio输入输出
    import sys
    for line in sys.stdin:
        try:
            # 解析输入
            request = json.loads(line.strip())
            tool_name = request.get("tool")
            params = request.get("params", {})
            
            # 调用对应的工具函数
            if tool_name not in tool_map:
                response = {
                    "error": f"工具 {tool_name} 不存在"
                }
                print(json.dumps(response))
                sys.stdout.flush()
                continue
            
            # 调用工具函数
            try:
                result = tool_map[tool_name](**params)
                # 确保结果是字符串
                if not isinstance(result, str):
                    result = str(result)
            except Exception as e:
                result = f"错误：{str(e)}"
            
            # 返回结果
            response = {
                "result": result
            }
            print(json.dumps(response))
        except Exception as e:
            # 返回错误信息
            response = {
                "error": str(e)
            }
            print(json.dumps(response))
        sys.stdout.flush()
