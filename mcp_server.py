#!/usr/bin/env python3
# streamable_mcp_server.py - Streamable HTTP MCP服务器
# 使用FastMCP框架，完整支持标准MCP协议和HTTP传输

import os
import sys
import json
from typing import Optional, Union
from fastmcp import FastMCP
from astronomy_tools import AstronomyTools, AstronomyEventsPredictor

# ========== 初始化工具类（服务器启动时只初始化一次） ==========
print("🚀 正在初始化天文工具...")
try:
    tools = AstronomyTools()
    events_predictor = AstronomyEventsPredictor()
    print("✅ 天文工具初始化完成")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    sys.exit(1)

# ========== 创建FastMCP服务器实例 ==========
mcp = FastMCP(name="Astronomy Server")

# ========== 注册所有工具 ==========

@mcp.tool()
def get_planet_position(planet_name: str, observation_time: str = None, 
                       latitude: float = None, longitude: float = None) -> str:
    """
    获取行星位置
    
    Args:
        planet_name: 行星名称，如 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'
        observation_time: 观测时间，可选
        latitude: 观测点纬度，可选
        longitude: 观测点经度，可选
    """
    try:
        return tools.get_planet_position(planet_name, observation_time, latitude, longitude)
    except Exception as e:
        return f"错误：{str(e)}"

@mcp.tool()
def coordinate_transformation(ra: float, dec: float, epoch: str = "J2000", target_system: str = "fk5") -> str:
    """
    天体坐标转换
    
    Args:
        ra: 赤经，小时
        dec: 赤纬，度
        epoch: 历元，默认为J2000
        target_system: 目标坐标系，默认为fk5
    """
    try:
        return tools.coordinate_transformation(ra, dec, epoch, target_system)
    except Exception as e:
        return f"错误：{str(e)}"

@mcp.tool()
def get_rise_set_times(body_name: str, latitude: float, longitude: float, date: str = None) -> str:
    """
    获取天体升起和落下时间
    
    Args:
        body_name: 天体名称，如 'sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn'
        latitude: 观测点纬度
        longitude: 观测点经度
        date: 日期，可选，格式 YYYY-MM-DD
    """
    try:
        return tools.get_rise_set_times(body_name, latitude, longitude, date)
    except Exception as e:
        return f"错误：{str(e)}"

@mcp.tool()
def get_current_sky_objects(latitude: float, longitude: float, date: str = None) -> str:
    """
    获取当前天空中的主要天体
    
    Args:
        latitude: 观测点纬度
        longitude: 观测点经度
        date: 日期，可选，格式 YYYY-MM-DD
    """
    try:
        return tools.get_current_sky_objects(latitude, longitude, date)
    except Exception as e:
        return f"错误：{str(e)}"

@mcp.tool()
def get_astrophysical_object_info(object_name: str) -> str:
    """
    查询天体基本信息
    
    Args:
        object_name: 天体名称
    """
    try:
        return tools.get_astrophysical_object_info(object_name)
    except Exception as e:
        return f"错误：{str(e)}"

@mcp.tool()
def get_galaxy_data(galaxy_name: str) -> str:
    """
    星系数据查询
    
    Args:
        galaxy_name: 星系名称
    """
    try:
        return tools.get_galaxy_data(galaxy_name)
    except Exception as e:
        return f"错误：{str(e)}"

@mcp.tool()
def get_nasa_apod(date: str = None, hd: bool = False) -> str:
    """
    获取NASA每日天文图
    
    Args:
        date: 日期，格式为YYYY-MM-DD，可选
        hd: 是否获取高清图像，可选
    """
    try:
        return tools.get_nasa_apod(date, hd)
    except Exception as e:
        return f"错误：{str(e)}"

@mcp.tool()
def get_neo_data(start_date: str = None, end_date: str = None, limit: int = 10) -> str:
    """
    获取近地天体数据
    
    Args:
        start_date: 开始日期，格式为YYYY-MM-DD，可选
        end_date: 结束日期，格式为YYYY-MM-DD，可选
        limit: 返回结果数量限制，可选
    """
    try:
        return tools.get_neo_data(start_date, end_date, limit)
    except Exception as e:
        return f"错误：{str(e)}"

@mcp.tool()
def get_tonight_best() -> str:
    """
    获取今晚最佳观测目标
    """
    try:
        return events_predictor.get_tonight_best()
    except Exception as e:
        return f"错误：{str(e)}"

@mcp.tool()
def get_weekly_events(start_date: Optional[str] = None) -> str:
    """
    获取未来一周的天象 - 增强版，支持多种参数格式
    
    Args:
        start_date: 起始日期，可以是：
            - None: 使用当前日期
            - 字符串: "YYYY-MM-DD" 格式
            - JSON字符串: '{"start_date": "2026-03-09"}' 或 '{}'
    """
    try:
        processed_start_date = start_date

        # 处理字符串参数
        if isinstance(start_date, str):
            # 检查是否是JSON格式
            if start_date.strip().startswith('{'):
                try:
                    data = json.loads(start_date)
                    if isinstance(data, dict):
                        # 处理 {} 或 {"start_date": null} 的情况
                        if not data:  # 空字典 {}
                            processed_start_date = None
                        elif 'start_date' in data:
                            if data['start_date'] is None:
                                processed_start_date = None
                            else:
                                processed_start_date = data['start_date']
                except json.JSONDecodeError:
                    # 不是有效的JSON，可能是普通日期字符串
                    # 检查是否是有效的日期格式
                    if start_date.strip() and start_date != "null":
                        # 验证日期格式
                        try:
                            from datetime import datetime
                            datetime.strptime(start_date, "%Y-%m-%d")
                            processed_start_date = start_date
                        except ValueError:
                            # 无效的日期格式，使用None
                            processed_start_date = None
                    else:
                        processed_start_date = None
            elif start_date == "" or start_date == "null":
                processed_start_date = None
            else:
                # 普通字符串，验证日期格式
                try:
                    from datetime import datetime
                    datetime.strptime(start_date, "%Y-%m-%d")
                    processed_start_date = start_date
                except ValueError:
                    # 无效的日期格式，使用None
                    processed_start_date = None

        # 调用业务逻辑函数
        result = events_predictor.get_weekly_events(processed_start_date)
        return result
    except Exception as e:
        import traceback
        print(f"错误详情: {e}")
        print(traceback.format_exc())
        return f"错误：{str(e)}"

@mcp.tool()
def get_monthly_events(year: Optional[Union[int, str, dict]] = None, 
                       month: Optional[Union[int, str]] = None) -> str:
    """
    获取未来一个月的天象 - 增强版，支持多种参数格式
    
    Args:
        year: 年份，可以是：
            - None: 使用当前年份
            - 整数: 2026
            - 字符串: "2026" 或 '{"year": 2026, "month": 8}'
            - 字典: {"year": 2026, "month": 8}
        month: 月份，可以是整数、字符串或None
    """
    try:
        # --- 智能参数解析层 ---
        processed_year = None
        processed_month = None

        # 1. 解析 year 参数
        if year is None:
            processed_year = None
        elif isinstance(year, int):
            processed_year = year
        elif isinstance(year, dict):
            processed_year = year.get('year')
            # 如果 month 还没处理，尝试从同一字典中获取
            if month is None and 'month' in year:
                processed_month = year.get('month')
        elif isinstance(year, str):
            # 检查是否是JSON格式
            if year.strip().startswith('{'):
                try:
                    data = json.loads(year)
                    if isinstance(data, dict):
                        processed_year = data.get('year')
                        if month is None and 'month' in data:
                            processed_month = data.get('month')
                except json.JSONDecodeError:
                    # 不是有效的JSON，尝试直接转为整数
                    try:
                        processed_year = int(year)
                    except ValueError:
                        processed_year = None
            else:
                # 普通字符串，尝试转为整数
                try:
                    processed_year = int(year)
                except ValueError:
                    processed_year = None

        # 2. 解析 month 参数 (如果尚未被解析)
        if processed_month is None:
            if month is None:
                processed_month = None
            elif isinstance(month, int):
                processed_month = month
            elif isinstance(month, dict):
                processed_month = month.get('month')
            elif isinstance(month, str):
                if month.strip().startswith('{'):
                    try:
                        data = json.loads(month)
                        if isinstance(data, dict):
                            processed_month = data.get('month')
                    except json.JSONDecodeError:
                        try:
                            processed_month = int(month)
                        except ValueError:
                            processed_month = None
                else:
                    try:
                        processed_month = int(month)
                    except ValueError:
                        processed_month = None

        # 调用业务逻辑函数
        result = events_predictor.get_monthly_events(processed_year, processed_month)
        return result
    except Exception as e:
        import traceback
        print(f"错误详情: {e}")
        print(traceback.format_exc())
        return f"错误：{str(e)}"

# ========== 资源端点 ==========

@mcp.resource("status://server")
def get_server_status() -> str:
    """获取服务器状态"""
    return json.dumps({
        "status": "running",
        "mode": "streamable-http",
        "version": "1.0.0"
    }, ensure_ascii=False)

# ========== 提示模板 ==========

@mcp.prompt()
def observation_guide(target: str) -> str:
    """
    生成观测指南
    
    Args:
        target: 观测目标名称
    """
    return f"""
请帮我制定一个观测 {target} 的计划，包含以下内容：
1. 目标的基本信息（类型、距离、视星等）
2. 最佳观测时间（月份、时段）
3. 需要使用的望远镜/设备建议
4. 观测注意事项
5. 可以拍摄的天体特征
"""

# ========== 启动服务器 ==========
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌌 天文MCP服务器 - Streamable HTTP模式")
    print("="*60)
    
    # 使用环境变量或默认值设置主机和端口
    host = os.environ.get("FASTMCP_HOST", "0.0.0.0")
    port = int(os.environ.get("FASTMCP_PORT", "8000"))
    
    print(f"📡 监听地址: http://{host}:{port}/mcp/")
    print(f"📦 已注册工具: 12个")
    print(f"💡 按 Ctrl+C 停止服务器")
    print("="*60 + "\n")
    
    # 启动服务器
    try:
        mcp.run(transport="streamable-http", host=host, port=port)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        import traceback
        traceback.print_exc()