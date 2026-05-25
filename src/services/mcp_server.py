#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# streamable_mcp_server.py - Streamable HTTP MCP服务器
# 使用FastMCP框架，完整支持标准MCP协议和HTTP传输
"""FastMCP 天文工具服务入口，注册天文计算、数据查询、天气、搜索和天象事件工具。"""

import json
import os
import sys
from typing import Optional, Union

from fastmcp import FastMCP

from src.utils.param_parser import ParamParser
from src.astronomy import (
    CelestialDatabaseService,
    EphemerisManager,
    EventsPredictor,
    NASAAPIService,
    PlanetaryCalculator,
    SearchService,
    WeatherService,
)
from src.core.config import settings
from src.core.errors import AgentError, ErrorCode, ErrorHandler, safe_tool_call
from src.tools.protocol import validate_tool_input

print("🚀 正在初始化天文工具...")
ephemeris = None
planetary = None
celestial_db = None
nasa_api = None
weather = None
search = None
events_predictor = None

try:
    ephemeris = EphemerisManager()
    print("✅ EphemerisManager 初始化完成")
except Exception as e:
    print(f"⚠️  EphemerisManager 初始化失败: {e}")

try:
    if ephemeris:
        planetary = PlanetaryCalculator(ephemeris)
        print("✅ PlanetaryCalculator 初始化完成")
except Exception as e:
    print(f"⚠️  PlanetaryCalculator 初始化失败: {e}")

try:
    celestial_db = CelestialDatabaseService()
    print("✅ CelestialDatabaseService 初始化完成")
except Exception as e:
    print(f"⚠️  CelestialDatabaseService 初始化失败: {e}")

try:
    nasa_api = NASAAPIService()
    print("✅ NASAAPIService 初始化完成")
except Exception as e:
    print(f"⚠️  NASAAPIService 初始化失败: {e}")

try:
    weather = WeatherService()
    print("✅ WeatherService 初始化完成")
except Exception as e:
    print(f"⚠️  WeatherService 初始化失败: {e}")

try:
    search = SearchService()
    print("✅ SearchService 初始化完成")
except Exception as e:
    print(f"⚠️  SearchService 初始化失败: {e}")

try:
    if ephemeris:
        events_predictor = EventsPredictor(ephemeris)
        print("✅ EventsPredictor 初始化完成")
except Exception as e:
    print(f"⚠️  EventsPredictor 初始化失败: {e}")

if ephemeris is None:
    print("❌ 星历数据初始化失败，服务器无法启动")
    sys.exit(1)

print("✅ 天文工具初始化完成（部分功能可能受限）")

mcp = FastMCP(name="Astronomy Server")


def _require_planetary():
    """确保行星计算服务已初始化。"""
    if planetary is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message=f"PlanetaryCalculator 未初始化，请检查星历数据文件 {settings.EPHEMERIS_FILE} 是否存在",
        )


def _require_celestial_db():
    """确保天体数据库服务已初始化。"""
    if celestial_db is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED, message="CelestialDatabaseService 未初始化"
        )


def _require_nasa_api():
    """确保 NASA API 服务已初始化。"""
    if nasa_api is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED, message="NASAAPIService 未初始化"
        )


def _require_weather():
    """确保天气服务已初始化。"""
    if weather is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED, message="WeatherService 未初始化"
        )


def _require_search():
    """确保搜索服务已初始化。"""
    if search is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED, message="SearchService 未初始化"
        )


def _require_events_predictor():
    """确保天象事件预测服务已初始化。"""
    if events_predictor is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message=f"EventsPredictor 未初始化，请检查星历数据文件 {settings.EPHEMERIS_FILE} 是否存在",
        )


@mcp.tool()
@safe_tool_call
def get_planet_position(
    planet_name: str,
    observation_time: str = None,
    latitude: float = None,
    longitude: float = None,
) -> str:
    """查询指定行星在给定时间地点的赤道坐标。"""
    params = validate_tool_input(
        "get_planet_position",
        {
            "planet_name": planet_name,
            "observation_time": observation_time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    _require_planetary()
    return planetary.get_planet_position(
        params.planet_name,
        params.observation_time,
        params.latitude,
        params.longitude,
    )


@mcp.tool()
@safe_tool_call
def get_altaz(
    planet_name: str,
    observation_time: str = None,
    latitude: float = None,
    longitude: float = None,
) -> str:
    """查询指定行星在给定时间地点的高度角和方位角。"""
    params = validate_tool_input(
        "get_altaz",
        {
            "planet_name": planet_name,
            "observation_time": observation_time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    _require_planetary()
    return planetary.get_altaz(
        params.planet_name,
        params.observation_time,
        params.latitude,
        params.longitude,
    )


@mcp.tool()
@safe_tool_call
def coordinate_transformation(
    ra: float, dec: float, epoch: str = "J2000", target_system: str = "fk5"
) -> str:
    """执行赤经赤纬坐标到目标坐标系统的转换。"""
    params = validate_tool_input(
        "coordinate_transformation",
        {
            "ra": ra,
            "dec": dec,
            "epoch": epoch,
            "target_system": target_system,
        },
    )
    _require_planetary()
    return planetary.coordinate_transformation(
        params.ra,
        params.dec,
        params.epoch,
        params.target_system,
    )


@mcp.tool()
@safe_tool_call
def get_rise_set_times(
    body_name: str, latitude: float, longitude: float, date: str = None
) -> str:
    """计算天体在指定地点和日期的升起、落下时间。"""
    params = validate_tool_input(
        "get_rise_set_times",
        {
            "body_name": body_name,
            "latitude": latitude,
            "longitude": longitude,
            "date": date,
        },
    )
    _require_planetary()
    return planetary.get_rise_set_times(
        params.body_name,
        params.latitude,
        params.longitude,
        params.date,
    )


@mcp.tool()
@safe_tool_call
def get_current_sky_objects(latitude: float, longitude: float, date: str = None) -> str:
    """查询指定地点和日期当前可见的主要天空目标。"""
    params = validate_tool_input(
        "get_current_sky_objects",
        {"latitude": latitude, "longitude": longitude, "date": date},
    )
    _require_planetary()
    return planetary.get_current_sky_objects(
        params.latitude,
        params.longitude,
        params.date,
    )


@mcp.tool()
@safe_tool_call
def get_astrophysical_object_info(object_name: str) -> str:
    """查询天体物理数据库中的目标基础信息。"""
    params = validate_tool_input(
        "get_astrophysical_object_info",
        {"object_name": object_name},
    )
    _require_celestial_db()
    return celestial_db.get_object_info(params.object_name)


@mcp.tool()
@safe_tool_call
def get_galaxy_data(galaxy_name: str) -> str:
    """查询星系数据库中的目标资料。"""
    params = validate_tool_input("get_galaxy_data", {"galaxy_name": galaxy_name})
    _require_celestial_db()
    return celestial_db.get_galaxy_data(params.galaxy_name)


@mcp.tool()
@safe_tool_call
def get_nasa_apod(date: str = None, hd: bool = False) -> str:
    """查询 NASA APOD 每日天文图片。"""
    params = validate_tool_input("get_nasa_apod", {"date": date, "hd": hd})
    _require_nasa_api()
    return nasa_api.get_apod(params.date, params.hd)


@mcp.tool()
@safe_tool_call
def get_neo_data(start_date: str = None, end_date: str = None, limit: int = 10) -> str:
    """查询 NASA 近地天体飞掠数据，并限制 API 支持的最大日期范围。"""
    params = validate_tool_input(
        "get_neo_data",
        {"start_date": start_date, "end_date": end_date, "limit": limit},
    )
    _require_nasa_api()
    from datetime import datetime, timedelta

    if params.start_date:
        start_dt = datetime.strptime(params.start_date, "%Y-%m-%d")
    else:
        start_dt = datetime.now()
        start_date = start_dt.strftime("%Y-%m-%d")

    if params.end_date:
        end_dt = datetime.strptime(params.end_date, "%Y-%m-%d")
    else:
        end_dt = start_dt + timedelta(days=7)
        end_date = end_dt.strftime("%Y-%m-%d")

    delta_days = (end_dt - start_dt).days
    if delta_days > 7:
        end_dt = start_dt + timedelta(days=7)
        end_date = end_dt.strftime("%Y-%m-%d")

    return nasa_api.get_neo_data(start_date, end_date, params.limit)


@mcp.tool()
@safe_tool_call
def get_weather(city: str = None, extensions: str = "base") -> str:
    """查询指定城市或区域的天气信息。"""
    params = validate_tool_input(
        "get_weather",
        {"city": city, "extensions": extensions},
    )
    _require_weather()
    return weather.get_weather(city=params.city, extensions=params.extensions)


@mcp.tool()
@safe_tool_call
def web_search(query: str, max_results: int = 5) -> str:
    """执行外部联网搜索并返回搜索结果摘要。"""
    params = validate_tool_input(
        "web_search",
        {"query": query, "max_results": max_results},
    )
    _require_search()
    return search.search(query=params.query, max_results=params.max_results)


@mcp.tool()
@safe_tool_call
def get_tonight_best() -> str:
    """查询今晚推荐观测目标。"""
    validate_tool_input("get_tonight_best", {})
    _require_events_predictor()
    return events_predictor.get_tonight_best()


@mcp.tool()
@safe_tool_call
def get_weekly_events(start_date: Optional[str] = None) -> str:
    """查询从指定日期开始的一周天象事件。"""
    params = ParamParser.parse_tool_input(
        start_date, expected_params={"start_date": None}
    )
    processed_start_date = ParamParser.normalize_date(params.get("start_date"))

    if processed_start_date:
        processed_start_date = processed_start_date.strftime("%Y-%m-%d")

    params = validate_tool_input(
        "get_weekly_events",
        {"start_date": processed_start_date},
    )

    _require_events_predictor()
    return events_predictor.get_weekly_events(params.start_date)


@mcp.tool()
@safe_tool_call
def get_monthly_events(
    year: Optional[Union[int, str, dict]] = None,
    month: Optional[Union[int, str]] = None,
) -> str:
    """查询指定年月的月度天象事件。"""
    params = ParamParser.parse_tool_input(
        {"year": year, "month": month}, expected_params={"year": None, "month": None}
    )

    validated = validate_tool_input(
        "get_monthly_events",
        {
            "year": ParamParser.safe_int(params.get("year"), default=None),
            "month": ParamParser.safe_int(params.get("month"), default=None),
        },
    )

    _require_events_predictor()
    return events_predictor.get_monthly_events(validated.year, validated.month)


@mcp.resource("status://server")
def get_server_status() -> str:
    """返回 MCP server 当前运行状态。"""
    return json.dumps(
        {"status": "running", "mode": "streamable-http", "version": "1.0.0"},
        ensure_ascii=False,
    )


@mcp.prompt()
def observation_guide(target: str) -> str:
    """返回观测指导 prompt 模板。"""
    return f"""
请帮我制定一个观测 {target} 的完整计划，包含以下内容：

一、目标基本信息
- 天体类型（恒星/行星/星系/星云/星团/彗星等）
- 距离（光年或天文单位）
- 视星等（apparent magnitude）与绝对星等（如适用）
- 角大小（arcmin/arcsec，如适用）
- 最佳观测季节与月份

二、观测时间规划
- 当月可见时段（升起/落下时间，或最佳观测窗口）
- 月相影响评估（新月期间适合深空，满月期间适合行星/月面）
- 建议的具体观测时段（前半夜/后半夜/凌晨）

三、设备建议
- 裸眼可见性评估
- 推荐望远镜口径与类型（折射/反射/DOB等）
- 推荐目镜倍率范围
- 摄影设备建议（如适用）：相机、赤道仪、导星方案

四、观测技巧与注意事项
- 寻星方法（跳星法/直推法等）
- 需要注意的环境条件（光害、透明度、视宁度）
- 极轴校准精度要求（如需跟踪拍摄）

五、可拍摄的天体特征
- 可见的表面细节或结构
- 推荐的曝光参数范围
- 后期处理建议（叠加、校准帧等）
"""


def main():
    """启动 Streamable HTTP 模式的 FastMCP 服务。"""
    print("\n" + "=" * 60)
    print("🌌 天文MCP服务器 - Streamable HTTP模式")
    print("=" * 60)

    host = os.environ.get("FASTMCP_HOST", "0.0.0.0")
    port = int(os.environ.get("FASTMCP_PORT", "8001"))

    print(f"📡 监听地址: http://{host}:{port}/mcp/")
    print(f"📦 已注册工具: 13个")
    print(f"💡 按 Ctrl+C 停止服务器")
    print("=" * 60 + "\n")

    try:
        mcp.run(transport="streamable-http", host=host, port=port)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
