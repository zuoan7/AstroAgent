#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# streamable_mcp_server.py - Streamable HTTP MCP服务器
# 使用FastMCP框架，完整支持标准MCP协议和HTTP传输

import os
import sys
import json
from typing import Optional, Union

from fastmcp import FastMCP
from src.astronomy import (
    EphemerisManager,
    PlanetaryCalculator,
    CelestialDatabaseService,
    NASAAPIService,
    WeatherService,
    SearchService,
    EventsPredictor,
)
from src.agent.param_parser import ParamParser
from src.core.config import settings
from src.core.errors import AgentError, ErrorCode, ErrorHandler, safe_tool_call

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
    if planetary is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message=f"PlanetaryCalculator 未初始化，请检查星历数据文件 {settings.EPHEMERIS_FILE} 是否存在"
        )


def _require_celestial_db():
    if celestial_db is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message="CelestialDatabaseService 未初始化"
        )


def _require_nasa_api():
    if nasa_api is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message="NASAAPIService 未初始化"
        )


def _require_weather():
    if weather is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message="WeatherService 未初始化"
        )


def _require_search():
    if search is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message="SearchService 未初始化"
        )


def _require_events_predictor():
    if events_predictor is None:
        raise AgentError(
            code=ErrorCode.TOOL_CALL_FAILED,
            message=f"EventsPredictor 未初始化，请检查星历数据文件 {settings.EPHEMERIS_FILE} 是否存在"
        )


@mcp.tool()
@safe_tool_call
def get_planet_position(planet_name: str, observation_time: str = None,
                       latitude: float = None, longitude: float = None) -> str:
    _require_planetary()
    result = planetary.get_planet_position(planet_name, observation_time, latitude, longitude)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@safe_tool_call
def coordinate_transformation(ra: float, dec: float, epoch: str = "J2000", target_system: str = "fk5") -> str:
    _require_planetary()
    return json.dumps(planetary.coordinate_transformation(ra, dec, epoch, target_system), ensure_ascii=False)


@mcp.tool()
@safe_tool_call
def get_rise_set_times(body_name: str, latitude: float, longitude: float, date: str = None) -> str:
    _require_planetary()
    return planetary.get_rise_set_times(body_name, latitude, longitude, date)


@mcp.tool()
@safe_tool_call
def get_current_sky_objects(latitude: float, longitude: float, date: str = None) -> str:
    _require_planetary()
    return planetary.get_current_sky_objects(latitude, longitude, date)


@mcp.tool()
@safe_tool_call
def get_astrophysical_object_info(object_name: str) -> str:
    _require_celestial_db()
    result = celestial_db.get_object_info(object_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@safe_tool_call
def get_galaxy_data(galaxy_name: str) -> str:
    _require_celestial_db()
    result = celestial_db.get_galaxy_data(galaxy_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@safe_tool_call
def get_nasa_apod(date: str = None, hd: bool = False) -> str:
    _require_nasa_api()
    return nasa_api.get_apod(date, hd)


@mcp.tool()
@safe_tool_call
def get_neo_data(start_date: str = None, end_date: str = None, limit: int = 10) -> str:
    _require_nasa_api()
    from datetime import datetime, timedelta

    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = datetime.now()
        start_date = start_dt.strftime("%Y-%m-%d")

    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = start_dt + timedelta(days=7)
        end_date = end_dt.strftime("%Y-%m-%d")

    delta_days = (end_dt - start_dt).days
    if delta_days > 7:
        end_dt = start_dt + timedelta(days=7)
        end_date = end_dt.strftime("%Y-%m-%d")

    result = nasa_api.get_neo_data(start_date, end_date, limit)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@safe_tool_call
def get_weather(city: str = None, extensions: str = "base") -> str:
    _require_weather()
    result = weather.get_weather(city=city, extensions=extensions)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@safe_tool_call
def web_search(query: str, max_results: int = 5) -> str:
    _require_search()
    result = search.search(query=query, max_results=max_results)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@safe_tool_call
def get_tonight_best() -> str:
    _require_events_predictor()
    return events_predictor.get_tonight_best()


@mcp.tool()
@safe_tool_call
def get_weekly_events(start_date: Optional[str] = None) -> str:
    params = ParamParser.parse_tool_input(
        start_date,
        expected_params={"start_date": None}
    )
    processed_start_date = ParamParser.normalize_date(params.get("start_date"))

    if processed_start_date:
        processed_start_date = processed_start_date.strftime("%Y-%m-%d")

    _require_events_predictor()
    return events_predictor.get_weekly_events(processed_start_date)


@mcp.tool()
@safe_tool_call
def get_monthly_events(year: Optional[Union[int, str, dict]] = None,
                       month: Optional[Union[int, str]] = None) -> str:
    params = ParamParser.parse_tool_input(
        {"year": year, "month": month},
        expected_params={"year": None, "month": None}
    )

    processed_year = ParamParser.safe_int(params.get("year"), default=None)
    processed_month = ParamParser.safe_int(params.get("month"), default=None)

    _require_events_predictor()
    return events_predictor.get_monthly_events(processed_year, processed_month)


@mcp.resource("status://server")
def get_server_status() -> str:
    return json.dumps({
        "status": "running",
        "mode": "streamable-http",
        "version": "1.0.0"
    }, ensure_ascii=False)


@mcp.prompt()
def observation_guide(target: str) -> str:
    return f"""
请帮我制定一个观测 {target} 的计划，包含以下内容：
1. 目标的基本信息（类型、距离、视星等）
2. 最佳观测时间（月份、时段）
3. 需要使用的望远镜/设备建议
4. 观测注意事项
5. 可以拍摄的天体特征
"""


def main():
    print("\n" + "="*60)
    print("🌌 天文MCP服务器 - Streamable HTTP模式")
    print("="*60)

    host = os.environ.get("FASTMCP_HOST", "0.0.0.0")
    port = int(os.environ.get("FASTMCP_PORT", "8001"))

    print(f"📡 监听地址: http://{host}:{port}/mcp/")
    print(f"📦 已注册工具: 13个")
    print(f"💡 按 Ctrl+C 停止服务器")
    print("="*60 + "\n")

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
