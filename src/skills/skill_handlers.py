from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.core.config import settings
from src.core.errors import AgentError, ErrorCode
from src.core.logger import logger
from src.agent.param_parser import ParamParser
from src.skills.mcp_client import MCPClient


def _summarize_weather(raw: str) -> str:
    try:
        data = json.loads(raw)
    except Exception:
        return ""

    if isinstance(data, dict) and data.get("error"):
        return ""

    parts = []

    live = data.get("live") or {}
    if live:
        city = live.get("city")
        weather = live.get("weather")
        temp = live.get("temperature")
        humidity = live.get("humidity")
        wind = live.get("windpower")
        if city:
            parts.append(f"{city} 当前天气：{weather}，气温约 {temp}°C，湿度 {humidity}%，风力 {wind} 级左右。")
        else:
            parts.append(f"当前天气：{weather}，气温约 {temp}°C，湿度 {humidity}%。")

    forecast = data.get("forecast") or {}
    if forecast:
        casts = forecast.get("casts") or []
        if casts:
            first_day = casts[0]
            date = first_day.get("date", "")
            day_weather = first_day.get("dayweather", "")
            night_weather = first_day.get("nightweather", "")
            day_temp = first_day.get("daytemp", "")
            night_temp = first_day.get("nighttemp", "")
            city = forecast.get("city", "")
            if city:
                parts.append(f"{city} ({date}) 白天：{day_weather}，{day_temp}°C；夜间：{night_weather}，{night_temp}°C。")
            else:
                parts.append(f"{date} 白天：{day_weather}，{day_temp}°C；夜间：{night_weather}，{night_temp}°C。")

    tips = data.get("observing_tips") or []
    for t in tips:
        parts.append(f"- {t}")

    return "\n".join(parts) if parts else ""


def _is_date_like(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t in ("今天", "明天", "今日", "次日", "today", "tomorrow"):
        return True
    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", t):
        return True
    return False


def _parse_location(location: str) -> tuple[Optional[float], Optional[float]]:
    if not location:
        return None, None
    text = location.replace("，", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        return None, None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
        return lat, lon
    except Exception:
        return None, None


def _parse_time_range(time_range: Optional[str]) -> tuple[str, str]:
    today = datetime.now().date()
    if not time_range:
        return today.strftime("%Y-%m-%d"), (today + timedelta(days=7)).strftime("%Y-%m-%d")

    text = str(time_range)
    if "30" in text and "天" in text:
        start = today
        end = today + timedelta(days=30)
    elif "本月" in text:
        start = today.replace(day=1)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
    else:
        start = today
        end = today + timedelta(days=7)

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


class ObservationPlannerHandler:
    def __call__(
        self,
        mcp: MCPClient,
        date: Optional[str] = None,
        location: Optional[str] = None,
        duration: Optional[str] = None,
    ) -> str:
        if not location and date:
            text = str(date).strip()
            if not _is_date_like(text):
                location = text
                date = None

        obs_date = ParamParser.parse_date(date)

        display_location = ParamParser.normalize_location(location)
        query_city = None
        if location is not None:
            if isinstance(location, dict):
                query_city = location.get("city") or display_location
            else:
                query_city = display_location

        weather_brief = ""
        weekly_events = ""
        tonight_best = ""

        parallel_calls = []
        if query_city:
            parallel_calls.append({
                "tool_name": "get_weather",
                "kwargs": {"city": query_city, "extensions": "all"},
                "_key": "weather",
            })
        parallel_calls.append({
            "tool_name": "get_weekly_events",
            "kwargs": {"start_date": obs_date.strftime("%Y-%m-%d")},
            "_key": "weekly_events",
        })
        today_str = datetime.now().strftime("%Y-%m-%d")
        if obs_date.strftime("%Y-%m-%d") == today_str:
            parallel_calls.append({
                "tool_name": "get_tonight_best",
                "kwargs": {},
                "_key": "tonight_best",
            })

        if len(parallel_calls) > 1:
            parallel_results = mcp.call_tools_parallel(parallel_calls)
            for i, result in enumerate(parallel_results):
                key = parallel_calls[i]["_key"]
                if key == "weather":
                    weather_brief = _summarize_weather(result)
                elif key == "weekly_events":
                    weekly_events = result
                elif key == "tonight_best":
                    tonight_best = result
        else:
            for call in parallel_calls:
                if call["_key"] == "weather":
                    weather_raw = mcp.call_tool("get_weather", city=query_city, extensions="all")
                    weather_brief = _summarize_weather(weather_raw)
                elif call["_key"] == "weekly_events":
                    weekly_events = mcp.call_tool(
                        "get_weekly_events",
                        start_date=obs_date.strftime("%Y-%m-%d"),
                    )
                elif call["_key"] == "tonight_best":
                    tonight_best = mcp.call_tool("get_tonight_best")

        plan_lines = [
            f"📅 观测日期：{obs_date.strftime('%Y-%m-%d')}",
        ]
        if display_location:
            plan_lines.append(f"📍 观测地点：{display_location}")
        else:
            plan_lines.append("📍 观测地点：未明确指定，本计划为一般性观测建议。")
        if duration:
            plan_lines.append(f"⏱️ 观测时段：{duration}")

        plan_lines.append("\n一、观测条件（天气概览）")
        if weather_brief:
            plan_lines.append(weather_brief)
        else:
            plan_lines.append("暂时无法获取指定地点的天气信息，请根据当地实际情况或天气预报应用调整计划。")

        plan_lines.append("\n二、本周重要天象（摘要）")
        plan_lines.append(ParamParser.shorten_text(weekly_events, 600))

        if tonight_best:
            plan_lines.append("\n三、系统给出的\"今晚最佳观测目标\"参考")
            plan_lines.append(ParamParser.shorten_text(tonight_best, 600))

        plan_lines.append("\n四、实用建议")
        plan_lines.append(
            "1. 根据云量与月相选择目标：若接近新月，可优先考虑深空天体；若接近满月，可多观测月面细节与行星。\n"
            "2. 提前抵达观测地，预留架台、极轴校准与试拍时间。\n"
            "3. 如湿度偏高或风力较大，请准备除露带、配重或防风措施。"
        )

        return "\n".join(plan_lines)


class CelestialEventsForecastHandler:
    def __call__(
        self,
        mcp: MCPClient,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> str:
        if start_date:
            start_dt = ParamParser.parse_date(start_date)
        else:
            start_dt = datetime.now()

        supported_min, supported_max = settings.SUPPORTED_YEAR_RANGE
        if not (supported_min <= start_dt.year <= supported_max):
            start_dt = start_dt.replace(year=supported_min)

        end_dt: Optional[datetime] = None
        if end_date:
            try:
                end_dt = ParamParser.parse_date(end_date)
                if not (supported_min <= end_dt.year <= supported_max):
                    end_dt = end_dt.replace(year=supported_min)
            except Exception:
                end_dt = None

        description_prefix: list[str] = []
        if end_dt:
            description_prefix.append(
                f"天象预报范围：从 {start_dt.strftime('%Y-%m-%d')} 开始，直到 {end_dt.strftime('%Y-%m-%d')}"
            )
        else:
            description_prefix.append(
                f"天象预报范围：从 {start_dt.strftime('%Y-%m-%d')} 开始"
                + (f"，直到 {end_date}" if end_date else "，未来一段时间内")
            )
        if event_type:
            description_prefix.append(f"用户关心的事件类型：{event_type}（当前版本为软筛选，仅供解释用）")

        use_weekly = False
        if end_dt:
            try:
                days = (end_dt - start_dt).days
                use_weekly = days <= 7
            except Exception:
                use_weekly = False
        else:
            use_weekly = True

        if use_weekly:
            body = mcp.call_tool(
                "get_weekly_events",
                start_date=start_dt.strftime("%Y-%m-%d"),
            )
        else:
            if end_dt:
                current_dt = start_dt
                monthly_calls = []
                while current_dt <= end_dt:
                    monthly_calls.append({
                        "tool_name": "get_monthly_events",
                        "kwargs": {"year": current_dt.year, "month": current_dt.month},
                    })
                    if current_dt.month == 12:
                        current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
                    else:
                        current_dt = current_dt.replace(month=current_dt.month + 1)
                monthly_results = mcp.call_tools_parallel(monthly_calls)
                body = "\n".join(monthly_results)
            else:
                year = start_dt.year
                month = start_dt.month
                body = mcp.call_tool(
                    "get_monthly_events",
                    year=year,
                    month=month,
                )

        description_prefix.append("\n下面是为你整理的天象预报：\n")
        description_prefix.append(ParamParser.shorten_text(body, 1200))
        return "\n".join(description_prefix)


class DeepSkyObservingGuideHandler:
    def __call__(
        self,
        mcp: MCPClient,
        target: str,
        observer_location: Optional[str] = None,
        date: Optional[str] = None,
        equipment: Optional[str] = None,
    ) -> str:
        if not target:
            return json.dumps(AgentError(
                code=ErrorCode.VALIDATION_ERROR,
                message="深空观测指导技能需要提供目标名称（target），例如\"M31\"或\"猎户座大星云\"",
                details={"skill": "deep-sky-observing-guide"}
            ).to_dict(), ensure_ascii=False)

        obs_date = ParamParser.parse_date(date) if date else datetime.now()

        parallel_calls = [
            {
                "tool_name": "get_astrophysical_object_info",
                "kwargs": {"object_name": target},
                "_key": "obj_info",
            },
        ]
        if any(x in target.lower() for x in ["galaxy", "星系", "m31", "m33"]):
            parallel_calls.append({
                "tool_name": "get_galaxy_data",
                "kwargs": {"galaxy_name": target},
                "_key": "galaxy_info",
            })
        if observer_location:
            parallel_calls.append({
                "tool_name": "get_weather",
                "kwargs": {"city": observer_location, "extensions": "all"},
                "_key": "weather",
            })

        parallel_results = mcp.call_tools_parallel(parallel_calls)
        obj_info_raw = ""
        galaxy_info_raw: Optional[str] = None
        weather_brief = ""
        for i, result in enumerate(parallel_results):
            key = parallel_calls[i]["_key"]
            if key == "obj_info":
                obj_info_raw = result
            elif key == "galaxy_info":
                galaxy_info_raw = result
            elif key == "weather":
                weather_brief = _summarize_weather(result)

        lines = [
            f"🎯 深空目标：{target}",
            f"📅 观测日期：{obs_date.strftime('%Y-%m-%d')}",
        ]
        if observer_location:
            lines.append(f"📍 观测地点：{observer_location}")
        if equipment:
            lines.append(f"🔭 计划使用设备：{equipment}")

        lines.append("\n一、目标基础信息（基于专业数据库）")
        lines.append(ParamParser.shorten_text(obj_info_raw, 600))
        if galaxy_info_raw:
            lines.append("\n补充：星系数据摘要")
            lines.append(ParamParser.shorten_text(galaxy_info_raw, 400))

        if weather_brief:
            lines.append("\n二、观测条件（天气简要）")
            lines.append(weather_brief)

        lines.append("\n三、观测建议")
        lines.append(
            "1. 建议选择无月光或月亮落下后 1-2 小时的时段进行深空观测。\n"
            "2. 若使用双筒或小口径望远镜，可优先寻找目标所在星座的亮星作\"跳星\"指引。\n"
            "3. 使用较低倍率（长焦距目镜）先锁定目标，再逐步提高放大倍率细看结构。"
        )

        return "\n".join(lines)


class NeoTrackerHandler:
    def __call__(
        self,
        mcp: MCPClient,
        time_range: Optional[str] = None,
        min_size: Optional[float] = None,
        max_distance: Optional[float] = None,
        observable_only: Optional[bool] = None,
    ) -> str:
        start_date, end_date = _parse_time_range(time_range)

        warning_lines = []
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            delta_days = (end_dt - start_dt).days

            if delta_days > 7:
                warning_lines.append(f"⚠️ 注意：NASA NEO API 最多只支持查询7天的数据。")
                warning_lines.append(f"请求范围：{start_date} 至 {end_date}（{delta_days}天）")
                warning_lines.append(f"将返回最近7天的数据：{start_date} 至 {(start_dt + timedelta(days=7)).strftime('%Y-%m-%d')}")
                warning_lines.append("")
        except Exception:
            pass

        raw_json = mcp.call_tool(
            "get_neo_data",
            start_date=start_date,
            end_date=end_date,
            limit=50,
        )

        data = None
        try:
            if isinstance(raw_json, str):
                data = json.loads(raw_json)
            else:
                data = raw_json
        except Exception:
            return raw_json

        if isinstance(data, dict) and data.get("error"):
            if isinstance(data.get("error"), bool) and data.get("code"):
                return json.dumps(data, ensure_ascii=False)
            return json.dumps(AgentError(
                code=ErrorCode.TOOL_CALL_FAILED,
                message=str(data.get("error")),
                details={"tool": "get_neo_data"}
            ).to_dict(), ensure_ascii=False)

        neos = []
        for day, objs in (data.get("near_earth_objects") or {}).items():
            for obj in objs:
                neos.append((day, obj))

        filtered = []
        for day, obj in neos:
            name = obj.get("name")
            est = obj.get("estimated_diameter", {}).get("meters", {})
            size = est.get("estimated_diameter_max")
            ca_list = obj.get("close_approach_data") or []
            if not ca_list:
                continue
            ca = ca_list[0]
            miss_lunar = None
            try:
                miss_lunar = float(ca.get("miss_distance", {}).get("lunar", "0"))
            except Exception:
                miss_lunar = None

            if min_size is not None and size is not None and size < min_size:
                continue
            if max_distance is not None and miss_lunar is not None and miss_lunar > max_distance:
                continue

            if observable_only:
                try:
                    abs_mag = float(obj.get("absolute_magnitude_h"))
                    if abs_mag > 25:
                        continue
                except Exception:
                    pass

            filtered.append(
                {
                    "date": day,
                    "name": name,
                    "size_m": size,
                    "miss_distance_lunar": miss_lunar,
                    "hazardous": obj.get("is_potentially_hazardous_asteroid"),
                }
            )

        if not filtered:
            warning_text = "\n".join(warning_lines) if warning_lines else ""
            return warning_text + "\n在给定的时间范围和筛选条件下，没有找到明显具有观测价值的近地天体飞掠事件。"

        if warning_lines:
            lines = warning_lines.copy()
            lines.extend([
                "",
                "📡 近地天体飞掠列表（按时间排序）：",
            ])
        else:
            lines = [
                "📡 近地天体飞掠列表（按时间排序）：",
            ]

        filtered.sort(key=lambda x: (x["date"], x.get("miss_distance_lunar") or 1e9))
        for item in filtered[:20]:
            hazard = "⚠️ 潜在威胁小行星" if item["hazardous"] else ""
            size_str = f"{item['size_m']:.0f} m" if item['size_m'] is not None else "未知"
            dist_str = f"{item['miss_distance_lunar']:.2f} 个地月距离" if item['miss_distance_lunar'] is not None else "未知"
            lines.append(
                f"- 日期 {item['date']}，目标 {item['name']}，"
                f"估计直径约 {size_str}，最近距离约 {dist_str} {hazard}"
            )

        lines.append(
            "\n注：以上数据来自 NASA NEO 数据接口，是否实际可见还与亮度、天空背景和观测设备有关。"
        )
        return "\n".join(lines)


class AstrophotographyCalculatorHandler:
    def __call__(
        self,
        mcp: MCPClient,
        target: str,
        camera: str,
        telescope: Optional[str] = None,
        mount: Optional[str] = None,
        location: Optional[str] = None,
        date: Optional[str] = None,
    ) -> str:
        obs_date = ParamParser.parse_date(date) if date else datetime.now()

        lines = [
            f"📷 天文摄影参数建议",
            f"🎯 拍摄目标：{target}",
            f"📅 拍摄日期：{obs_date.strftime('%Y-%m-%d')}",
            f"📸 相机：{camera}",
        ]
        if telescope:
            lines.append(f"🔭 望远镜/镜头：{telescope}")
        if mount:
            lines.append(f"🗜 赤道仪/支架：{mount}")
        if location:
            lines.append(f"📍 拍摄地点：{location}")

        lines.append("\n一、曝光时间估算（星点不拖尾的经验值）")
        lines.append(
            "若使用广角/标准镜头并在赤道仪跟踪下：\n"
            "- 星野/银河：单张 20-60 秒，ISO 1600-6400，光圈尽量开大。\n"
            "- 星云/星团：根据目标亮度，单张 120-300 秒，ISO 800-3200。\n"
            "若非跟踪（固定三脚架），可按\"500 规则\"粗略估计：曝光秒数 ≈ 500 / 焦距（全画幅等效）。"
        )

        lines.append("\n二、总曝光时间与叠加")
        lines.append(
            "为了获得较低噪点和更丰富细节，建议：\n"
            "- 星云/星系：累计曝光时间 1–3 小时以上（例如 120s × 30–90 张）。\n"
            "- 银河/星野：累计曝光 20 分钟以上即可有明显提升。\n"
            "请务必拍摄暗场/平场/偏置帧，以便后期校正。"
        )

        lines.append("\n三、赤道仪与极轴校准建议")
        lines.append(
            "若使用赤道仪：\n"
            "- 极轴误差越小，可用的单张曝光时间越长。\n"
            "- 建议使用极轴镜/电子极轴校准工具，将极轴误差控制在 1–2 角分以内。"
        )

        return "\n".join(lines)


class CelestialPositionCalculatorHandler:
    def __call__(
        self,
        mcp: MCPClient,
        target: str,
        datetime: Optional[str] = None,
        location: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> str:
        if not target:
            return json.dumps(AgentError(
                code=ErrorCode.VALIDATION_ERROR,
                message="天体位置计算技能需要提供目标名称（target），例如\"mars\"\"jupiter\"等",
                details={"skill": "celestial-position-calculator"}
            ).to_dict(), ensure_ascii=False)

        import datetime as dt_mod
        obs_time = ParamParser.parse_date(datetime) if datetime else dt_mod.datetime.now()

        if location:
            lat, lon = _parse_location(location)
        else:
            lat, lon = None, None

        if lat is None or lon is None:
            lat, lon = 39.9, 116.4

        result_raw = mcp.call_tool(
            "get_planet_position",
            planet_name=target,
            observation_time=obs_time.isoformat(),
            latitude=lat,
            longitude=lon,
        )

        body = ParamParser.shorten_text(result_raw, 600)
        fmt = (output_format or "radec").lower()

        header = (
            f"🪐 天体位置计算\n"
            f"- 目标：{target}\n"
            f"- 时间：{obs_time.isoformat()}\n"
            f"- 观测点：纬度 {lat}，经度 {lon}\n"
            f"- 输出坐标系偏好：{fmt}（当前实现以赤道坐标为主，若需 altaz 请在最终回答中由 LLM 补充说明）\n"
        )

        return header + "\n原始计算结果（来自底层工具）：\n" + body


SKILL_HANDLERS: Dict[str, type] = {
    "observation-planner": ObservationPlannerHandler,
    "celestial-events-forecast": CelestialEventsForecastHandler,
    "deep-sky-observing-guide": DeepSkyObservingGuideHandler,
    "neo-tracker": NeoTrackerHandler,
    "astrophotography-calculator": AstrophotographyCalculatorHandler,
    "celestial-position-calculator": CelestialPositionCalculatorHandler,
}
