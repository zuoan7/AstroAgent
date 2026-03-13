from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

import httpx
from logger import logger

MCP_SERVER_URL = "http://localhost:8001/mcp"


class AstronomySkillRouter:
    """
    天文技能路由层。

    说明：
    - 对上：只暴露“技能”接口（与 skill.yaml 中的 name 一一对应）
    - 对下：直接通过内部 MCP 客户端调用具体工具（get_weather / get_weekly_events 等）
    - 这样可以保证上层 Agent 不再直接面向底层工具，而是通过技能完成复杂任务编排
    """

    def __init__(self) -> None:
        self._mcp_session_id: Optional[str] = None
        self._mcp_initialized = False
        self._http_client: Optional[httpx.Client] = None
        self._init_mcp_session_sync()

        self._call_mcp_tool = self._call_mcp_tool_internal
        # 注册表：技能名 -> 具体实现函数
        self._registry: Dict[str, Callable[..., str]] = {
            "observation-planner": self._observation_planner,
            "celestial-events-forecast": self._celestial_events_forecast,
            "deep-sky-observing-guide": self._deep_sky_observing_guide,
            "neo-tracker": self._neo_tracker,
            "astrophotography-calculator": self._astrophotography_calculator,
            "celestial-position-calculator": self._celestial_position_calculator,
        }
        # 简单直通型技能配置：skill_name -> {"tool_name": str, "param_mapping": {skill_param: tool_param}}
        self._simple_skills: Dict[str, Dict[str, Any]] = {}

        # 预注册一个基于 get_weather 的简单天气查询技能
        # skill 名：weather-lookup，底层工具：get_weather
        self.register_simple_skill(
            skill_name="weather-lookup",
            tool_name="get_weather",
            param_mapping={
                "city": "city",
                "location": "city",
                "extensions": "extensions",
            },
        )

    # ===== 公共接口 =====

    def list_skills(self) -> Dict[str, str]:
        """返回可用技能名称及简要说明（说明保持与 skill.yaml 一致的语义）。"""
        return {
            "observation-planner": "生成指定日期和地点的天文观测计划",
            "celestial-events-forecast": "查询指定时间段的天象事件",
            "deep-sky-observing-guide": "为指定深空天体提供观测指导",
            "neo-tracker": "追踪近地天体飞掠事件",
            "astrophotography-calculator": "计算天文摄影参数与建议",
            "celestial-position-calculator": "计算天体在指定时间的位置",
            "weather-lookup": "查询指定城市的观测相关天气信息",
        }

    def register_simple_skill(
        self,
        skill_name: str,
        tool_name: str,
        param_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        注册“单工具直通型”技能：
        - skill_name: 对上暴露的技能名
        - tool_name: 对下实际调用的 MCP 工具名
        - param_mapping: 可选，skill 参数名 -> MCP 工具参数名 的映射，默认同名直传
        """
        self._simple_skills[skill_name] = {
            "tool_name": tool_name,
            "param_mapping": param_mapping or {},
        }

    def call(self, name: str, **params: Any) -> str:
        """
        调用指定技能。

        Args:
            name: 技能名称（与 skill.yaml 中的 name 对应）
            params: 技能参数
        """
        if name in self._registry:
            return self._registry[name](**params)

        # 若是简单直通型技能，则自动包装单一 MCP 工具调用
        if name in self._simple_skills:
            cfg = self._simple_skills[name]
            tool_name = cfg["tool_name"]
            mapping: Dict[str, str] = cfg.get("param_mapping", {})
            tool_kwargs: Dict[str, Any] = {}
            for k, v in params.items():
                tool_key = mapping.get(k, k)
                tool_kwargs[tool_key] = v
            raw = self._call_mcp_tool(tool_name, **tool_kwargs)
            # 这里统一做一次温和截断，避免把底层工具的大块 JSON 直接抛给上层
            return self._shorten_text(raw, 1200)

        raise ValueError(f"未知技能：{name}")

    def call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        """直接调用底层 MCP 工具（绕过 Skill 路由）"""
        return self._call_mcp_tool(tool_name, **kwargs)

    # ===== MCP 通信层 =====

    def _init_mcp_session_sync(self):
        """使用同步方式初始化MCP会话"""
        try:
            client = httpx.Client(timeout=30.0)
            
            logger.info("正在建立SSE连接...")
            sse_response = client.get(
                MCP_SERVER_URL,
                headers={"Accept": "text/event-stream"}
            )
            
            session_id = sse_response.headers.get("mcp-session-id")
            if not session_id:
                raise Exception("无法获取session ID")
            
            logger.info(f"✅ 获取到session ID: {session_id}")
            
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "AstroAgent-SkillRouter",
                        "version": "1.0.0"
                    }
                },
                "id": 1
            }
            
            response = client.post(
                MCP_SERVER_URL,
                json=init_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"初始化失败: {response.status_code}")
            
            init_result = self._parse_sse_response(response.text)
            if init_result:
                logger.debug(f"初始化成功，服务器信息: {init_result.get('result', {}).get('serverInfo', {})}")
            
            notif_request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            
            client.post(
                MCP_SERVER_URL,
                json=notif_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id
                }
            )
            
            logger.info("获取工具列表...")
            list_request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2
            }
            
            response = client.post(
                MCP_SERVER_URL,
                json=list_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": session_id
                }
            )
            
            tools_result = self._parse_sse_response(response.text)
            if tools_result:
                tools_list = tools_result.get("result", {}).get("tools", [])
                logger.info(f"✅ 从服务器获取到 {len(tools_list)} 个工具")
            
            self._mcp_session_id = session_id
            self._mcp_initialized = True
            self._http_client = client
            
            logger.info(f"✅ MCP会话初始化成功，会话ID: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ MCP会话初始化失败: {e}")
            self._mcp_initialized = False
            self._http_client = None

    def _parse_sse_response(self, response_text: str) -> Optional[dict]:
        """解析 SSE 格式的响应"""
        try:
            lines = response_text.strip().split('\n')
            for line in lines:
                if line.startswith("data: "):
                    json_str = line[6:]
                    return json.loads(json_str)
            return None
        except Exception as e:
            logger.error(f"解析 SSE 响应失败: {e}")
            return None

    def _call_mcp_tool_internal(self, tool_name: str, **kwargs) -> str:
        """调用MCP工具（内部方法）"""
        if not self._mcp_initialized or not self._mcp_session_id:
            logger.error("❌ MCP会话未初始化")
            return f"错误：MCP会话未初始化，请检查桥服务器是否运行"
        
        try:
            processed_kwargs = {}
            for key, value in kwargs.items():
                if key in ['year', 'month', 'limit']:
                    try:
                        if isinstance(value, str) and value.isdigit():
                            processed_kwargs[key] = int(value)
                        elif isinstance(value, str):
                            processed_kwargs[key] = value
                        else:
                            processed_kwargs[key] = value
                    except:
                        processed_kwargs[key] = value
                else:
                    processed_kwargs[key] = value
            
            request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": processed_kwargs
                },
                "id": int(time.time() * 1000)
            }
            
            logger.debug(f"调用工具 {tool_name}，处理后的参数: {processed_kwargs}")
            
            response = self._http_client.post(
                MCP_SERVER_URL,
                json=request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": self._mcp_session_id
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                return f"HTTP错误: {response.status_code}"
            
            result = self._parse_sse_response(response.text)
            if not result:
                logger.error(f"无法解析响应: {response.text[:200]}")
                return f"解析响应失败"
            
            logger.debug(f"工具响应: {json.dumps(result, ensure_ascii=False)[:200]}")
            
            if "error" in result:
                error_msg = result["error"].get("message", "未知错误")
                error_code = result["error"].get("code", "")
                return f"工具调用错误 [{error_code}]: {error_msg}"
            
            if "result" in result:
                if "content" in result["result"]:
                    content = result["result"]["content"]
                    if isinstance(content, list) and len(content) > 0:
                        for item in content:
                            if item.get("type") == "text":
                                return item.get("text", "")
                
                if isinstance(result["result"], str):
                    return result["result"]
                
                return str(result["result"])
            
            logger.warning(f"未知响应格式: {result}")
            return str(result)
            
        except httpx.TimeoutException:
            logger.error(f"❌ MCP工具调用超时: {tool_name}")
            return f"调用工具超时，请稍后重试"
        except httpx.ConnectError:
            logger.error(f"❌ 无法连接到MCP服务器: {MCP_SERVER_URL}")
            return f"错误：无法连接到MCP服务器"
        except Exception as e:
            logger.error(f"❌ 调用工具 {tool_name} 失败: {e}")
            return f"调用工具失败: {str(e)}"

    # ===== 具体技能实现 =====

    def _observation_planner(
        self,
        date: Optional[str] = None,
        location: Optional[str] = None,
        duration: Optional[str] = None,
    ) -> str:
        """
        skill: observation-planner
        生成指定日期和地点的天文观测计划。
        参数设计参考 skill.yaml：
        - date: 观测日期，支持“今天”“明天”或 YYYY-MM-DD
        - location: 观测地点（城市名称或经纬度），可选；未提供时给出一般性观测建议
        - duration: 观测时段，如“整晚”“前半夜”“后半夜”
        """
        # 兼容 ReAct 工具解析可能把城市名误塞到 date 的情况：
        # 若 location 为空且 date 看起来不像日期，则将其视为地点。
        if not location and date:
            text = str(date).strip()
            # 简单判断是否“像日期”：包含数字日期或典型日期关键词
            if not self._is_date_like(text):
                location = text
                date = None

        obs_date = self._normalize_date(date)

        # 统一规范化地点信息，避免出现 {"location": "铁岭"} 这类结构透传到底层工具
        display_location: Optional[str] = None
        query_city: Optional[str] = None
        if location is not None:
            # dict 形式：优先 location/city/adcode/citycode
            if isinstance(location, dict):
                display_location = (
                    location.get("location")
                    or location.get("city")
                    or location.get("adcode")
                    or location.get("citycode")
                )
                query_city = location.get("city") or display_location
            else:
                text = str(location).strip()
                # 字符串恰好是 JSON 对象
                if text.startswith("{") and text.endswith("}"):
                    try:
                        obj = json.loads(text)
                        display_location = (
                            obj.get("location")
                            or obj.get("city")
                            or obj.get("adcode")
                            or obj.get("citycode")
                        )
                        query_city = obj.get("city") or display_location
                    except Exception:
                        display_location = text
                        query_city = text
                else:
                    display_location = text
                    query_city = text

        # 1) 查询天气（如果提供了可用城市信息）
        weather_brief = ""
        if query_city:
            weather_raw = self._call_mcp_tool(
                "get_weather",
                city=query_city,
                extensions="all",
            )
            weather_brief = self._summarize_weather(weather_raw)

        # 2) 查询一周天象（用于给出当日附近的天象背景）
        weekly_events = self._call_mcp_tool(
            "get_weekly_events",
            start_date=obs_date.strftime("%Y-%m-%d"),
        )

        # 3) 今晚推荐（仅当观测日期是今天时）
        tonight_best = ""
        today_str = datetime.now().strftime("%Y-%m-%d")
        if obs_date.strftime("%Y-%m-%d") == today_str:
            tonight_best = self._call_mcp_tool("get_tonight_best")

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
        plan_lines.append(self._shorten_text(weekly_events, 600))

        if tonight_best:
            plan_lines.append("\n三、系统给出的“今晚最佳观测目标”参考")
            plan_lines.append(self._shorten_text(tonight_best, 600))

        plan_lines.append("\n四、实用建议")
        plan_lines.append(
            "1. 根据云量与月相选择目标：若接近新月，可优先考虑深空天体；若接近满月，可多观测月面细节与行星。\n"
            "2. 提前抵达观测地，预留架台、极轴校准与试拍时间。\n"
            "3. 如湿度偏高或风力较大，请准备除露带、配重或防风措施。"
        )

        return "\n".join(plan_lines)

    def _celestial_events_forecast(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> str:
        """
        skill: celestial-events-forecast
        查询指定时间段的天象事件。
        目前实现：
        - 若时间跨度 <= 7 天：调用 get_weekly_events
        - 若时间跨度 > 7 天或未指定 end_date：调用 get_monthly_events
        - event_type 暂时仅用于在文案层提示筛选意图，不做严格机器过滤
        """
        if start_date:
            start_dt = self._normalize_date(start_date)
        else:
            start_dt = datetime.now()

        end_dt: Optional[datetime] = None
        if end_date:
            try:
                end_dt = self._normalize_date(end_date)
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

        # 时间跨度粗略判断
        use_weekly = False
        if end_dt:
            try:
                days = (end_dt - start_dt).days
                use_weekly = days <= 7
            except Exception:
                use_weekly = False
        else:
            # 未给 end_date，默认按一周预报
            use_weekly = True

        if use_weekly:
            body = self._call_mcp_tool(
                "get_weekly_events",
                start_date=start_dt.strftime("%Y-%m-%d"),
            )
        else:
            year = start_dt.year
            month = start_dt.month
            body = self._call_mcp_tool(
                "get_monthly_events",
                year=year,
                month=month,
            )

        description_prefix.append("\n下面是为你整理的天象预报：\n")
        description_prefix.append(self._shorten_text(body, 1200))
        return "\n".join(description_prefix)

    def _deep_sky_observing_guide(
        self,
        target: str,
        observer_location: Optional[str] = None,
        date: Optional[str] = None,
        equipment: Optional[str] = None,
    ) -> str:
        """
        skill: deep-sky-observing-guide
        为指定深空天体提供观测指导。
        """
        if not target:
            return "深空观测指导技能需要提供目标名称（target），例如“M31”或“猎户座大星云”。"

        obs_date = self._normalize_date(date) if date else datetime.now()

        # 1) 查询天体基本信息
        obj_info_raw = self._call_mcp_tool(
            "get_astrophysical_object_info",
            object_name=target,
        )

        # 2) 如果可能是星系目标，再尝试一次星系数据库
        galaxy_info_raw: Optional[str] = None
        if any(x in target.lower() for x in ["galaxy", "星系", "m31", "m33"]):
            galaxy_info_raw = self._call_mcp_tool(
                "get_galaxy_data",
                galaxy_name=target,
            )

        # 3) 可选：根据地点查询天气
        weather_brief = ""
        if observer_location:
            weather_raw = self._call_mcp_tool(
                "get_weather",
                city=observer_location,
                extensions="all",
            )
            weather_brief = self._summarize_weather(weather_raw)

        lines = [
            f"🎯 深空目标：{target}",
            f"📅 观测日期：{obs_date.strftime('%Y-%m-%d')}",
        ]
        if observer_location:
            lines.append(f"📍 观测地点：{observer_location}")
        if equipment:
            lines.append(f"🔭 计划使用设备：{equipment}")

        lines.append("\n一、目标基础信息（基于专业数据库）")
        lines.append(self._shorten_text(obj_info_raw, 600))
        if galaxy_info_raw:
            lines.append("\n补充：星系数据摘要")
            lines.append(self._shorten_text(galaxy_info_raw, 400))

        if weather_brief:
            lines.append("\n二、观测条件（天气简要）")
            lines.append(weather_brief)

        lines.append("\n三、观测建议")
        lines.append(
            "1. 建议选择无月光或月亮落下后 1–2 小时的时段进行深空观测。\n"
            "2. 若使用双筒或小口径望远镜，可优先寻找目标所在星座的亮星作“跳星”指引。\n"
            "3. 使用较低倍率（长焦距目镜）先锁定目标，再逐步提高放大倍率细看结构。"
        )

        return "\n".join(lines)

    def _neo_tracker(
        self,
        time_range: Optional[str] = None,
        min_size: Optional[float] = None,
        max_distance: Optional[float] = None,
        observable_only: Optional[bool] = None,
    ) -> str:
        """
        skill: neo-tracker
        追踪近地天体飞掠事件。
        - time_range: “未来30天”“本月”等自然语言时间范围
        - min_size: 最小直径（米）
        - max_distance: 最大距离（地月距离倍数）
        - observable_only: 是否仅返回有较大亮度、接近地球、理论上有观测价值的天体
        """
        start_date, end_date = self._parse_time_range(time_range)

        raw_json = self._call_mcp_tool(
            "get_neo_data",
            start_date=start_date,
            end_date=end_date,
            limit=50,
        )

        try:
            data = json.loads(raw_json)
        except Exception:
            # 工具已做过错误处理，直接原样返回
            return raw_json

        # NASA NEO feed 结构：near_earth_objects: { "YYYY-MM-DD": [ {...}, ... ] }
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

            # 尺寸筛选
            if min_size is not None and size is not None and size < min_size:
                continue
            # 距离筛选
            if max_distance is not None and miss_lunar is not None and miss_lunar > max_distance:
                continue

            # 观测价值粗筛（非常简化的逻辑）
            if observable_only:
                try:
                    abs_mag = float(obj.get("absolute_magnitude_h"))
                    # 绝对星等越小越亮，这里仅做非常粗的阈值
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
            return "在给定的时间范围和筛选条件下，没有找到明显具有观测价值的近地天体飞掠事件。"

        lines = [
            "📡 近地天体飞掠列表（按时间排序）：",
        ]
        filtered.sort(key=lambda x: (x["date"], x.get("miss_distance_lunar") or 1e9))
        for item in filtered[:20]:
            hazard = "⚠️ 潜在威胁小行星" if item["hazardous"] else ""
            lines.append(
                f"- 日期 {item['date']}，目标 {item['name']}，"
                f"估计直径约 {item['size_m']:.0f} m，最近距离约 {item['miss_distance_lunar']:.2f} 个地月距离 {hazard}"
            )

        lines.append(
            "\n注：以上数据来自 NASA NEO 数据接口，是否实际可见还与亮度、天空背景和观测设备有关。"
        )
        return "\n".join(lines)

    def _astrophotography_calculator(
        self,
        target: str,
        camera: str,
        telescope: Optional[str] = None,
        mount: Optional[str] = None,
        location: Optional[str] = None,
        date: Optional[str] = None,
    ) -> str:
        """
        skill: astrophotography-calculator
        计算天文摄影参数。
        当前实现主要为经验规则型计算，不依赖 MCP 工具。
        """
        obs_date = self._normalize_date(date) if date else datetime.now()

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

        # 非严格的“500 规则”估算（假设 24mm 全画幅）
        lines.append("\n一、曝光时间估算（星点不拖尾的经验值）")
        lines.append(
            "若使用广角/标准镜头并在赤道仪跟踪下：\n"
            "- 星野/银河：单张 20–60 秒，ISO 1600–6400，光圈尽量开大。\n"
            "- 星云/星团：根据目标亮度，单张 120–300 秒，ISO 800–3200。\n"
            "若非跟踪（固定三脚架），可按“500 规则”粗略估计：曝光秒数 ≈ 500 / 焦距（全画幅等效）。"
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

    def _celestial_position_calculator(
        self,
        target: str,
        datetime: str,
        location: str,
        output_format: Optional[str] = None,
    ) -> str:
        """
        skill: celestial-position-calculator
        计算天体在指定时间的位置。
        当前主要支持太阳系行星（通过 get_planet_position），输出以赤道坐标为主。
        """
        if not target:
            return "天体位置计算技能需要提供目标名称（target），例如“mars”“jupiter”等。"

        obs_time = self._normalize_datetime(datetime)
        lat, lon = self._parse_location(location)
        if lat is None or lon is None:
            return "暂时无法解析 location 为经纬度，请提供“纬度,经度”形式，如“39.9,116.4”。"

        result_raw = self._call_mcp_tool(
            "get_planet_position",
            planet_name=target,
            observation_time=obs_time.isoformat(),
            latitude=lat,
            longitude=lon,
        )

        # 工具已返回 JSON/dict 风格，这里只做轻度包装
        body = self._shorten_text(result_raw, 600)
        fmt = (output_format or "radec").lower()

        header = (
            f"🪐 天体位置计算\n"
            f"- 目标：{target}\n"
            f"- 时间：{obs_time.isoformat()}\n"
            f"- 观测点：纬度 {lat}，经度 {lon}\n"
            f"- 输出坐标系偏好：{fmt}（当前实现以赤道坐标为主，若需 altaz 请在最终回答中由 LLM 补充说明）\n"
        )

        return header + "\n原始计算结果（来自底层工具）：\n" + body

    # ===== 辅助方法 =====

    def _normalize_date(self, date_str: Optional[str]) -> datetime:
        """将“今天/明天/2026-03-14”等统一为 datetime.date。"""
        today = datetime.now()
        if not date_str:
            return today
        text = str(date_str).strip()
        if text in ("今天", "今日", "today"):
            return today
        if text in ("明天", "次日", "tomorrow"):
            return today + timedelta(days=1)
        # 尝试从文本中提取类似 YYYY-MM-DD 或 YYYY/MM/DD 的日期片段，
        # 以兼容诸如“2026-08-01 天象预报范围：...”之类的复合字符串。
        m = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", text)
        if m:
            candidate = m.group(1).replace("/", "-")
            try:
                return datetime.strptime(candidate, "%Y-%m-%d")
            except Exception:
                pass
        # 兜底：直接尝试常见格式解析
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
        # 最后兜底当前日期
        return today

    def _is_date_like(self, text: str) -> bool:
        """判断一个字符串是否“像日期”，用于纠正常被误塞入 date 的地点名称。"""
        t = text.strip()
        if not t:
            return False
        if t in ("今天", "明天", "今日", "次日", "today", "tomorrow"):
            return True
        if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", t):
            return True
        return False

    def _normalize_datetime(self, dt_str: str) -> datetime:
        """尽量把时间字符串解析成 datetime，失败则使用当前时间。"""
        if not dt_str:
            return datetime.now()
        text = str(dt_str).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
        return datetime.now()

    def _parse_location(self, location: str) -> tuple[Optional[float], Optional[float]]:
        """
        解析 location 为 (lat, lon)。
        当前实现只处理简单的“纬度,经度”数字形式。
        """
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

    def _parse_time_range(self, time_range: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """
        将自然语言 time_range 转换为 (start_date, end_date)，格式 YYYY-MM-DD。
        """
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

    def _summarize_weather(self, raw: str) -> str:
        """从 get_weather 的原始 JSON 文本中提取一个简要观测建议。"""
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

    def _shorten_text(self, text: Any, max_len: int) -> str:
        """将任意对象转为字符串，并在长度超过 max_len 时截断。"""
        if text is None:
            return ""
        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

