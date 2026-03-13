import json
from langchain_core.tools import Tool
from typing import List, Any, Callable, Optional
from logger import logger


class AgentTools:
    def __init__(
        self,
        rag_retriever: Any,
        skill_router: Any,
    ):
        self._rag = rag_retriever
        self._skill_router = skill_router
        self._tools: Optional[List[Tool]] = None

    def get_tools(self) -> List[Tool]:
        if self._tools is None:
            self._tools = self._init_tools()
        return self._tools

    def _init_tools(self) -> List[Tool]:
        tools: List[Tool] = []

        tools.append(
            Tool(
                name="RAGRetrieve",
                func=self._rag_retrieve,
                description="使用本地RAG知识库检索天文知识、概念解释、历史资料等。参数：query（查询语句，中文即可）。",
            )
        )

        tools.append(
            Tool(
                name="WeatherLookup",
                func=self._weather_lookup_skill,
                description=(
                    "查询指定城市的观测相关天气信息（skill: weather-lookup，对应 MCP 工具 get_weather）。\n"
                    "参数：city（城市名称或adcode，可选），"
                    "location（城市名称，和 city 等价，可选），"
                    "extensions（\"base\" 实时 或 \"all\" 预报，默认 all）。"
                ),
            )
        )

        tools.append(
            Tool(
                name="ObservationPlanner",
                func=self._observation_planner_skill,
                description=(
                    "生成指定日期和地点的天文观测计划（skill: observation-planner）。\n"
                    '参数：date（观测日期，可为"今天""明天"或YYYY-MM-DD，可选），'
                    'location（观测地点，城市名或"纬度,经度"，必填），'
                    'duration（观测时段，如"整夜""前半夜""后半夜"，可选）。'
                ),
            )
        )

        tools.append(
            Tool(
                name="CelestialEventsForecast",
                func=self._celestial_events_forecast_skill,
                description=(
                    "查询指定时间段的天象事件（skill: celestial-events-forecast）。\n"
                    "参数：start_date（开始日期YYYY-MM-DD，可选），"
                    "end_date（结束日期YYYY-MM-DD，可选），"
                    "event_type（事件类型，如'流星雨''行星合月''月食'，可选，用于意图说明）。"
                ),
            )
        )

        tools.append(
            Tool(
                name="DeepSkyObservingGuide",
                func=self._deep_sky_observing_guide_skill,
                description=(
                    "为指定深空天体提供观测指导（skill: deep-sky-observing-guide）。\n"
                    "参数：target（天体名称，如'M31''猎户座大星云'，必填），"
                    "observer_location（观测者位置，可选），"
                    "date（观测日期，可选），"
                    "equipment（设备描述，如'裸眼''双筒''8寸望远镜'，可选）。"
                ),
            )
        )

        tools.append(
            Tool(
                name="NEOTracker",
                func=self._neo_tracker_skill,
                description=(
                    "追踪近地天体飞掠事件（skill: neo-tracker）。\n"
                    "参数：time_range（时间范围，如'未来30天''本月'，可选），"
                    "min_size（最小直径，单位米，可选），"
                    "max_distance（最大距离，单位地月距离倍数，可选），"
                    "observable_only（是否只返回具有观测价值的目标，布尔值，可选）。"
                ),
            )
        )

        tools.append(
            Tool(
                name="AstrophotographyCalculator",
                func=self._astrophotography_calculator_skill,
                description=(
                    "计算天文摄影参数与拍摄建议（skill: astrophotography-calculator）。\n"
                    "参数：target（拍摄目标，必填），"
                    "camera（相机型号，必填），"
                    "telescope（望远镜型号或焦距，可选），"
                    "mount（赤道仪型号，可选），"
                    "location（拍摄地点，可选），"
                    "date（拍摄日期，可选）。"
                ),
            )
        )

        tools.append(
            Tool(
                name="CelestialPositionCalculator",
                func=self._celestial_position_calculator_skill,
                description=(
                    "计算天体在指定时间的位置（skill: celestial-position-calculator）。\n"
                    "参数：target（目标名称，如'mars''jupiter'等，必填），"
                    "datetime（观测时间，建议YYYY-MM-DD HH:MM 格式，可选，默认当前时间），"
                    "location（观测地点，经纬度'纬度,经度'形式，可选），"
                    "output_format（输出坐标格式，如'altaz''radec'，可选）。"
                ),
            )
        )

        logger.info(f"✅ 成功注册 {len(tools)} 个高层技能工具（含RAG）")
        return tools

    def _rag_retrieve(self, query: str) -> str:
        if isinstance(query, dict):
            query = query.get('query', query)
        elif isinstance(query, str):
            try:
                if query.strip().startswith('{'):
                    data = json.loads(query)
                    if isinstance(data, dict) and 'query' in data:
                        query = data['query']
            except:
                pass
        return self._rag.get_relevant_context(query)

    def _weather_lookup_skill(
        self,
        city: str = None,
        location: str = None,
        extensions: str = "all",
    ) -> str:
        if isinstance(city, dict):
            data = city
            city = data.get('city') or data.get('location')
            location = data.get('location') or data.get('city')
            extensions = data.get('extensions', extensions)
        elif isinstance(city, str):
            try:
                if city.strip().startswith('{'):
                    data = json.loads(city)
                    if isinstance(data, dict):
                        city = data.get('city') or data.get('location')
                        location = data.get('location') or data.get('city')
                        extensions = data.get('extensions', extensions)
            except:
                pass
        target = city or location
        return self._skill_router.call(
            "weather-lookup",
            city=target,
            extensions=extensions,
        )

    def _observation_planner_skill(
        self,
        date: str = None,
        location: str = None,
        duration: str = None,
    ) -> str:
        if isinstance(date, dict):
            data = date
            date = data.get('date')
            location = data.get('location')
            duration = data.get('duration')
        elif isinstance(date, str):
            try:
                if date.strip().startswith('{'):
                    data = json.loads(date)
                    if isinstance(data, dict):
                        date = data.get('date')
                        location = data.get('location')
                        duration = data.get('duration')
            except:
                pass
        return self._skill_router.call(
            "observation-planner",
            date=date,
            location=location,
            duration=duration,
        )

    def _celestial_events_forecast_skill(
        self,
        start_date: str = None,
        end_date: str = None,
        event_type: str = None,
    ) -> str:
        if isinstance(start_date, dict):
            data = start_date
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            event_type = data.get('event_type')
        elif isinstance(start_date, str):
            try:
                if start_date.strip().startswith('{'):
                    data = json.loads(start_date)
                    if isinstance(data, dict):
                        start_date = data.get('start_date')
                        end_date = data.get('end_date')
                        event_type = data.get('event_type')
            except:
                pass
        return self._skill_router.call(
            "celestial-events-forecast",
            start_date=start_date,
            end_date=end_date,
            event_type=event_type,
        )

    def _deep_sky_observing_guide_skill(
        self,
        target: str,
        observer_location: str = None,
        date: str = None,
        equipment: str = None,
    ) -> str:
        if isinstance(target, dict):
            data = target
            target = data.get('target')
            observer_location = data.get('observer_location')
            date = data.get('date')
            equipment = data.get('equipment')
        elif isinstance(target, str):
            try:
                if target.strip().startswith('{'):
                    data = json.loads(target)
                    if isinstance(data, dict):
                        target = data.get('target')
                        observer_location = data.get('observer_location')
                        date = data.get('date')
                        equipment = data.get('equipment')
            except:
                pass
        return self._skill_router.call(
            "deep-sky-observing-guide",
            target=target,
            observer_location=observer_location,
            date=date,
            equipment=equipment,
        )

    def _neo_tracker_skill(
        self,
        time_range: str = None,
        min_size: float = None,
        max_distance: float = None,
        observable_only: bool = None,
    ) -> str:
        if isinstance(time_range, dict):
            data = time_range
            time_range = data.get('time_range')
            min_size = data.get('min_size')
            max_distance = data.get('max_distance')
            observable_only = data.get('observable_only')
        elif isinstance(time_range, str):
            try:
                if time_range.strip().startswith('{'):
                    data = json.loads(time_range)
                    if isinstance(data, dict):
                        time_range = data.get('time_range')
                        min_size = data.get('min_size')
                        max_distance = data.get('max_distance')
                        observable_only = data.get('observable_only')
            except:
                pass
        return self._skill_router.call(
            "neo-tracker",
            time_range=time_range,
            min_size=min_size,
            max_distance=max_distance,
            observable_only=observable_only,
        )

    def _astrophotography_calculator_skill(
        self,
        target: str,
        camera: str = None,
        telescope: str = None,
        mount: str = None,
        location: str = None,
        date: str = None,
    ) -> str:
        if isinstance(target, dict):
            data = target
            target = data.get('target')
            camera = data.get('camera')
            telescope = data.get('telescope')
            mount = data.get('mount')
            location = data.get('location')
            date = data.get('date')
        elif isinstance(target, str):
            try:
                if target.strip().startswith('{'):
                    # 移除可能的注释部分
                    clean_target = target.split('#')[0].strip()
                    data = json.loads(clean_target)
                    if isinstance(data, dict):
                        target = data.get('target')
                        camera = data.get('camera')
                        telescope = data.get('telescope')
                        mount = data.get('mount')
                        location = data.get('location')
                        date = data.get('date')
            except:
                pass
        return self._skill_router.call(
            "astrophotography-calculator",
            target=target,
            camera=camera,
            telescope=telescope,
            mount=mount,
            location=location,
            date=date,
        )

    def _celestial_position_calculator_skill(
        self,
        target: str,
        datetime: str = None,
        location: str = None,
        output_format: str = None,
    ) -> str:
        if isinstance(target, dict):
            data = target
            target = data.get('target')
            datetime = data.get('datetime')
            location = data.get('location')
            output_format = data.get('output_format')
        elif isinstance(target, str):
            try:
                if target.strip().startswith('{'):
                    # 移除可能的注释部分
                    clean_target = target.split('#')[0].strip()
                    data = json.loads(clean_target)
                    if isinstance(data, dict):
                        target = data.get('target')
                        datetime = data.get('datetime')
                        location = data.get('location')
                        output_format = data.get('output_format')
            except:
                pass
        return self._skill_router.call(
            "celestial-position-calculator",
            target=target,
            datetime=datetime,
            location=location,
            output_format=output_format,
        )
