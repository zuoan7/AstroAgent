# -*- coding: utf-8 -*-
"""
天象预测器模块 - 月相、日出日落、天象事件预报
"""

import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from skyfield import almanac
from skyfield.api import wgs84
from .base import EphemerisManager
from src.core.config import settings
from src.agent.param_parser import ParamParser

from src.core.logger import logger


class EventsPredictor:
    """
    天象预测器
    
    提供月相计算、日出日落时间、行星可见性、天象事件预报等功能。
    """
    
    def __init__(self, ephemeris: EphemerisManager = None, location=None):
        # 使用传入的星历管理器或创建新的（确保单例）
        self.ephemeris = ephemeris or EphemerisManager()
        
        # 设置观测位置
        if location:
            self.lat, self.lon = location
        else:
            self.lat, self.lon = settings.DEFAULT_LOCATION
        
        # 创建观测点
        if self.ephemeris.is_loaded:
            self.earth = self.ephemeris.earth
            self.observer_location = wgs84.latlon(self.lat, self.lon)
        
        # 加载特殊天象数据
        self.special_events = self._load_special_events()
    
    def _load_special_events(self) -> list:
        """从YAML配置文件加载特殊天象数据，支持按年份加载"""
        events = self._load_events_from_yaml()
        if events is not None:
            return events

        logger.warning("YAML天象数据加载失败，使用内置降级数据")
        return self._get_fallback_events()

    def _load_events_from_yaml(self) -> Optional[list]:
        """从YAML文件加载天象事件数据"""
        try:
            import yaml
            from src.core.config import resolve_path
            from pathlib import Path

            data_dir = resolve_path(settings.ASTRONOMY_DATA_DIR)
            year = datetime.now().year
            yaml_path = Path(data_dir) / f"events_{year}.yaml"

            if not yaml_path.exists():
                for y in range(year - 1, year + 2):
                    alt_path = Path(data_dir) / f"events_{y}.yaml"
                    if alt_path.exists():
                        yaml_path = alt_path
                        break

            if not yaml_path.exists():
                logger.warning(f"未找到天象数据文件: {yaml_path}")
                return None

            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not data:
                return None

            events = []

            for shower in data.get("meteor_showers", []):
                events.append({
                    "date": shower["date"],
                    "event": shower["event"],
                    "description": shower["description"],
                    "type": "meteor_shower",
                    "peak_time": shower.get("peak_time", ""),
                })

            for eclipse in data.get("eclipses", []):
                events.append({
                    "date": eclipse["date"],
                    "event": eclipse["event"],
                    "description": eclipse["description"],
                    "type": "eclipse",
                })

            for conj in data.get("conjunctions", []):
                events.append({
                    "date": conj["date"],
                    "event": conj["event"],
                    "description": conj["description"],
                    "type": "conjunction",
                })

            for opp in data.get("oppositions", []):
                events.append({
                    "date": opp["date"],
                    "event": opp["event"],
                    "description": opp["description"],
                    "type": "opposition",
                })

            for season in data.get("seasons", []):
                events.append({
                    "date": season["date"],
                    "event": season["event"],
                    "description": season["description"],
                    "type": "season",
                })

            events.sort(key=lambda x: x["date"])
            logger.info(f"✅ 从YAML加载了 {len(events)} 个天象事件: {yaml_path}")
            return events

        except ImportError:
            logger.warning("PyYAML未安装，无法加载YAML天象数据")
            return None
        except Exception as e:
            logger.error(f"加载YAML天象数据失败: {e}")
            return None

    def _get_fallback_events(self) -> list:
        """内置降级数据，仅在YAML加载失败时使用"""
        return []
    
    def get_moon_phase(self, date) -> tuple:
        """
        计算指定日期的月相
        
        Args:
            date: datetime对象
            
        Returns:
            (月相名称, 月相描述) 元组
        """
        if not self.ephemeris.is_loaded:
            return ("⚠️ 数据未加载", "无法计算月相")
        
        t = self.ephemeris.timescale.utc(date.year, date.month, date.day)
        
        # 获取月亮和太阳的位置
        e = self.ephemeris.planets['earth']
        m = self.ephemeris.planets['moon']
        s = self.ephemeris.planets['sun']
        
        moon_earth = e.at(t).observe(m).apparent()
        sun_earth = e.at(t).observe(s).apparent()
        
        moon_ra, _, _ = moon_earth.radec()
        sun_ra, _, _ = sun_earth.radec()
        
        # 计算相位角
        ra_diff = (moon_ra.hours - sun_ra.hours) * 15
        if ra_diff < 0:
            ra_diff += 360
        
        phase_angle = ra_diff
        
        # 根据角度判断月相
        for threshold, name, desc in settings.MOON_PHASE_THRESHOLDS:
            if phase_angle < threshold:
                return (name, desc)
        
        return (settings.MOON_PHASE_THRESHOLDS[0][1], settings.MOON_PHASE_THRESHOLDS[0][2])
    
    def get_sunrise_sunset(self, date) -> tuple:
        """
        计算指定日期的日出日落时间
        
        Args:
            date: datetime对象
            
        Returns:
            (日出时间, 日落时间) 元组
        """
        if not self.ephemeris.is_loaded:
            return (None, None)
        
        t0 = self.ephemeris.timescale.utc(date.year, date.month, date.day, 0)
        t1 = self.ephemeris.timescale.utc(date.year, date.month, date.day, 23, 59)
        
        f = almanac.sunrise_sunset(self.ephemeris.planets, self.observer_location)
        times, events = almanac.find_discrete(t0, t1, f)
        
        sunrise = None
        sunset = None
        
        for t, e in zip(times, events):
            if e == 1:  # 日出
                sunrise = t.utc_datetime()
            elif e == 0:  # 日落
                sunset = t.utc_datetime()
        
        return (sunrise, sunset)
    
    def get_visible_planets(self, date) -> list:
        """
        获取指定日期晚上8点可见的行星
        
        Args:
            date: datetime对象
            
        Returns:
            可见行星列表（包含名称和方向）
        """
        if not self.ephemeris.is_loaded:
            return []
        
        t = self.ephemeris.timescale.utc(date.year, date.month, date.day, 20)
        
        planets_info = [
            {"name_cn": settings.PLANET_NAMES_CN.get(p, p), 
             "obj": self.ephemeris.planets[settings.PLANET_MAPPING[p]]}
            for p in ['mercury', 'venus', 'mars', 'jupiter', 'saturn']
        ]
        
        visible = []
        
        for p in planets_info:
            observer = self.earth + self.observer_location
            astrometric = observer.at(t).observe(p['obj']).apparent()
            alt, az, _ = astrometric.altaz()
            
            if alt.degrees > 15:
                direction = ParamParser.get_direction_from_azimuth(az.degrees)
                visible.append(f"{p['name_cn']}（{direction}天空）")
        
        return visible
    
    def get_weekly_events(self, start_date=None) -> str:
        """
        获取未来一周的天象预报
        
        Args:
            start_date: 开始日期（可选）
            
        Returns:
            格式化的周预报字符串
        """
        try:
            now = datetime.now()
            
            # 解析开始日期
            current_date = ParamParser.parse_date(start_date, default=now)
            
            # 计算结束日期
            end_date = current_date + timedelta(days=7)
            
            # 筛选本周特殊天象
            weekly_events = [
                event for event in self.special_events
                if current_date.date() <= datetime.strptime(event['date'], "%Y-%m-%d").date() < end_date.date()
            ]
            
            # 生成预报文本
            start_str = current_date.strftime("%Y-%m-%d")
            end_str = (end_date - timedelta(days=1)).strftime("%Y-%m-%d")
            
            forecast = f"🌌 **{start_str} 至 {end_str} 一周天象预报**\n\n"
            
            if weekly_events:
                forecast += "✨ **本周特殊天象**\n"
                for event in weekly_events:
                    forecast += f"• **{event['date']}** {event['event']}\n"
                    forecast += f"  {event['description']}\n"
            else:
                forecast += "本周没有特殊天象。\n"
                forecast += "但你可以关注日常可见的行星和月相变化。\n\n"
            
            # 添加每日月相
            forecast += "🌙 **本周月相变化**\n"
            for i in range(7):
                forecast_date = current_date + timedelta(days=i)
                moon_phase, _ = self.get_moon_phase(forecast_date)
                forecast += f"• {forecast_date.strftime('%Y-%m-%d')}: {moon_phase}\n"
            
            # 添加行星可见性建议
            forecast += "\n🪐 **本周行星可见性**\n"
            month = current_date.month
            visibility = settings.PLANET_VISIBILITY_BY_MONTH.get(month, {})
            
            if visibility.get('best'):
                forecast += f"• {visibility['best']}：整夜可见，是本周最佳观测目标\n"
            if visibility.get('morning'):
                forecast += f"• {visibility['morning']}：早晨东方低空可见\n"
            if visibility.get('evening'):
                forecast += f"• {visibility['evening']}：傍晚西方低空可见\n"
            if visibility.get('evening2'):
                forecast += f"• {visibility['evening2']}：前半夜可见\n"
            if visibility.get('late_night'):
                forecast += f"• {visibility['late_night']}：后半夜可见\n"
            
            logger.debug(f"生成周预报: {start_str} 至 {end_str}")
            return forecast
            
        except Exception as e:
            logger.error(f"生成周预报失败: {e}", exc_info=True)
            return f"错误：{str(e)}"
    
    def get_monthly_events(self, year=None, month=None) -> str:
        """
        获取未来一个月的天象预报
        
        Args:
            year: 年份（可选）
            month: 月份（可选）
            
        Returns:
            格式化的月预报字符串
        """
        try:
            now = datetime.now()
            
            # 使用 ParamParser 统一解析参数
            now = datetime.now()
            
            # 解析年份和月份参数
            params = ParamParser.parse_tool_input(
                {"year": year, "month": month},
                expected_params={"year": None, "month": None}
            )
            
            year_val = ParamParser.safe_int(params.get("year"), default=now.year)
            month_val = ParamParser.safe_int(params.get("month"), default=None)
            
            # 如果没有指定月份，使用当前月份的下一月
            if month_val is None:
                if year_val == now.year:
                    month_val = now.month + 1
                    if month_val > 12:
                        month_val = 1
                        year_val += 1
                else:
                    month_val = 1
            
            # 确保月份有效
            if month_val < 1 or month_val > 12:
                month_val = 1
            
            # 筛选本月特殊天象
            monthly_events = [
                event for event in self.special_events
                if datetime.strptime(event['date'], "%Y-%m-%d").year == year_val
                and datetime.strptime(event['date'], "%Y-%m-%d").month == month_val
            ]
            
            # 生成预报文本
            month_names = ["1月", "2月", "3月", "4月", "5月", "6月",
                          "7月", "8月", "9月", "10月", "11月", "12月"]
            
            forecast = f"🔭 **{year_val}年{month_names[month_val-1]}天象预报**\n\n"
            
            if monthly_events:
                forecast += "✨ **本月特殊天象**\n"
                for event in monthly_events:
                    forecast += f"• **{event['date']}** {event['event']}\n"
                    forecast += f"  {event['description']}\n"
            else:
                forecast += "本月没有特殊天象。\n"
                forecast += "但你可以关注日常可见的行星和月相变化。\n\n"
            
            # 添加行星可见性建议
            forecast += "🪐 **本月行星可见性**\n"
            visibility = settings.PLANET_VISIBILITY_BY_MONTH.get(month_val, {})
            
            if visibility.get('best'):
                forecast += f"• {visibility['best']}：整夜可见，是本月最佳观测目标\n"
            if visibility.get('morning'):
                forecast += f"• {visibility['morning']}：早晨东方低空可见\n"
            if visibility.get('evening'):
                forecast += f"• {visibility['evening']}：傍晚可见\n"
            if visibility.get('evening2'):
                forecast += f"• {visibility['evening2']}：前半夜可见\n"
            if visibility.get('late_night'):
                forecast += f"• {visibility['late_night']}：后半夜可见\n"
            
            logger.debug(f"生成月预报: {year_val}年{month_val}月")
            return forecast
            
        except Exception as e:
            logger.error(f"生成月预报失败: {e}", exc_info=True)
            return f"错误：{str(e)}"
    
    def get_tonight_best(self) -> str:
        """
        获取今晚最佳观测目标
        
        Returns:
            格式化的观测指南字符串
        """
        today = datetime.now()
        day_str = today.strftime("%Y-%m-%d")
        
        # 检查今晚的特殊天象
        special_tonight = None
        for event in self.special_events:
            if event["date"] == day_str:
                special_tonight = event
                break
        
        # 获取各项数据
        moon_phase, moon_desc = self.get_moon_phase(today)
        planets = self.get_visible_planets(today)
        sunrise, sunset = self.get_sunrise_sunset(today)
        
        # 生成响应
        response = f"🌙 **今晚（{day_str}）观测指南**\n\n"
        
        if special_tonight:
            response += f"✨ **特别推荐**：{special_tonight['event']}\n"
            response += f"  {special_tonight['description']}\n\n"
        
        response += f"【月相】{moon_phase}\n"
        response += f"  {moon_desc}\n\n"
        
        if planets:
            response += f"【可见行星】\n"
            for p in planets:
                response += f"  • {p}\n"
        else:
            response += f"【可见行星】今晚无明显可见行星\n"
        
        response += f"\n【时间信息】\n"
        response += f"  • 日落：{sunset.strftime('%H:%M') if sunset else '未知'}\n"
        response += f"  • 日出：{sunrise.strftime('%H:%M') if sunrise else '未知'}\n"
        
        # 根据月相给出建议
        if "新月" in moon_phase:
            response += f"\n💡 {settings.OBSERVING_TIPS_TEMPLATES['new_moon']}"
        elif "满月" in moon_phase:
            response += f"\n💡 {settings.OBSERVING_TIPS_TEMPLATES['full_moon']}"
        
        return response


__all__ = ['EventsPredictor']
