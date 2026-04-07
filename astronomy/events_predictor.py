# -*- coding: utf-8 -*-
"""
天象预测器模块 - 月相、日出日落、天象事件预报
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from skyfield import almanac
from skyfield.api import wgs84
from .base import EphemerisManager
from agent.param_parser import ParamParser
from utils.helpers import (
    parse_date,
    get_direction_from_azimuth,
)
from constants import (
    PLANET_MAPPING,
    PLANET_NAMES_CN,
    PLANET_MAX_MAGNITUDE,
    DEFAULT_LOCATION,
    MOON_PHASE_THRESHOLDS,
    PLANET_VISIBILITY_BY_MONTH,
)

logger = logging.getLogger(__name__)


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
            self.lat, self.lon = DEFAULT_LOCATION
        
        # 创建观测点
        if self.ephemeris.is_loaded:
            self.earth = self.ephemeris.earth
            self.observer_location = wgs84.latlon(self.lat, self.lon)
        
        # 加载特殊天象数据
        self.special_events = self._load_special_events()
    
    def _load_special_events(self) -> list:
        """加载特殊天象数据"""
        return [
            # 流星雨
            {"date": "2026-01-03", "event": "象限仪座流星雨极大",
             "description": "每小时约110颗，月光干扰小，后半夜可见",
             "type": "meteor_shower", "peak_time": "凌晨"},
            
            {"date": "2026-04-22", "event": "天琴座流星雨极大",
             "description": "每小时约18颗，无月光干扰",
             "type": "meteor_shower", "peak_time": "后半夜"},
            
            {"date": "2026-05-05", "event": "宝瓶座η流星雨极大",
             "description": "每小时约40颗，凌晨可见",
             "type": "meteor_shower", "peak_time": "凌晨"},
            
            {"date": "2026-08-12", "event": "英仙座流星雨极大",
             "description": "每小时约100颗，年度最佳流星雨！",
             "type": "meteor_shower", "peak_time": "整夜"},
            
            {"date": "2026-10-21", "event": "猎户座流星雨极大",
             "description": "每小时约20颗，后半夜可见",
             "type": "meteor_shower", "peak_time": "后半夜"},
            
            {"date": "2026-11-17", "event": "狮子座流星雨极大",
             "description": "每小时约15颗，可能有爆发",
             "type": "meteor_shower", "peak_time": "后半夜"},
            
            {"date": "2026-12-14", "event": "双子座流星雨极大",
             "description": "每小时约120颗，整夜可见，年度最佳！",
             "type": "meteor_shower", "peak_time": "整夜"},
            
            # 日月食
            {"date": "2026-03-03", "event": "月全食",
             "description": "已发生，亚洲、欧洲可见",
             "type": "eclipse"},
            
            {"date": "2026-08-12", "event": "日偏食",
             "description": "北京时间20:02开始，最大遮挡约80%",
             "type": "eclipse"},
            
            {"date": "2026-08-28", "event": "月偏食",
             "description": "凌晨4:23开始，可见红月亮",
             "type": "eclipse"},
            
            # 行星特殊位置
            {"date": "2026-03-09", "event": "金星合土星",
             "description": "日落后西方低空，两者相距约1.5度，肉眼可见",
             "type": "conjunction"},
            
            {"date": "2026-06-08", "event": "金星合木星",
             "description": "日落后西方低空，两者相距约0.5度，非常壮观",
             "type": "conjunction"},
            
            {"date": "2026-10-04", "event": "土星冲日",
             "description": "土星整夜可见，是观测土星环的最佳时机",
             "type": "opposition"},
            
            {"date": "2026-11-26", "event": "天王星冲日",
             "description": "天王星最亮，可用望远镜观测",
             "type": "opposition"},
            
            # 节气
            {"date": "2026-03-20", "event": "春分",
             "description": "22:45:58，太阳直射赤道，昼夜等长",
             "type": "season"},
            
            {"date": "2026-06-21", "event": "夏至",
             "description": "16:25:19，一年中白昼最长",
             "type": "season"},
            
            {"date": "2026-09-23", "event": "秋分",
             "description": "08:08:12，昼夜等长",
             "type": "season"},
            
            {"date": "2026-12-22", "event": "冬至",
             "description": "04:48:24，一年中夜晚最长",
             "type": "season"},
        ]
    
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
        for threshold, name, desc in MOON_PHASE_THRESHOLDS:
            if phase_angle < threshold:
                return (name, desc)
        
        return (MOON_PHASE_THRESHOLDS[0][1], MOON_PHASE_THRESHOLDS[0][2])
    
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
            {"name_cn": PLANET_NAMES_CN.get(p, p), 
             "obj": self.ephemeris.planets[PLANET_MAPPING[p]]}
            for p in ['mercury', 'venus', 'mars', 'jupiter', 'saturn']
        ]
        
        visible = []
        
        for p in planets_info:
            observer = self.earth + self.observer_location
            astrometric = observer.at(t).observe(p['obj']).apparent()
            alt, az, _ = astrometric.altaz()
            
            if alt.degrees > 15:
                direction = get_direction_from_azimuth(az.degrees)
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
            current_date = parse_date(start_date, default=now)
            
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
            visibility = PLANET_VISIBILITY_BY_MONTH.get(month, {})
            
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
            visibility = PLANET_VISIBILITY_BY_MONTH.get(month_val, {})
            
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
        from constants import OBSERVING_TIPS_TEMPLATES
        if "新月" in moon_phase:
            response += f"\n💡 {OBSERVING_TIPS_TEMPLATES['new_moon']}"
        elif "满月" in moon_phase:
            response += f"\n💡 {OBSERVING_TIPS_TEMPLATES['full_moon']}"
        
        return response


__all__ = ['EventsPredictor']
