# -*- coding: utf-8 -*-
"""
行星计算模块 - 行星位置、坐标转换、升起落下时间等
"""

import logging
from datetime import datetime
from skyfield.api import wgs84, load
from astropy.coordinates import ICRS, FK5, SkyCoord
import ephem

from .base import EphemerisManager
from utils.helpers import (
    parse_mixed_input,
    parse_date,
    parse_coordinate_string,
)
from constants import (
    PLANET_MAPPING,
    VALID_PLANETS,
    SUPPORTED_BODIES,
)

logger = logging.getLogger(__name__)


class PlanetaryCalculator:
    """
    行星位置计算器
    
    提供行星位置查询、坐标转换、天体升降时间等功能。
    """
    
    def __init__(self, ephemeris: EphemerisManager):
        self.ephemeris = ephemeris
    
    def get_planet_position(self, planet_name, observation_time=None, latitude=None, longitude=None):
        """
        获取行星位置
        
        Args:
            planet_name: 行星名称（'mercury', 'venus', 'mars'等）
            observation_time: 观测时间（可选）
            latitude: 观测点纬度（度，可选）
            longitude: 观测点经度（度，可选）
            
        Returns:
            包含赤经、赤纬和距离的字典
        """
        if not self.ephemeris.is_loaded:
            from core.errors import ErrorHandler
            error = ErrorHandler.create_tool_error(
                "get_planet_position",
                "行星数据未加载，无法计算行星位置"
            )
            return error.to_dict()
        
        try:
            # 统一参数解析
            params = parse_mixed_input(planet_name, {
                "planet_name": None,
                "observation_time": None,
                "latitude": None,
                "longitude": None
            })
            
            if params.get("planet_name"):
                planet_name = params["planet_name"]
                observation_time = params.get("observation_time") or observation_time
                latitude = params.get("latitude") or latitude
                longitude = params.get("longitude") or longitude
            
            # 确保行星名称是字符串
            if not isinstance(planet_name, str):
                planet_name = str(planet_name)
            
            # 验证行星名称
            if planet_name.lower() not in VALID_PLANETS:
                from core.errors import ErrorHandler
                error = ErrorHandler.create_tool_error(
                    "get_planet_position",
                    f"无效的行星名称。有效行星: {', '.join(VALID_PLANETS)}",
                    {"planet_name": planet_name}
                )
                return error.to_dict()
            
            # 解析观测时间
            ts = load.timescale()
            if observation_time is None:
                t = ts.now()
            else:
                dt = parse_date(observation_time)
                t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
            
            # 获取行星对象
            planet_id = PLANET_MAPPING[planet_name.lower()]
            planet = self.ephemeris.planets[planet_id]
            
            # 计算位置
            if latitude is not None and longitude is not None:
                observer = self.ephemeris.earth + wgs84.latlon(latitude, longitude)
                astrometric = observer.at(t).observe(planet)
                ra, dec, distance = astrometric.radec()
            else:
                astrometric = self.ephemeris.earth.at(t).observe(planet)
                ra, dec, distance = astrometric.radec()
            
            return {
                'ra': ra.hours,
                'dec': dec.degrees,
                'distance_au': distance.au
            }
            
        except Exception as e:
            from core.errors import ErrorHandler
            error = ErrorHandler.handle(e, {"tool": "get_planet_position", "planet_name": planet_name})
            return error.to_dict()
    
    def coordinate_transformation(self, ra, dec, epoch='J2000', target_system='fk5'):
        """
        天体坐标转换
        
        Args:
            ra: 赤经（小时），或包含ra/dec的字典/字符串
            dec: 赤纬（度）
            epoch: 历元（默认J2000）
            target_system: 目标坐标系（'fk5'或'icrs'）
            
        Returns:
            转换后的坐标字典
        """
        try:
            # 处理多种输入格式
            if isinstance(ra, dict):
                if 'ra' in ra and 'dec' in ra:
                    dec = ra['dec']
                    ra = ra['ra']
                else:
                    raise ValueError("无效的输入格式，需要包含ra和dec键")
                    
            elif isinstance(ra, str):
                from utils.helpers import extract_key_value_pairs
                coords = extract_key_value_pairs(ra, ['ra', 'dec'])
                if 'ra' in coords and 'dec' in coords:
                    ra = coords['ra']
                    dec = coords['dec']
                else:
                    raise ValueError("无效的输入格式，需要包含ra和dec值")
            
            # 创建ICRS坐标并转换
            icrs_coord = SkyCoord(ra=ra, dec=dec, unit=('hourangle', 'deg'), 
                                  frame='icrs', equinox=epoch)
            
            if target_system.lower() == 'fk5':
                fk5_coord = icrs_coord.transform_to(FK5(equinox=epoch))
                return {'ra': fk5_coord.ra.hour, 'dec': fk5_coord.dec.degree}
            elif target_system.lower() == 'icrs':
                return {'ra': ra, 'dec': dec}
            else:
                raise ValueError("不支持的目标坐标系。支持的坐标系: 'icrs', 'fk5'")
                
        except Exception as e:
            logger.error(f"坐标转换失败: {e}")
            raise
    
    def get_rise_set_times(self, body_name, latitude, longitude, date=None):
        """
        获取天体升起和落下时间
        
        Args:
            body_name: 天体名称（'sun', 'moon', 'mercury'等）
            latitude: 观测点纬度（度）
            longitude: 观测点经度（度）
            date: 日期（可选）
            
        Returns:
            包含rise_time和set_time的字典
        """
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = parse_date(date)
        
        observer.date = date.strftime('%Y/%m/%d %H:%M:%S')
        
        # 获取天体对象
        body_name_lower = body_name.lower()
        if body_name_lower == 'sun':
            body = ephem.Sun()
        elif body_name_lower == 'moon':
            body = ephem.Moon()
        elif body_name_lower in ['mercury', 'venus', 'mars', 'jupiter', 'saturn']:
            body = getattr(ephem, body_name.capitalize())()
        else:
            raise ValueError(f"不支持的天体名称: {body_name}。支持的天体: {', '.join(SUPPORTED_BODIES)}")
        
        # 计算升起落下时间
        try:
            rising = observer.next_rising(body)
            setting = observer.next_setting(body)
            
            rise_time = ephem.Date(rising).datetime()
            set_time = ephem.Date(setting).datetime()
            
            return {'rise_time': rise_time, 'set_time': set_time}
        except ephem.AlwaysUpError:
            return {'error': '天体始终在天空中'}
        except ephem.NeverUpError:
            return {'error': '天体始终在地平线以下'}
    
    def get_current_sky_objects(self, latitude, longitude, date=None):
        """
        获取当前天空中的主要天体
        
        Args:
            latitude: 观测点纬度（度），或字符串格式的参数
            longitude: 观测点经度（度）
            date: 日期（可选）
            
        Returns:
            包含各天体信息的字典
        """
        # 处理字符串格式的坐标输入
        if isinstance(latitude, str):
            coords = parse_coordinate_string(latitude)
            if coords:
                latitude, longitude = coords
            else:
                raise ValueError("无效的输入格式，需要包含latitude和longitude值")
        
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = parse_date(date)
        
        planets = ['mercury', 'venus', 'mars', 'jupiter', 'saturn']
        sky_objects = {}
        
        # 获取每个行星的位置
        for planet in planets:
            try:
                if self.ephemeris.is_loaded:
                    position = self.get_planet_position(planet, date, latitude, longitude)
                    sky_objects[planet] = position
                else:
                    sky_objects[planet] = {'error': '行星数据未加载'}
            except Exception as e:
                sky_objects[planet] = {'error': str(e)}
        
        # 获取太阳和月亮的升起落下时间
        try:
            sun_times = self.get_rise_set_times('sun', latitude, longitude, date)
            sky_objects['sun'] = sun_times
        except Exception as e:
            sky_objects['sun'] = {'error': str(e)}
        
        try:
            moon_times = self.get_rise_set_times('moon', latitude, longitude, date)
            sky_objects['moon'] = moon_times
        except Exception as e:
            sky_objects['moon'] = {'error': str(e)}
        
        return sky_objects


__all__ = ['PlanetaryCalculator']
