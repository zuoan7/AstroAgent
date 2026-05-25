# -*- coding: utf-8 -*-
"""
行星计算模块 - 行星位置、坐标转换、升起落下时间等
"""

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from skyfield.api import wgs84, load
from skyfield import almanac as skyfield_almanac
from astropy.coordinates import ICRS, FK5, SkyCoord
import ephem

from .base import EphemerisManager
from src.core.config import settings
from src.core.errors import AgentError, ErrorCode, ErrorHandler
from src.core.logger import logger
from src.utils.param_parser import ParamParser


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
            包含赤经、赤纬、距离和地平坐标(altitude/azimuth)的字典。
            当提供latitude和longitude时，额外返回altitude和azimuth。
        """
        if not self.ephemeris.is_loaded:
            error = ErrorHandler.create_tool_error(
                "get_planet_position",
                "行星数据未加载，无法计算行星位置"
            )
            return error.to_dict()

        try:
            params = ParamParser.parse_mixed_input(planet_name, {
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

            if not isinstance(planet_name, str):
                planet_name = str(planet_name)

            if planet_name.lower() not in settings.VALID_PLANETS:
                error = ErrorHandler.create_tool_error(
                    "get_planet_position",
                    f"无效的行星名称。有效行星: {', '.join(settings.VALID_PLANETS)}",
                    {"planet_name": planet_name}
                )
                return error.to_dict()

            ts = load.timescale()
            if observation_time is None:
                t = ts.now()
            else:
                dt = ParamParser.parse_date(observation_time)
                t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

            planet_id = settings.PLANET_MAPPING[planet_name.lower()]
            planet = self.ephemeris.planets[planet_id]

            result = {}

            if latitude is not None and longitude is not None:
                observer = self.ephemeris.earth + wgs84.latlon(latitude, longitude)
                astrometric = observer.at(t).observe(planet)
                ra, dec, distance = astrometric.radec()
                apparent = astrometric.apparent()
                alt, az, _ = apparent.altaz()

                result = {
                    'ra_hours': ra.hours,
                    'ra_degrees': ra._degrees,
                    'dec': dec.degrees,
                    'distance_au': distance.au,
                    'altitude': round(alt.degrees, 2),
                    'azimuth': round(az.degrees, 2),
                }
            else:
                astrometric = self.ephemeris.earth.at(t).observe(planet)
                ra, dec, distance = astrometric.radec()

                result = {
                    'ra_hours': ra.hours,
                    'ra_degrees': ra._degrees,
                    'dec': dec.degrees,
                    'distance_au': distance.au,
                }

            return result

        except Exception as e:
            error = ErrorHandler.handle(e, {"tool": "get_planet_position", "planet_name": planet_name})
            return error.to_dict()

    def get_altaz(self, planet_name, observation_time=None, latitude=None, longitude=None):
        """
        获取行星的地平坐标(高度角/方位角)

        Args:
            planet_name: 行星名称（'mercury', 'venus', 'mars'等）
            observation_time: 观测时间（可选），默认当前时间
            latitude: 观测点纬度（度，必填）
            longitude: 观测点经度（度，必填）

        Returns:
            包含altitude(高度角)和azimuth(方位角)的字典，
            或包含error的字典

        Raises:
            ValueError: 当latitude或longitude未提供时
        """
        if not self.ephemeris.is_loaded:
            error = ErrorHandler.create_tool_error(
                "get_altaz",
                "行星数据未加载，无法计算地平坐标"
            )
            return error.to_dict()

        if latitude is None or longitude is None:
            error = ErrorHandler.create_param_error(
                "latitude/longitude",
                "计算地平坐标需要提供观测位置的纬度和经度"
            )
            return error.to_dict()

        try:
            if not isinstance(planet_name, str):
                planet_name = str(planet_name)

            if planet_name.lower() not in settings.VALID_PLANETS:
                error = ErrorHandler.create_tool_error(
                    "get_altaz",
                    f"无效的行星名称。有效行星: {', '.join(settings.VALID_PLANETS)}",
                    {"planet_name": planet_name}
                )
                return error.to_dict()

            ts = load.timescale()
            if observation_time is None:
                t = ts.now()
            else:
                dt = ParamParser.parse_date(observation_time)
                t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

            planet_id = settings.PLANET_MAPPING[planet_name.lower()]
            planet = self.ephemeris.planets[planet_id]

            observer = self.ephemeris.earth + wgs84.latlon(latitude, longitude)
            astrometric = observer.at(t).observe(planet).apparent()
            alt, az, distance = astrometric.altaz()

            return {
                'planet': planet_name,
                'altitude': round(alt.degrees, 2),
                'azimuth': round(az.degrees, 2),
                'distance_au': round(distance.au, 4),
            }

        except Exception as e:
            error = ErrorHandler.handle(e, {"tool": "get_altaz", "planet_name": planet_name})
            return error.to_dict()

    @staticmethod
    def _detect_ra_unit(ra_value):
        """
        自动检测赤经(RA)值的单位

        RA值范围：
        - 小时(hourangle): 0~24
        - 度(degrees): 0~360

        Args:
            ra_value: RA数值

        Returns:
            'hours' 或 'degrees'
        """
        if ra_value is None:
            return 'hours'

        if isinstance(ra_value, (int, float)):
            if abs(ra_value) > 24:
                return 'degrees'
            return 'hours'

        return 'hours'

    def coordinate_transformation(self, ra, dec, epoch='J2000', target_system='fk5', ra_unit='auto'):
        """
        天体坐标转换

        Args:
            ra: 赤经，支持小时或度数单位。或包含ra/dec的字典/字符串
            dec: 赤纬（度）
            epoch: 历元（默认J2000）
            target_system: 目标坐标系（'fk5'或'icrs'）
            ra_unit: 赤经单位，'auto'自动检测, 'hours'小时, 'degrees'度数

        Returns:
            转换后的坐标字典，包含ra_hours, ra_degrees, dec
        """
        try:
            if isinstance(ra, dict):
                if 'ra_hours' in ra:
                    ra = ra['ra_hours']
                    ra_unit = 'hours'
                elif 'ra_degrees' in ra:
                    ra = ra['ra_degrees']
                    ra_unit = 'degrees'
                elif 'ra' in ra and 'dec' in ra:
                    dec = ra['dec']
                    ra = ra['ra']
                else:
                    raise ValueError("无效的输入格式，需要包含ra和dec键")

            elif isinstance(ra, str):
                from src.utils.param_parser import ParamParser
                coords = ParamParser.extract_key_value_pairs(ra, ['ra', 'dec'])
                if 'ra' in coords and 'dec' in coords:
                    ra = coords['ra']
                    dec = coords['dec']
                else:
                    raise ValueError("无效的输入格式，需要包含ra和dec值")

            if ra_unit == 'auto':
                ra_unit = self._detect_ra_unit(ra)

            if ra_unit == 'degrees':
                ra_hours = ra / 15.0
            else:
                ra_hours = ra

            icrs_coord = SkyCoord(ra=ra_hours, dec=dec, unit=('hourangle', 'deg'),
                                  frame='icrs', equinox=epoch)

            if target_system.lower() == 'fk5':
                fk5_coord = icrs_coord.transform_to(FK5(equinox=epoch))
                return {
                    'ra_hours': fk5_coord.ra.hour,
                    'ra_degrees': fk5_coord.ra.degree,
                    'dec': fk5_coord.dec.degree
                }
            elif target_system.lower() == 'icrs':
                ra_deg = ra_hours * 15.0
                return {
                    'ra_hours': ra_hours,
                    'ra_degrees': ra_deg,
                    'dec': dec
                }
            else:
                raise ValueError("不支持的目标坐标系。支持的坐标系: 'icrs', 'fk5'")

        except Exception as e:
            logger.error(f"坐标转换失败: {e}")
            raise

    def get_rise_set_times(self, body_name, latitude, longitude, date=None, tz_name='Asia/Shanghai'):
        """
        获取天体升起和落下时间

        使用skyfield库进行计算，确保UTC时间统一传入，
        结果输出时转换为用户指定的本地时区。

        Args:
            body_name: 天体名称（'sun', 'moon', 'mercury'等）
            latitude: 观测点纬度（度）
            longitude: 观测点经度（度）
            date: 日期（可选），支持datetime对象或字符串
            tz_name: 时区名称（默认'Asia/Shanghai'，即UTC+8）

        Returns:
            包含rise_time和set_time的字典，时间为本地时区datetime
        """
        if date is None:
            date = datetime.now(timezone.utc)
        elif isinstance(date, str):
            date = ParamParser.parse_date(date)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
        elif isinstance(date, datetime):
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)

        try:
            local_tz = ZoneInfo(tz_name)
        except Exception:
            local_tz = ZoneInfo('Asia/Shanghai')

        local_date = date.astimezone(local_tz)
        utc_date = date.astimezone(timezone.utc)

        ts = self.ephemeris.timescale if self.ephemeris.is_loaded else load.timescale()
        observer = wgs84.latlon(latitude, longitude)

        t0 = ts.utc(utc_date.year, utc_date.month, utc_date.day)
        t1 = ts.utc(utc_date.year, utc_date.month, utc_date.day, 23, 59, 59)

        body_name_lower = body_name.lower()

        if body_name_lower == 'sun':
            f = skyfield_almanac.sunrise_sunset(self.ephemeris.planets, observer)
            times, events = skyfield_almanac.find_discrete(t0, t1, f)

            rise_time = None
            set_time = None

            for t, e in zip(times, events):
                utc_dt = t.utc_datetime()
                local_dt = utc_dt.astimezone(local_tz)
                if e == 1:
                    rise_time = local_dt
                elif e == 0:
                    set_time = local_dt

            return {'rise_time': rise_time, 'set_time': set_time}

        elif body_name_lower == 'moon':
            f = skyfield_almanac.risings_and_settings(
                self.ephemeris.planets, self.ephemeris.planets['moon'], observer
            )
            times, events = skyfield_almanac.find_discrete(t0, t1, f)

            rise_time = None
            set_time = None

            for t, e in zip(times, events):
                utc_dt = t.utc_datetime()
                local_dt = utc_dt.astimezone(local_tz)
                if e == 1:
                    rise_time = local_dt
                elif e == 0:
                    set_time = local_dt

            return {'rise_time': rise_time, 'set_time': set_time}

        elif body_name_lower in ['mercury', 'venus', 'mars', 'jupiter', 'saturn']:
            planet_id = settings.PLANET_MAPPING.get(body_name_lower)
            if planet_id is None:
                raise ValueError(f"不支持的行星名称: {body_name}")

            planet = self.ephemeris.planets[planet_id]
            f = skyfield_almanac.risings_and_settings(
                self.ephemeris.planets, planet, observer
            )
            times, events = skyfield_almanac.find_discrete(t0, t1, f)

            rise_time = None
            set_time = None

            for t, e in zip(times, events):
                utc_dt = t.utc_datetime()
                local_dt = utc_dt.astimezone(local_tz)
                if e == 1:
                    rise_time = local_dt
                elif e == 0:
                    set_time = local_dt

            return {'rise_time': rise_time, 'set_time': set_time}

        else:
            raise ValueError(f"不支持的天体名称: {body_name}。支持的天体: {', '.join(settings.SUPPORTED_BODIES)}")

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
        if isinstance(latitude, str):
            coords = ParamParser.parse_coordinate_string(latitude)
            if coords:
                latitude, longitude = coords
            else:
                raise ValueError("无效的输入格式，需要包含latitude和longitude值")

        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = ParamParser.parse_date(date)

        planets = ['mercury', 'venus', 'mars', 'jupiter', 'saturn']
        sky_objects = {}

        for planet in planets:
            try:
                if self.ephemeris.is_loaded:
                    position = self.get_planet_position(planet, date, latitude, longitude)
                    sky_objects[planet] = position
                else:
                    sky_objects[planet] = AgentError(
                        code=ErrorCode.TOOL_CALL_FAILED,
                        message='行星数据未加载',
                        details={"planet": planet}
                    ).to_dict()
            except Exception as e:
                sky_objects[planet] = ErrorHandler.handle(
                    e, {"planet": planet}
                ).to_dict()

        try:
            sun_times = self.get_rise_set_times('sun', latitude, longitude, date)
            sky_objects['sun'] = sun_times
        except Exception as e:
            sky_objects['sun'] = ErrorHandler.handle(
                e, {"body": "sun"}
            ).to_dict()

        try:
            moon_times = self.get_rise_set_times('moon', latitude, longitude, date)
            sky_objects['moon'] = moon_times
        except Exception as e:
            sky_objects['moon'] = ErrorHandler.handle(
                e, {"body": "moon"}
            ).to_dict()

        return sky_objects


__all__ = ['PlanetaryCalculator']
