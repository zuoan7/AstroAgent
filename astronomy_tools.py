import math
from datetime import datetime
from skyfield.api import load, wgs84
from astropy.coordinates import ICRS, FK5, SkyCoord, EarthLocation
from astropy.time import Time
import ephem
from astroquery.simbad import Simbad
from astroquery.ned import Ned
import requests

class AstronomyTools:
    def __init__(self):
        # 加载Skyfield的行星数据
        try:
            self.planets = load('de421.bsp')
            self.earth = self.planets['earth']
            self.data_loaded = True
        except Exception as e:
            print(f"警告: 无法加载行星数据: {e}")
            print("部分功能可能无法使用")
            self.data_loaded = False
            self.planets = None
            self.earth = None
    
    def get_planet_position(self, planet_name, observation_time=None, latitude=None, longitude=None):
        """
        获取行星位置
        :param planet_name: 行星名称，如 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'
        :param observation_time: 观测时间，默认为当前时间
        :param latitude: 观测点纬度（度）
        :param longitude: 观测点经度（度）
        :return: 行星位置（赤经、赤纬）
        """
        # 检查数据是否加载
        if not self.data_loaded:
            raise Exception("行星数据未加载，无法计算行星位置")
        
        # 检查行星名称是否有效
        valid_planets = {'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'}
        if planet_name.lower() not in valid_planets:
            raise ValueError(f"无效的行星名称。有效行星: {', '.join(valid_planets)}")
        
        # 行星名称映射到de421.bsp中的正确名称或ID
        planet_mapping = {
            'mercury': 199,  # MERCURY
            'venus': 299,    # VENUS
            'mars': 499,     # MARS
            'jupiter': 5,    # JUPITER BARYCENTER
            'saturn': 6,     # SATURN BARYCENTER
            'uranus': 7,     # URANUS BARYCENTER
            'neptune': 8     # NEPTUNE BARYCENTER
        }
        
        # 使用当前时间或指定时间
        try:
            ts = load.timescale()
            if observation_time is None:
                t = ts.now()
            else:
                t = ts.utc(observation_time.year, observation_time.month, observation_time.day,
                          observation_time.hour, observation_time.minute, observation_time.second)
            
            # 计算行星位置
            planet_id = planet_mapping[planet_name.lower()]
            planet = self.planets[planet_id]
            if latitude is not None and longitude is not None:
                # 在指定地点观测
                observer = self.earth + wgs84.latlon(latitude, longitude)
                astrometric = observer.at(t).observe(planet)
                ra, dec, distance = astrometric.radec()
            else:
                # 地心观测
                astrometric = self.earth.at(t).observe(planet)
                ra, dec, distance = astrometric.radec()
            
            return {
                'ra': ra.hours,
                'dec': dec.degrees,
                'distance_au': distance.au
            }
        except Exception as e:
            raise Exception(f"计算行星位置时出错: {e}")
    
    def coordinate_transformation(self, ra, dec, epoch='J2000', target_system='fk5'):
        """
        天体坐标转换
        :param ra: 赤经（小时）
        :param dec: 赤纬（度）
        :param epoch: 历元，默认为J2000
        :param target_system: 目标坐标系，默认为fk5
        :return: 转换后的坐标
        """
        # 创建ICRS坐标
        icrs_coord = SkyCoord(ra=ra, dec=dec, unit=('hourangle', 'deg'), frame='icrs', equinox=epoch)
        
        # 转换到目标坐标系
        if target_system.lower() == 'fk5':
            fk5_coord = icrs_coord.transform_to(FK5(equinox=epoch))
            return {
                'ra': fk5_coord.ra.hour,
                'dec': fk5_coord.dec.degree
            }
        elif target_system.lower() == 'icrs':
            return {
                'ra': ra,
                'dec': dec
            }
        else:
            raise ValueError("不支持的目标坐标系。支持的坐标系: 'icrs', 'fk5'")
    
    def get_rise_set_times(self, body_name, latitude, longitude, date=None):
        """
        获取天体升起和落下时间
        :param body_name: 天体名称，如 'sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn'
        :param latitude: 观测点纬度（度）
        :param longitude: 观测点经度（度）
        :param date: 日期，默认为当前日期
        :return: 升起和落下时间
        """
        # 创建观测者
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        
        # 设置日期
        if date is None:
            date = datetime.now()
        observer.date = date.strftime('%Y/%m/%d %H:%M:%S')
        
        # 获取天体
        if body_name.lower() == 'sun':
            body = ephem.Sun()
        elif body_name.lower() == 'moon':
            body = ephem.Moon()
        elif body_name.lower() in ['mercury', 'venus', 'mars', 'jupiter', 'saturn']:
            body = getattr(ephem, body_name.capitalize())()
        else:
            raise ValueError("不支持的天体名称。支持的天体: 'sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn'")
        
        # 计算升起和落下时间
        try:
            rising = observer.next_rising(body)
            setting = observer.next_setting(body)
            
            # 转换为datetime对象
            rise_time = ephem.Date(rising).datetime()
            set_time = ephem.Date(setting).datetime()
            
            return {
                'rise_time': rise_time,
                'set_time': set_time
            }
        except ephem.AlwaysUpError:
            return {'error': '天体始终在天空中'}
        except ephem.NeverUpError:
            return {'error': '天体始终在地平线以下'}
    
    def get_current_sky_objects(self, latitude, longitude, date=None):
        """
        获取当前天空中的主要天体
        :param latitude: 观测点纬度（度）
        :param longitude: 观测点经度（度）
        :param date: 日期，默认为当前日期
        :return: 当前天空中的主要天体信息
        """
        if date is None:
            date = datetime.now()
        
        # 行星列表
        planets = ['mercury', 'venus', 'mars', 'jupiter', 'saturn']
        sky_objects = {}
        
        # 获取每个行星的位置
        for planet in planets:
            try:
                if self.data_loaded:
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
    
    def get_astrophysical_object_info(self, object_name):
        """
        查询天体基本信息
        :param object_name: 天体名称
        :return: 天体基本信息
        """
        try:
            # 使用SIMBAD查询天体信息
            result = Simbad.query_object(object_name)
            if result is None:
                return {'error': '未找到该天体'}
            
            # 提取关键信息
            info = {
                'name': object_name,
                'ra': result['RA'][0] if 'RA' in result.colnames else None,
                'dec': result['DEC'][0] if 'DEC' in result.colnames else None,
                'main_id': result['MAIN_ID'][0] if 'MAIN_ID' in result.colnames else None,
                'otype': result['OTYPE'][0] if 'OTYPE' in result.colnames else None
            }
            
            return info
        except Exception as e:
            return {'error': f'查询天体信息时出错: {e}'}
    
    def get_galaxy_data(self, galaxy_name):
        """
        星系数据查询
        :param galaxy_name: 星系名称
        :return: 星系数据
        """
        try:
            # 使用NED查询星系数据
            result = Ned.query_object(galaxy_name)
            if result is None:
                return {'error': '未找到该星系'}
            
            # 提取关键信息
            info = {
                'name': galaxy_name,
                'ra': result['RA(deg)'][0] if 'RA(deg)' in result.colnames else None,
                'dec': result['DEC(deg)'][0] if 'DEC(deg)' in result.colnames else None,
                'redshift': result['Redshift'][0] if 'Redshift' in result.colnames else None,
                'magnitude': result['Magnitude'][0] if 'Magnitude' in result.colnames else None,
                'type': result['Type'][0] if 'Type' in result.colnames else None
            }
            
            return info
        except Exception as e:
            return {'error': f'查询星系数据时出错: {e}'}
    
    def get_nasa_apod(self, date=None, hd=False):
        """
        获取NASA每日天文图
        :param date: 日期，格式为YYYY-MM-DD，默认为当前日期
        :param hd: 是否获取高清图像
        :return: APOD信息
        """
        try:
            # NASA APOD API URL
            url = "https://api.nasa.gov/planetary/apod"
            
            # API密钥
            api_key = "WAKNyJSTmnhaML2WFuIHLVvKK9HkWp6dGoj2gqCk"
            
            # 参数
            params = {
                "api_key": api_key,
                "hd": str(hd).lower()
            }
            
            # 如果指定了日期
            if date:
                params["date"] = date
            
            # 发送请求
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # 解析响应
            data = response.json()
            
            return data
        except Exception as e:
            return {'error': f'获取NASA每日天文图时出错: {e}'}
    
    def get_neo_data(self, start_date=None, end_date=None, limit=20):
        """
        获取近地天体数据
        :param start_date: 开始日期，格式为YYYY-MM-DD，默认为今天
        :param end_date: 结束日期，格式为YYYY-MM-DD，默认为开始日期+7天
        :param limit: 返回结果数量限制
        :return: 近地天体数据
        """
        try:
            # NASA NEO API URL
            url = "https://api.nasa.gov/neo/rest/v1/feed"
            
            # API密钥
            api_key = "WAKNyJSTmnhaML2WFuIHLVvKK9HkWp6dGoj2gqCk"
            
            # 参数
            params = {
                "api_key": api_key,
                "limit": limit
            }
            
            # 如果指定了日期
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            
            # 发送请求
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # 解析响应
            data = response.json()
            
            return data
        except Exception as e:
            return {'error': f'获取近地天体数据时出错: {e}'}