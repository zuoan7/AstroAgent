import math
import os
from datetime import datetime, timedelta
from skyfield.api import load, wgs84
from skyfield import almanac
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
        :param planet_name: 行星名称，如 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'，或者是字符串格式的参数
        :param observation_time: 观测时间，默认为当前时间
        :param latitude: 观测点纬度（度）
        :param longitude: 观测点经度（度）
        :return: 行星位置（赤经、赤纬）
        """
        # 检查数据是否加载
        if not self.data_loaded:
            raise Exception("行星数据未加载，无法计算行星位置")
        
        # 处理字典格式的输入（来自LangChain agent）
        if isinstance(planet_name, dict):
            # 从字典中提取行星名称
            planet_name = planet_name.get('planet_name', planet_name)
        # 处理字符串格式的输入（来自LangChain agent，如 "planet_name='mars'"）
        elif isinstance(planet_name, str):
            import re
            # 尝试从字符串中提取行星名称
            match = re.search(r"planet_name=['\"]([^'\"]+)['\"]", planet_name)
            if match:
                planet_name = match.group(1)
            # 也处理没有引号的情况
            elif '=' in planet_name:
                parts = planet_name.split('=')
                if len(parts) == 2:
                    planet_name = parts[1].strip().strip("'\"").strip()
        
        # 确保行星名称是字符串
        if not isinstance(planet_name, str):
            planet_name = str(planet_name)
        
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
        :param ra: 赤经（小时），或者包含ra和dec键的字典，或者是字符串格式的参数
        :param dec: 赤纬（度），当ra为字典或字符串时，此参数会被忽略
        :param epoch: 历元，默认为J2000
        :param target_system: 目标坐标系，默认为fk5
        :return: 转换后的坐标
        """
        # 处理字典格式的输入（来自agent）
        if isinstance(ra, dict):
            if 'ra' in ra and 'dec' in ra:
                dec = ra['dec']
                ra = ra['ra']
            else:
                raise ValueError("无效的输入格式，需要包含ra和dec键")
        # 处理字符串格式的输入（如 "ra=10.5, dec=20.5"）
        elif isinstance(ra, str):
            import re
            # 尝试从字符串中提取ra和dec值
            ra_match = re.search(r"ra=([\d.]+)", ra)
            dec_match = re.search(r"dec=([-\d.]+)", ra)
            if ra_match and dec_match:
                ra = float(ra_match.group(1))
                dec = float(dec_match.group(1))
            else:
                raise ValueError("无效的输入格式，需要包含ra和dec值")
        
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
        :param latitude: 观测点纬度（度），或者是字符串格式的参数
        :param longitude: 观测点经度（度），当latitude为字符串时，此参数会被忽略
        :param date: 日期，默认为当前日期
        :return: 当前天空中的主要天体信息
        """
        # 处理字符串格式的输入（来自LangChain agent，如 "latitude=39.9042, longitude=116.4074"）
        if isinstance(latitude, str):
            import re
            # 尝试从字符串中提取latitude和longitude值
            lat_match = re.search(r"latitude=([-\d.]+)", latitude)
            lon_match = re.search(r"longitude=([-\d.]+)", latitude)
            if lat_match and lon_match:
                latitude = float(lat_match.group(1))
                longitude = float(lon_match.group(1))
            else:
                raise ValueError("无效的输入格式，需要包含latitude和longitude值")
        
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
            # 常用天体的标准名称映射
            name_mapping = {
                'Andromeda Galaxy': 'M31',
                'Sirius': 'Sirius',
                'Orion Nebula': 'M42'
            }
            
            # 使用标准名称
            query_name = name_mapping.get(object_name, object_name)
            
            # 自定义Simbad查询，添加需要的字段
            custom_simbad = Simbad()
            custom_simbad.add_votable_fields('ra', 'dec', 'main_id', 'otype')
            
            # 使用SIMBAD查询天体信息
            result = custom_simbad.query_object(query_name)
            if result is None:
                return {'error': f'未找到该天体: {object_name}'}
            
            # 打印所有可用的列名，以便调试
            # print(f"SIMBAD返回的列名: {result.colnames}")
            
            # 提取关键信息
            info = {
                'name': object_name,
                'ra': str(result['ra'][0]) if 'ra' in result.colnames else None,
                'dec': str(result['dec'][0]) if 'dec' in result.colnames else None,
                'main_id': str(result['main_id'][0]) if 'main_id' in result.colnames else None,
                'otype': str(result['otype'][0]) if 'otype' in result.colnames else None
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
            # 常用星系的标准名称映射
            name_mapping = {
                'Milky Way': 'Milky Way Galaxy',
                'Andromeda Galaxy': 'Andromeda Galaxy',
                'Triangulum Galaxy': 'Triangulum Galaxy'
            }
            
            # 使用标准名称
            query_name = name_mapping.get(galaxy_name, galaxy_name)
            
            # 使用NED查询星系数据
            result = Ned.query_object(query_name)
            if result is None:
                return {'error': f'未找到该星系: {galaxy_name}'}
            
            # 打印所有可用的列名，以便调试
            print(f"NED返回的列名: {result.colnames}")
            
            # 提取星等信息
            magnitude = None
            if 'Magnitude and Filter' in result.colnames:
                mag_str = str(result['Magnitude and Filter'][0])
                # 尝试从字符串中提取星等值
                import re
                mag_match = re.search(r'\d+\.\d+', mag_str)
                if mag_match:
                    magnitude = float(mag_match.group())
            
            # 提取关键信息
            info = {
                'name': galaxy_name,
                'ra': str(result['RA'][0]) if 'RA' in result.colnames else None,
                'dec': str(result['DEC'][0]) if 'DEC' in result.colnames else None,
                'redshift': float(result['Redshift'][0]) if 'Redshift' in result.colnames else None,
                'magnitude': magnitude,
                'type': str(result['Type'][0]) if 'Type' in result.colnames else None
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
            api_key = os.getenv('NASA_API_KEY')
            
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
            api_key = os.getenv('NASA_API_KEY')
            
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


class AstronomyEventsPredictor:
    """天象预测工具类"""
    
    def __init__(self, location=None):
        """
        初始化预测器
        :param location: 观测地坐标 (纬度, 经度)，默认北京
        """
        # 加载星历文件（你已下载的de421.bsp）
        self.planets = load('de421.bsp')
        self.ts = load.timescale()
        
        # 设置默认观测地（北京）
        if location:
            self.lat, self.lon = location
        else:
            self.lat, self.lon = 39.9, 116.4  # 北京
        
        # 正确创建观测点
        self.earth = self.planets['earth']
        self.observer_location = wgs84.latlon(self.lat, self.lon)
        
        # 预置2026年主要天象数据
        self.special_events_2026 = self._load_special_events()
        
        # print(f"✅ 天象预测器初始化完成，观测地：北纬{self.lat}° 东经{self.lon}°")
    
    def _load_special_events(self):
        """
        加载2026年特殊天象数据
        数据来源：天文通、有趣天文奇观等网站
        """
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
    
    def get_moon_phase(self, date):
        """
        计算指定日期的月相
        :param date: datetime对象
        :return: 月相名称和描述
        """
        # 使用skyfield计算月相
        t = self.ts.utc(date.year, date.month, date.day)
        
        # 获取月亮和太阳的黄经差
        e = self.planets['earth']
        m = self.planets['moon']
        s = self.planets['sun']
        
        # 计算月亮和太阳的相对位置
        moon_earth = e.at(t).observe(m).apparent()
        sun_earth = e.at(t).observe(s).apparent()
        
        # 计算黄经差（简化版）
        moon_ra, moon_dec, _ = moon_earth.radec()
        sun_ra, sun_dec, _ = sun_earth.radec()
        
        # 粗略计算相位（实际应该用黄经，这里简化用赤经差）
        ra_diff = (moon_ra.hours - sun_ra.hours) * 15  # 转换为度
        if ra_diff < 0:
            ra_diff += 360
        
        phase_angle = ra_diff
        
        # 根据角度判断月相
        if phase_angle < 22.5 or phase_angle >= 337.5:
            return "🌑 新月", "月光微弱，适合观测深空天体"
        elif phase_angle < 67.5:
            return "🌒 娥眉月", "傍晚可见，适合观测"
        elif phase_angle < 112.5:
            return "🌓 上弦月", "下午至前半夜可见"
        elif phase_angle < 157.5:
            return "🌔 盈凸月", "傍晚至后半夜可见"
        elif phase_angle < 202.5:
            return "🌕 满月", "整夜可见，月光强，不适合深空观测"
        elif phase_angle < 247.5:
            return "🌖 亏凸月", "前半夜可见"
        elif phase_angle < 292.5:
            return "🌗 下弦月", "后半夜可见"
        else:
            return "🌘 残月", "凌晨可见"
    
    def get_sunrise_sunset(self, date):
        """
        计算日出日落时间
        """
        t0 = self.ts.utc(date.year, date.month, date.day, 0)
        t1 = self.ts.utc(date.year, date.month, date.day, 23, 59)
        
        # 查找日出和日落
        f = almanac.sunrise_sunset(self.planets, self.observer_location)
        times, events = almanac.find_discrete(t0, t1, f)
        
        sunrise = None
        sunset = None
        
        for t, e in zip(times, events):
            if e == 1:  # 日出
                sunrise = t.utc_datetime()
            elif e == 0:  # 日落
                sunset = t.utc_datetime()
        
        return sunrise, sunset
    
    def get_visible_planets(self, date):
        """
        计算指定日期可见的行星
        """
        t = self.ts.utc(date.year, date.month, date.day, 20)  # 晚上8点
        
        # 行星名称映射到Skyfield的行星ID
        planet_mapping = {
            'mercury': 199,  # MERCURY
            'venus': 299,    # VENUS
            'mars': 499,     # MARS
            'jupiter': 5,    # JUPITER BARYCENTER
            'saturn': 6,     # SATURN BARYCENTER
        }
        
        planets_info = [
            {"name": "水星", "obj": self.planets[planet_mapping['mercury']], "max_mag": -1.9},
            {"name": "金星", "obj": self.planets[planet_mapping['venus']], "max_mag": -4.6},
            {"name": "火星", "obj": self.planets[planet_mapping['mars']], "max_mag": -2.0},
            {"name": "木星", "obj": self.planets[planet_mapping['jupiter']], "max_mag": -2.7},
            {"name": "土星", "obj": self.planets[planet_mapping['saturn']], "max_mag": -0.3},
        ]
        
        visible = []
        earth = self.planets['earth']
        
        for p in planets_info:
            # 计算行星位置（从观测点观测）
            observer = self.earth + self.observer_location
            astrometric = observer.at(t).observe(p['obj']).apparent()
            alt, az, distance = astrometric.altaz()
            
            # 如果高度 > 15度，认为可见
            if alt.degrees > 15:
                direction = self._get_direction(az.degrees)
                visible.append(f"{p['name']}（{direction}天空）")
        
        return visible
    
    def _get_direction(self, az_degrees):
        """将方位角转为中文方向"""
        if az_degrees < 45 or az_degrees >= 315:
            return "北"
        elif az_degrees < 135:
            return "东"
        elif az_degrees < 225:
            return "南"
        else:
            return "西"
    
    def get_weekly_events(self, start_date=None):
        """
        获取未来一周的天象
        :param start_date: 起始日期，默认今天
        :return: 格式化的天象信息
        """
        if start_date is None:
            start_date = datetime.now().replace(tzinfo=None)
        else:
            # 确保是naive datetime
            if hasattr(start_date, 'tzinfo') and start_date.tzinfo:
                start_date = start_date.replace(tzinfo=None)
        
        end_date = start_date + timedelta(days=7)
        
        # 收集天象
        daily_events = []
        special_events = []
        
        # 遍历每一天
        current = start_date
        while current <= end_date:
            day_str = current.strftime("%Y-%m-%d")
            
            # 计算月相
            moon_phase, moon_desc = self.get_moon_phase(current)
            
            # 计算可见行星
            planets = self.get_visible_planets(current)
            
            # 计算日出日落
            sunrise, sunset = self.get_sunrise_sunset(current)
            
            daily = {
                "date": day_str,
                "weekday": self._get_weekday(current.weekday()),
                "moon_phase": moon_phase,
                "moon_desc": moon_desc,
                "planets": planets,
                "sunrise": sunrise.strftime("%H:%M") if sunrise else "未知",
                "sunset": sunset.strftime("%H:%M") if sunset else "未知",
            }
            daily_events.append(daily)
            
            # 检查特殊天象
            for event in self.special_events_2026:
                if event["date"] == day_str:
                    special_events.append(event)
            
            current += timedelta(days=1)
        
        return self._format_weekly_response(daily_events, special_events)
    
    def get_monthly_events(self, year=None, month=None):
        """
        获取未来一个月的天象
        :param year: 年份，默认当前年
        :param month: 月份，默认下个月
        """
        today = datetime.now()
        
        if year is None:
            year = today.year
        if month is None:
            month = today.month + 1
            if month > 12:
                month = 1
                year += 1
        
        # 获取该月的所有特殊天象
        month_events = []
        for event in self.special_events_2026:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d")
            if event_date.year == year and event_date.month == month:
                month_events.append(event)
        
        # 按日期排序
        month_events.sort(key=lambda x: x["date"])
        
        return self._format_monthly_response(year, month, month_events)
    
    def _get_weekday(self, w):
        """数字星期转中文"""
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        return weekdays[w]
    
    def _format_weekly_response(self, daily_events, special_events):
        """格式化一周天象响应"""
        response = "🔭 **未来一周天象预报**\n\n"
        
        # 特殊天象置顶
        if special_events:
            response += "✨ **特别推荐**\n"
            for event in special_events:
                response += f"• {event['date']} {event['event']}：{event['description']}\n"
            response += "\n"
        
        # 每日看点
        response += "📅 **每日看点**\n"
        for day in daily_events:
            response += f"\n**{day['date']} {day['weekday']}**\n"
            response += f"• 月相：{day['moon_phase']}（{day['moon_desc']}）\n"
            response += f"• 可见行星：{', '.join(day['planets']) if day['planets'] else '无明显可见行星'}\n"
            response += f"• 日出：{day['sunrise']}  日落：{day['sunset']}\n"
        
        # 观测建议
        response += "\n💡 **观测建议**\n"
        response += "• 新月前后最适合观测深空天体\n"
        response += "• 行星最佳观测时间是合日和冲日前后\n"
        response += "• 城市光污染下，明亮的行星和月亮仍是好目标\n"
        
        return response
    
    def _format_monthly_response(self, year, month, events):
        """格式化一个月天象响应"""
        month_names = ["1月", "2月", "3月", "4月", "5月", "6月", 
                      "7月", "8月", "9月", "10月", "11月", "12月"]
        
        # 确保year和month是整数
        if isinstance(year, dict):
            year = year.get('year', datetime.now().year)
        if isinstance(month, dict):
            month = month.get('month', datetime.now().month + 1)
        
        response = f"🔭 **{year}年{month_names[month-1]}天象预报**\n\n"
        
        if not events:
            response += f"{year}年{month}月没有特殊天象。\n"
            response += "但你可以关注日常可见的行星和月相变化。\n"
        else:
            response += "✨ **本月特殊天象**\n"
            for event in events:
                response += f"• **{event['date']}** {event['event']}\n"
                response += f"  {event['description']}\n"
        
        # 增加行星可见性概览（根据月份动态调整）
        response += "\n🪐 **本月行星可见性**\n"
        
        # 不同月份的行星可见性信息
        planet_visibility = {
            1: [
                "• 木星：前半夜可见，亮度约-2.0等",
                "• 土星：后半夜可见",
                "• 金星：早晨东方低空可见"
            ],
            2: [
                "• 木星：前半夜可见，逐渐西沉",
                "• 土星：后半夜可见",
                "• 金星：早晨东方低空可见"
            ],
            3: [
                "• 木星：傍晚西方低空可见",
                "• 土星：后半夜可见",
                "• 金星：早晨东方低空可见"
            ],
            4: [
                "• 土星：整夜可见，是本月最佳观测目标",
                "• 金星：早晨东方低空可见",
                "• 木星：傍晚西方低空可见（月末接近合日）"
            ],
            5: [
                "• 土星：整夜可见，是本月最佳观测目标",
                "• 金星：早晨东方低空可见",
                "• 火星：后半夜可见"
            ],
            6: [
                "• 土星：整夜可见，是本月最佳观测目标",
                "• 金星：早晨东方低空可见",
                "• 火星：后半夜可见"
            ],
            7: [
                "• 土星：整夜可见，是本月最佳观测目标",
                "• 火星：前半夜可见",
                "• 木星：早晨东方低空可见"
            ],
            8: [
                "• 土星：整夜可见，是本月最佳观测目标",
                "• 火星：前半夜可见",
                "• 木星：早晨东方低空可见"
            ],
            9: [
                "• 土星：整夜可见，是本月最佳观测目标",
                "• 火星：前半夜可见",
                "• 木星：早晨东方低空可见"
            ],
            10: [
                "• 木星：整夜可见，是本月最佳观测目标",
                "• 土星：前半夜可见",
                "• 火星：前半夜可见"
            ],
            11: [
                "• 木星：整夜可见，是本月最佳观测目标",
                "• 土星：傍晚西方低空可见",
                "• 火星：前半夜可见"
            ],
            12: [
                "• 木星：整夜可见，是本月最佳观测目标",
                "• 金星：傍晚西方低空可见",
                "• 火星：前半夜可见"
            ]
        }
        
        # 添加当前月份的行星可见性信息
        for line in planet_visibility.get(month, ["• 无明显可见行星"]):
            response += line + "\n"
        
        return response
    
    def get_tonight_best(self):
        """获取今晚最佳观测目标"""
        today = datetime.now()
        day_str = today.strftime("%Y-%m-%d")
        
        # 检查今晚有无特殊天象
        special_tonight = None
        for event in self.special_events_2026:
            if event["date"] == day_str:
                special_tonight = event
                break
        
        # 获取月相
        moon_phase, moon_desc = self.get_moon_phase(today)
        
        # 获取可见行星
        planets = self.get_visible_planets(today)
        
        # 获取日出日落
        sunrise, sunset = self.get_sunrise_sunset(today)
        
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
            response += f"\n💡 **今晚特别适合深空观测！** 无月光干扰，可以挑战星系、星云。"
        elif "满月" in moon_phase:
            response += f"\n💡 **满月光太强**，建议观测明亮的行星和双星。"
        
        return response