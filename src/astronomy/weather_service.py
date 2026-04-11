import requests
from cachetools import TTLCache
from pybreaker import CircuitBreaker

from src.core.config import settings
from src.core.errors import ErrorHandler, ErrorCode
from src.agent.param_parser import ParamParser
from src.core.logger import logger
from src.astronomy.base_api_service import BaseAPIService

_WEATHER_CACHE = TTLCache(maxsize=256, ttl=1800)

_WEATHER_BREAKER = CircuitBreaker(fail_max=5, reset_timeout=60)


class WeatherService(BaseAPIService):

    _api_key_attr = "AMAP_API_KEY"
    _cache = _WEATHER_CACHE
    _breaker = _WEATHER_BREAKER

    def reverse_geocode(self, longitude: float, latitude: float) -> str:
        try:
            if not self.api_key:
                return None

            params = {
                "key": self.api_key,
                "location": f"{longitude},{latitude}",
                "output": "JSON",
            }

            resp = self._request_get(settings.AMAP_GEOCODE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if str(data.get("status")) == "1" and data.get("regeocode"):
                address = data["regeocode"].get("addressComponent", {})
                city = address.get("city") or address.get("province")
                return city if city else None

            return None

        except Exception as e:
            logger.warning(f"逆地理编码失败: {e}")
            return None

    def get_weather(self, city=None, extensions="base") -> dict:
        try:
            params = ParamParser.parse_mixed_input(city, {"city": None, "extensions": extensions})

            if params.get("city"):
                city = params["city"]
                extensions = params.get("extensions", extensions)

            if city and ParamParser.is_coordinates(city):
                parts = city.split(",")
                lon = float(parts[1].strip())
                lat = float(parts[0].strip())
                city_name = self.reverse_geocode(lon, lat)
                if city_name:
                    city = city_name

            if not self.api_key:
                error = ErrorHandler.create_tool_error(
                    "get_weather",
                    "AMAP_API_KEY 未配置，无法查询天气"
                )
                return error.to_dict()

            if not city:
                city = settings.AMAP_DEFAULT_CITY

            cache_key = f"weather:{city}:{extensions or 'base'}"
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

            req_params = {
                "key": self.api_key,
                "city": city,
                "extensions": extensions or "base",
                "output": "JSON",
            }

            resp = self._request_get(settings.AMAP_WEATHER_URL, params=req_params)
            resp.raise_for_status()
            data = resp.json()

            if str(data.get("status")) != "1":
                error = ErrorHandler.create_tool_error(
                    "get_weather",
                    data.get("info") or "高德天气查询失败",
                    {"raw": data}
                )
                return error.to_dict()

            result = self._process_weather_response(data, city, extensions)
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"天气查询失败: {e}")
            error = ErrorHandler.handle(e, {"tool": "get_weather", "city": city})
            return error.to_dict()

    def _process_weather_response(self, data: dict, city: str, extensions: str) -> dict:

        lives = data.get("lives") or []
        forecasts = data.get("forecasts") or []

        result = {
            "query_city": city,
            "extensions": extensions or "base",
            "raw": data,
        }

        if lives:
            live = lives[0]
            weather = live.get("weather")
            humidity = live.get("humidity")
            windpower = live.get("windpower")

            result["live"] = {
                "city": live.get("city"),
                "weather": weather,
                "temperature": live.get("temperature"),
                "humidity": humidity,
                "winddirection": live.get("winddirection"),
                "windpower": windpower,
                "reporttime": live.get("reporttime"),
            }

            tips = self._generate_observing_tips(weather, humidity, windpower)
            result["observing_tips"] = tips

        if forecasts:
            result["forecast"] = forecasts[0]

        return result

    def _generate_observing_tips(self, weather: str, humidity: str, windpower: str) -> list:

        tips = []

        if weather and any(k in weather for k in ["雨", "雪", "雷", "雾", "霾"]):
            tips.append(settings.OBSERVING_TIPS_TEMPLATES['bad_weather'])
        else:
            tips.append(settings.OBSERVING_TIPS_TEMPLATES['good_weather'])

        if humidity is not None:
            try:
                h = float(humidity)
                if h >= 80:
                    tips.append(settings.OBSERVING_TIPS_TEMPLATES['high_humidity'])
            except (ValueError, TypeError):
                pass

        if windpower is not None:
            try:
                wp = float(str(windpower).replace("级", "").strip())
                if wp >= 4:
                    tips.append(settings.OBSERVING_TIPS_TEMPLATES['high_wind'])
            except (ValueError, TypeError):
                pass

        return tips


__all__ = ['WeatherService']
