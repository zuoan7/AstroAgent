from __future__ import annotations

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.skill_param_builder import SkillParamBuilder


def _builder() -> SkillParamBuilder:
    return SkillParamBuilder(None)


def test_weather_extracts_city_instead_of_whole_query():
    params = _builder().build("weather-lookup", "北京今晚云多吗？")

    assert params["city"] == "北京"
    assert params["extensions"] == "all"


def test_weather_defaults_to_beijing_when_city_missing():
    params = _builder().build("weather-lookup", "今晚风大会不会影响架望远镜？")

    assert params["city"] == "北京"
    assert params["extensions"] == "all"


def test_celestial_position_direction_query_builds_altaz_params():
    params = _builder().build("celestial-position-calculator", "今晚北京能在什么方向看到木星？")

    assert params["target"] == "木星"
    assert params["location"] == "北京"
    assert params["datetime"] == "今晚"
    assert params["output_format"] == "altaz"


def test_celestial_position_rise_set_query_builds_rise_set_params():
    params = _builder().build("celestial-position-calculator", "今晚木星大概几点升起？")

    assert params["target"] == "木星"
    assert params["location"] == "北京"
    assert params["datetime"] == "今晚"
    assert params["output_format"] == "rise_set"


def test_celestial_position_height_query_builds_altaz_params():
    params = _builder().build("celestial-position-calculator", "明晚广州看火星，高度会不会太低？")

    assert params["target"] == "火星"
    assert params["location"] == "广州"
    assert params["datetime"] == "明晚"
    assert params["output_format"] == "altaz"


def test_sunset_darkness_query_builds_solar_rise_set_params():
    params = _builder().build("celestial-position-calculator", "明天北京日落后多久天会比较黑？")

    assert params["target"] == "太阳"
    assert params["location"] == "北京"
    assert params["datetime"] == "明天"
    assert params["output_format"] == "rise_set"


def test_deep_sky_extracts_m31_and_ngc_catalog_targets():
    builder = _builder()

    assert builder.build("deep-sky-observing-guide", "M31 适合怎么观测？")["target"] == "M31"
    assert (
        builder.build("deep-sky-observing-guide", "NGC 7000 用双筒能看吗？")["target"]
        == "NGC7000"
    )
    assert (
        builder.build("deep-sky-observing-guide", "IC 434 适合拍吗？")["target"]
        == "IC434"
    )


def test_astrophotography_extracts_24mm_lens_and_defaults_camera():
    params = _builder().build(
        "astrophotography-calculator",
        "用 24mm 镜头拍银河，单张曝光多久比较稳？",
    )

    assert params["target"] == "银河"
    assert params["camera"] == "未指定相机"
    assert params["telescope"] == "24mm 镜头"
    assert "location" not in params


def test_astrophotography_extracts_mount_iso_and_aperture():
    params = _builder().build(
        "astrophotography-calculator",
        "固定三脚架拍星野，ISO1600，光圈 f/2.8，快门怎么设？",
    )

    assert params["target"] == "星野"
    assert params["mount"] == "固定三脚架"
    assert params["iso"] == "ISO 1600"
    assert params["aperture"] == "f/2.8"


def test_event_range_extracts_monthly_intent_for_general_public_events():
    builder = _builder()

    this_month = builder.build("celestial-events-forecast", "这个月有什么比较重要的天象？")
    friend_events = builder.build(
        "celestial-events-forecast",
        "我想带朋友看天象，有没有适合普通人看的？",
    )

    assert this_month["start_date"] is not None
    assert this_month["end_date"] is not None
    assert friend_events["start_date"] is not None
    assert friend_events["end_date"] is not None
