# -*- coding: utf-8 -*-
"""
工具函数模块 - 向后兼容层
所有功能已迁移至 src.utils.param_parser.ParamParser
本模块保留函数签名以保持向后兼容性
"""

from src.utils.param_parser import ParamParser


def parse_mixed_input(value, expected_params=None):
    return ParamParser.parse_mixed_input(value, expected_params)


def parse_json_string(text):
    return ParamParser.parse_json_string(text)


def parse_date(date_str, default=None):
    return ParamParser.parse_date(date_str, default)


def parse_coordinate_string(text):
    return ParamParser.parse_coordinate_string(text)


def is_coordinates(text):
    return ParamParser.is_coordinates(text)


def extract_key_value_pairs(text, keys):
    return ParamParser.extract_key_value_pairs(text, keys)


def safe_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    return default


def safe_int(value, default=None):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
    return default


def safe_bool(value, default=None):
    return ParamParser.safe_bool(value, default=default)


def shorten_text(text, max_len=1200):
    return ParamParser.shorten_text(text, max_len)


def normalize_location(location):
    return ParamParser.normalize_location(location)


def get_direction_from_azimuth(az_degrees):
    return ParamParser.get_direction_from_azimuth(az_degrees)


def extract_image_url(text):
    return ParamParser.extract_image_url(text)
