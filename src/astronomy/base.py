# -*- coding: utf-8 -*-
"""
基础模块 - 星历数据管理和共享
"""

import logging
from skyfield.api import load
from skyfield import almanac
from src.core.config import settings

logger = logging.getLogger(__name__)


class EphemerisManager:
    """
    星历数据管理器（单例模式）
    
    负责加载和管理星历数据文件，确保整个应用只加载一次，
    避免重复加载造成的内存浪费。
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EphemerisManager._initialized:
            return
        
        EphemerisManager._initialized = True
        
        self.planets = None
        self.earth = None
        self.timescale = None
        self.is_loaded = False
        
        self._load_ephemeris()
    
    def _load_ephemeris(self):
        """加载星历数据"""
        try:
            logger.info("正在加载星历数据...")
            self.planets = load('de421.bsp')
            self.earth = self.planets['earth']
            self.timescale = load.timescale()
            self.is_loaded = True
            logger.info("✅ 星历数据加载成功")
        except Exception as e:
            logger.warning(f"⚠️ 无法加载行星数据: {e}")
            logger.warning("部分功能可能无法使用")
            self.is_loaded = False
            self.planets = None
            self.earth = None
            self.timescale = load.timescale()
    
    def check_loaded(self):
        """检查星历数据是否已加载"""
        if not self.is_loaded:
            raise RuntimeError("星历数据未加载，无法执行计算。请确保 de421.bsp 文件存在。")
        return True


__all__ = ['EphemerisManager']
