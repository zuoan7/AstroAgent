# -*- coding: utf-8 -*-
"""
天体数据库查询模块 - SIMBAD和NED数据库接口
"""

import re
import logging

from astroquery.simbad import Simbad
from astroquery.ned import Ned

from src.core.config import settings

logger = logging.getLogger(__name__)


class CelestialDatabaseService:
    """
    天体数据库服务
    
    提供对SIMBAD和NED天文数据库的统一访问接口。
    """
    
    def __init__(self):
        pass
    
    def get_object_info(self, object_name: str) -> dict:
        """
        查询天体基本信息（使用SIMBAD数据库）
        
        Args:
            object_name: 天体名称
            
        Returns:
            天体信息字典
        """
        try:
            # 使用标准名称映射
            query_name = settings.CELESTIAL_NAME_MAPPING.get(object_name, object_name)
            
            # 配置Simbad查询
            custom_simbad = Simbad()
            custom_simbad.add_votable_fields('ra', 'dec', 'main_id', 'otype')
            
            # 执行查询
            result = custom_simbad.query_object(query_name)
            
            if result is None:
                return {'error': f'未找到该天体: {object_name}'}
            
            if len(result) == 0:
                return {'error': f'未找到该天体: {object_name}'}
            
            # 安全提取结果
            info = {
                'name': object_name,
                'ra': str(result['ra'][0]) if 'ra' in result.colnames and len(result['ra']) > 0 else None,
                'dec': str(result['dec'][0]) if 'dec' in result.colnames and len(result['dec']) > 0 else None,
                'main_id': str(result['main_id'][0]) if 'main_id' in result.colnames and len(result['main_id']) > 0 else None,
                'otype': str(result['otype'][0]) if 'otype' in result.colnames and len(result['otype']) > 0 else None
            }
            
            return info
            
        except Exception as e:
            logger.error(f"查询天体信息失败: {e}")
            return {'error': f'查询天体信息时出错: {e}'}
    
    def get_galaxy_data(self, galaxy_name: str) -> dict:
        """
        查询星系数据（使用NED数据库）
        
        Args:
            galaxy_name: 星系名称
            
        Returns:
            星系数据字典
        """
        try:
            # 使用标准名称映射
            query_name = settings.GALAXY_NAME_MAPPING.get(galaxy_name, galaxy_name)
            
            # 执行NED查询
            result = Ned.query_object(query_name)
            
            if result is None:
                return {'error': f'未找到该星系: {galaxy_name}'}
            
            if len(result) == 0:
                return {'error': f'未找到该星系: {galaxy_name}'}
            
            # 提取星等信息
            magnitude = None
            if 'Magnitude and Filter' in result.colnames and len(result['Magnitude and Filter']) > 0:
                mag_str = str(result['Magnitude and Filter'][0])
                mag_match = re.search(r'\d+\.\d+', mag_str)
                if mag_match:
                    magnitude = float(mag_match.group())
            
            # 安全提取关键信息
            info = {
                'name': galaxy_name,
                'ra': str(result['RA'][0]) if 'RA' in result.colnames and len(result['RA']) > 0 else None,
                'dec': str(result['DEC'][0]) if 'DEC' in result.colnames and len(result['DEC']) > 0 else None,
                'redshift': float(result['Redshift'][0]) if 'Redshift' in result.colnames and len(result['Redshift']) > 0 else None,
                'magnitude': magnitude,
                'type': str(result['Type'][0]) if 'Type' in result.colnames and len(result['Type']) > 0 else None
            }
            
            return info
            
        except Exception as e:
            logger.error(f"查询星系数据失败: {e}")
            return {'error': f'查询星系数据时出错: {e}'}


__all__ = ['CelestialDatabaseService']
