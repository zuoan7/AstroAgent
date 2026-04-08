"""
兼容性模块 - 为了平滑过渡到新的目录结构
这个文件提供从旧导入路径到新导入路径的映射
建议新代码直接使用新的导入路径
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 重定向旧导入到新位置
import src.agent as agent
import src.core.config as config
import src.core.logger as logger
import src.skills.router as skills
import src.core.errors as core_errors

# 导出常用模块
__all__ = ['agent', 'config', 'logger', 'skills', 'core_errors']

print("⚠️  注意：兼容性模块已启用，建议更新代码使用新的导入路径")
print("旧导入: from agent import AstroAgent")
print("新导入: from src.agent import AstroAgent")