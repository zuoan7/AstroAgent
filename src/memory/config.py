"""
记忆层配置模块

该模块提供统一的配置管理接口，导出核心配置对象 settings，
用于管理记忆层的各项参数和特性开关。

主要功能：
- 导入并导出核心配置 settings
- 提供记忆层统一的配置访问点
- 支持会话记忆和长期记忆的配置统一管理

使用示例：
    from src.memory.config import settings

    # 访问配置项
    user_id = settings.DEFAULT_USER_ID
    memory_size = settings.MEMORY_SIZE
"""

from src.core.config import settings

__all__ = ["settings"]
