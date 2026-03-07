from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """项目配置"""
    # 千问模型配置
    DASHSCOPE_API_KEY: str
    MODEL_NAME: str = "qwen-max"
    EMBEDDING_MODEL_NAME: str = "text-embedding-v2"
    
    # RAG配置
    RAG_ENABLED: bool = True
    VECTOR_DB_PATH: str = "./vector_db"
    
    # 记忆配置
    MEMORY_SIZE: int = 15
    MEMORY_WINDOW: int = 8
    
    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局配置实例
settings = Settings()
