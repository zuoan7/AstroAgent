import logging
import os
from logging.handlers import RotatingFileHandler

# 日志配置
def setup_logger(name="AstroAgent", log_dir="./logs"):
    """
    设置日志配置
    :param name: 日志名称
    :param log_dir: 日志目录
    :return: 配置好的logger实例
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 日志文件路径
    log_file = os.path.join(log_dir, f"{name}.log")
    
    # 创建文件handler，支持日志轮转
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 定义日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建全局logger实例
logger = setup_logger()
