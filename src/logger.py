"""日志配置模块"""
import logging
import sys
from pathlib import Path
from src.config import LOG_FILE

def setup_logger(name: str = "greeter", level: int = logging.INFO) -> logging.Logger:
    """
    设置并返回配置好的日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        
    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str = "greeter") -> logging.Logger:
    """
    获取或创建指定名称的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        Logger实例
    """
    return setup_logger(name)

# 创建默认logger实例
logger = setup_logger()
