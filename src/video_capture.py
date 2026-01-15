"""视频采集模块"""
import cv2
import numpy as np
from typing import Optional
from src.logger import logger


class CameraConnectionError(Exception):
    """摄像头连接错误"""
    pass


class VideoCapture:
    """视频采集器 - 负责从USB摄像头采集视频流"""
    
    def __init__(self, camera_index: int = 0):
        """
        初始化摄像头连接
        
        Args:
            camera_index: 摄像头索引，默认为0（系统默认摄像头）
            
        Raises:
            CameraConnectionError: 摄像头连接失败时抛出
        """
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        
        if not self.capture.isOpened():
            error_msg = f"无法打开摄像头 {camera_index}"
            logger.error(error_msg)
            raise CameraConnectionError(error_msg)
        
        logger.info(f"成功连接到摄像头 {camera_index}")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        读取一帧图像
        
        Returns:
            BGR格式的numpy数组，如果读取失败返回None
        """
        if not self.is_opened():
            logger.warning("摄像头未打开，无法读取帧")
            return None
        
        ret, frame = self.capture.read()
        
        if not ret or frame is None:
            logger.warning("读取视频帧失败")
            return None
        
        return frame
    
    def is_opened(self) -> bool:
        """
        检查摄像头是否正常打开
        
        Returns:
            True表示摄像头已打开，False表示未打开
        """
        return self.capture is not None and self.capture.isOpened()
    
    def release(self):
        """释放摄像头资源"""
        if self.capture is not None:
            self.capture.release()
            logger.info(f"已释放摄像头 {self.camera_index} 资源")
    
    def get_fps(self) -> float:
        """
        获取摄像头帧率
        
        Returns:
            摄像头的帧率（FPS）
        """
        if not self.is_opened():
            logger.warning("摄像头未打开，无法获取帧率")
            return 0.0
        
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        # 如果无法获取FPS（某些摄像头返回0），使用默认值
        if fps == 0:
            fps = 30.0
            logger.warning(f"无法获取摄像头FPS，使用默认值 {fps}")
        
        return fps
    
    def get_resolution(self) -> tuple[int, int]:
        """
        获取摄像头分辨率
        
        Returns:
            (width, height) 元组
        """
        if not self.is_opened():
            logger.warning("摄像头未打开，无法获取分辨率")
            return (0, 0)
        
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return (width, height)
    
    def __enter__(self):
        """支持上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持上下文管理器 - 自动释放资源"""
        self.release()
        return False
