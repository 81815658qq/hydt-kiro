"""视频采集模块"""
import cv2
import numpy as np
import time
from typing import Optional, List
from src.logger import logger


class CameraConnectionError(Exception):
    """摄像头连接错误"""
    pass


def detect_available_cameras(max_cameras: int = 5) -> List[int]:
    """
    检测系统中可用的摄像头
    
    Args:
        max_cameras: 最大检测摄像头数量
        
    Returns:
        可用摄像头索引列表
    """
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
            logger.info(f"检测到摄像头: {i}")
    
    if not available_cameras:
        logger.warning("未检测到任何可用摄像头")
    else:
        logger.info(f"共检测到 {len(available_cameras)} 个摄像头: {available_cameras}")
    
    return available_cameras


class VideoCapture:
    """视频采集器 - 负责从USB摄像头采集视频流"""
    
    def __init__(self, camera_index: int = 0, max_reconnect_attempts: int = 3, resolution: tuple = (1280, 720)):
        """
        初始化摄像头连接
        
        Args:
            camera_index: 摄像头索引，默认为0（系统默认摄像头）
            max_reconnect_attempts: 最大重连尝试次数
            resolution: 视频分辨率 (width, height)，默认为720p
            
        Raises:
            CameraConnectionError: 摄像头连接失败时抛出
        """
        self.camera_index = camera_index
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = 1.0  # 重连延迟（秒）
        self.resolution = resolution
        self.capture = None
        
        # 尝试连接摄像头
        self._connect()
    
    def _connect(self):
        """连接摄像头，支持重试"""
        for attempt in range(self.max_reconnect_attempts):
            try:
                logger.info(f"尝试连接摄像头 {self.camera_index} (尝试 {attempt + 1}/{self.max_reconnect_attempts})")
                
                self.capture = cv2.VideoCapture(self.camera_index)
                
                if self.capture.isOpened():
                    # 设置分辨率
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    
                    # 获取实际设置的分辨率
                    actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    logger.info(f"成功连接到摄像头 {self.camera_index}")
                    logger.info(f"请求分辨率: {self.resolution[0]}x{self.resolution[1]}, 实际分辨率: {actual_width}x{actual_height}")
                    return
                
                # 连接失败，释放资源
                if self.capture:
                    self.capture.release()
                    self.capture = None
                
                if attempt < self.max_reconnect_attempts - 1:
                    logger.warning(f"摄像头连接失败，{self.reconnect_delay}秒后重试...")
                    time.sleep(self.reconnect_delay)
                    
            except Exception as e:
                logger.error(f"连接摄像头时发生异常: {e}")
                if attempt < self.max_reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)
        
        # 所有尝试都失败
        error_msg = f"无法打开摄像头 {self.camera_index}，已尝试 {self.max_reconnect_attempts} 次"
        logger.error(error_msg)
        raise CameraConnectionError(error_msg)
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        读取一帧图像，支持自动重连
        
        Returns:
            BGR格式的numpy数组，如果读取失败返回None
        """
        if not self.is_opened():
            logger.warning("摄像头未打开，尝试重新连接...")
            try:
                self._reconnect()
            except CameraConnectionError:
                logger.error("摄像头重连失败")
                return None
        
        try:
            ret, frame = self.capture.read()
            
            if not ret or frame is None:
                logger.warning("读取视频帧失败，尝试重新连接...")
                try:
                    self._reconnect()
                    # 重连后再次尝试读取
                    ret, frame = self.capture.read()
                    if not ret or frame is None:
                        logger.error("重连后仍无法读取帧")
                        return None
                except CameraConnectionError:
                    logger.error("摄像头重连失败")
                    return None
            
            return frame
            
        except Exception as e:
            logger.error(f"读取帧时发生异常: {e}")
            return None
    
    def _reconnect(self):
        """重新连接摄像头"""
        logger.info("正在重新连接摄像头...")
        
        # 释放旧连接
        if self.capture:
            self.capture.release()
            self.capture = None
        
        # 尝试重新连接
        self._connect()
    
    def is_opened(self) -> bool:
        """
        检查摄像头是否正常打开
        
        Returns:
            True表示摄像头已打开，False表示未打开
        """
        return self.capture is not None and self.capture.isOpened()
    
    def release(self):
        """释放摄像头资源"""
        try:
            if self.capture is not None:
                self.capture.release()
                logger.info(f"已释放摄像头 {self.camera_index} 资源")
        except Exception as e:
            logger.error(f"释放摄像头资源时发生异常: {e}")
    
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
