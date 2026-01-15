"""迎宾服务模块

核心业务逻辑，协调各组件完成迎宾功能。
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from src.logger import get_logger
from src.video_capture import VideoCapture, CameraConnectionError
from src.face_detection import FaceDetector, FaceDetection
from src.face_feature_extractor import FaceFeatureExtractor
from src.face_database import FaceDatabase, VisitorRecord
from src.blessing_generator import BlessingGenerator
from src.video_renderer import VideoRenderer

logger = get_logger(__name__)


class GreeterService:
    """迎宾服务，协调各组件完成迎宾功能"""
    
    def __init__(
        self,
        camera_index: int = 0,
        similarity_threshold: float = 0.7,
        storage_dir: str = "./face_data"
    ):
        """
        初始化迎宾服务
        
        Args:
            camera_index: 摄像头索引
            similarity_threshold: 人脸相似度阈值
            storage_dir: 人脸数据存储目录
            
        Raises:
            CameraConnectionError: 摄像头连接失败时抛出
        """
        logger.info("Initializing GreeterService...")
        
        self.similarity_threshold = similarity_threshold
        
        # 初始化各组件
        try:
            self.video_capture = VideoCapture(camera_index)
            self.face_detector = FaceDetector(min_detection_confidence=0.5)
            self.feature_extractor = FaceFeatureExtractor()
            self.face_database = FaceDatabase(storage_dir)
            self.blessing_generator = BlessingGenerator()
            self.video_renderer = VideoRenderer()
            
            logger.info("GreeterService initialized successfully")
            logger.info(f"Camera index: {camera_index}")
            logger.info(f"Similarity threshold: {similarity_threshold}")
            logger.info(f"Storage directory: {storage_dir}")
            logger.info(f"Existing visitors: {self.face_database.get_total_visitors()}")
            
        except CameraConnectionError as e:
            logger.error(f"Failed to initialize GreeterService: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during GreeterService initialization: {e}")
            raise
    
    def process_frame(self) -> Optional[np.ndarray]:
        """
        处理一帧：检测人脸、识别访客、渲染祝福语
        
        Returns:
            渲染后的帧（BGR格式），如果处理失败则返回None
        """
        try:
            # 1. 从摄像头读取帧
            frame = self.video_capture.read_frame()
            if frame is None:
                logger.warning("Failed to read frame from camera")
                return None
            
            # 2. 检测所有人脸
            detections = self.face_detector.detect_faces(frame)
            
            # 3. 处理每个检测到的人脸
            blessings = []
            for detection in detections:
                blessing = self._process_face(frame, detection)
                blessings.append(blessing)
            
            # 4. 渲染帧
            rendered_frame = self.video_renderer.render_frame(
                frame,
                detections,
                blessings,
                self.face_database.get_total_visitors()
            )
            
            return rendered_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def _process_face(
        self, 
        frame: np.ndarray, 
        detection: FaceDetection
    ) -> str:
        """
        处理单个人脸：提取特征、识别访客、分配祝福语
        
        Args:
            frame: 视频帧
            detection: 人脸检测结果
            
        Returns:
            祝福语字符串
        """
        try:
            # 提取人脸区域
            face_region = self.face_detector.extract_face_region(frame, detection)
            if face_region is None:
                logger.warning("Failed to extract face region")
                return "欢迎光临"
            
            # 提取特征向量
            features = self.feature_extractor.extract_features(face_region)
            if features is None:
                logger.warning("Failed to extract features")
                return "欢迎光临"
            
            # 在数据库中查找匹配的访客
            matching_visitor = self.face_database.find_matching_visitor(
                features, 
                self.similarity_threshold
            )
            
            if matching_visitor:
                # 已知访客，使用已有祝福语
                logger.info(f"Recognized visitor: {matching_visitor.visitor_id}")
                return matching_visitor.blessing
            else:
                # 新访客，添加到数据库
                blessing = self.blessing_generator.get_random_blessing()
                visitor = self.face_database.add_visitor(
                    features,
                    face_region,
                    blessing
                )
                logger.info(f"New visitor added: {visitor.visitor_id} with blessing: {blessing}")
                return blessing
                
        except Exception as e:
            logger.error(f"Error processing face: {e}")
            return "欢迎光临"
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            包含统计数据的字典
        """
        return {
            "total_visitors": self.face_database.get_total_visitors(),
            "camera_fps": self.video_capture.get_fps(),
            "camera_resolution": self.video_capture.get_resolution()
        }
    
    def start(self):
        """启动服务"""
        logger.info("GreeterService started")
    
    def stop(self):
        """停止服务并释放资源"""
        try:
            logger.info("Stopping GreeterService...")
            
            # 释放摄像头资源
            if hasattr(self, 'video_capture'):
                self.video_capture.release()
            
            # 保存数据库
            if hasattr(self, 'face_database'):
                self.face_database.save_to_disk()
            
            logger.info("GreeterService stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping GreeterService: {e}")
    
    def __enter__(self):
        """支持上下文管理器"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持上下文管理器 - 自动释放资源"""
        self.stop()
        return False
