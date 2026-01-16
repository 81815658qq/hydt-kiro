"""人脸检测模块"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import mediapipe as mp
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FaceDetection:
    """人脸检测结果数据类"""
    x: float  # 边界框左上角x坐标（归一化0-1）
    y: float  # 边界框左上角y坐标（归一化0-1）
    width: float  # 边界框宽度（归一化0-1）
    height: float  # 边界框高度（归一化0-1）
    landmarks: List[Tuple[float, float]]  # 面部关键点
    
    def to_pixel_coords(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """
        转换为像素坐标
        
        Args:
            frame_width: 帧宽度（像素）
            frame_height: 帧高度（像素）
            
        Returns:
            (x, y, width, height) 像素坐标元组
        """
        return (
            int(self.x * frame_width),
            int(self.y * frame_height),
            int(self.width * frame_width),
            int(self.height * frame_height)
        )


class FaceDetector:
    """人脸检测器，使用MediaPipe检测视频帧中的所有人脸"""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        初始化MediaPipe人脸检测模型
        
        Args:
            min_detection_confidence: 最小检测置信度阈值（0-1）
        """
        self.min_detection_confidence = min_detection_confidence
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        logger.info(f"FaceDetector initialized with confidence threshold: {min_detection_confidence}")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        检测帧中的所有人脸
        
        Args:
            frame: BGR格式的视频帧（numpy数组）
            
        Returns:
            人脸检测结果列表，如果没有检测到人脸则返回空列表
        """
        try:
            # MediaPipe需要RGB格式
            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
            
            # 执行检测
            results = self.face_detection.process(rgb_frame)
            
            if not results.detections:
                return []
            
            detections = []
            for detection in results.detections:
                # 获取边界框（相对坐标）
                bbox = detection.location_data.relative_bounding_box
                
                # 验证坐标有效性（允许MediaPipe返回略微超出范围的值）
                if bbox.xmin < 0 or bbox.ymin < 0 or bbox.width <= 0 or bbox.height <= 0:
                    logger.warning("Invalid bounding box detected, skipping")
                    continue
                
                # 裁剪坐标到有效范围[0, 1]
                x = max(0.0, min(1.0, bbox.xmin))
                y = max(0.0, min(1.0, bbox.ymin))
                width = max(0.01, min(1.0, bbox.width))
                height = max(0.01, min(1.0, bbox.height))
                
                # 确保边界框不超出图像范围
                if x + width > 1.0:
                    width = 1.0 - x
                if y + height > 1.0:
                    height = 1.0 - y
                
                # 提取关键点（也需要裁剪到有效范围）
                landmarks = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        lx = max(0.0, min(1.0, keypoint.x))
                        ly = max(0.0, min(1.0, keypoint.y))
                        landmarks.append((lx, ly))
                
                face_detection = FaceDetection(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    landmarks=landmarks
                )
                detections.append(face_detection)
            
            logger.debug(f"Detected {len(detections)} face(s)")
            return detections
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def extract_face_region(
        self, 
        frame: np.ndarray, 
        detection: FaceDetection,
        expand_ratio: float = 0.0,
        expand_top: float = None,
        expand_bottom: float = None,
        expand_left: float = None,
        expand_right: float = None
    ) -> Optional[np.ndarray]:
        """
        从帧中提取人脸区域图像
        
        Args:
            frame: BGR格式的视频帧
            detection: 人脸检测结果
            expand_ratio: 边界框扩展比例（0.0-1.0），用于包含更多周边区域如头发
            expand_top: 顶部扩展比例（优先于expand_ratio）
            expand_bottom: 底部扩展比例（优先于expand_ratio）
            expand_left: 左侧扩展比例（优先于expand_ratio）
            expand_right: 右侧扩展比例（优先于expand_ratio）
            
        Returns:
            人脸区域图像（BGR格式），如果提取失败则返回None
        """
        try:
            height, width = frame.shape[:2]
            x, y, w, h = detection.to_pixel_coords(width, height)
            
            # 如果需要扩展边界框
            if expand_ratio > 0 or any([expand_top, expand_bottom, expand_left, expand_right]):
                # 使用指定的扩展比例，如果没有指定则使用默认的expand_ratio
                top_ratio = expand_top if expand_top is not None else expand_ratio
                bottom_ratio = expand_bottom if expand_bottom is not None else expand_ratio
                left_ratio = expand_left if expand_left is not None else expand_ratio
                right_ratio = expand_right if expand_right is not None else expand_ratio
                
                expand_w_left = int(w * left_ratio)
                expand_w_right = int(w * right_ratio)
                expand_h_top = int(h * top_ratio)
                expand_h_bottom = int(h * bottom_ratio)
                
                x = x - expand_w_left
                y = y - expand_h_top
                w = w + expand_w_left + expand_w_right
                h = h + expand_h_top + expand_h_bottom
            
            # 确保坐标在有效范围内
            x = max(0, x)
            y = max(0, y)
            x_end = min(width, x + w)
            y_end = min(height, y + h)
            
            if x >= x_end or y >= y_end:
                logger.warning("Invalid face region coordinates")
                return None
            
            face_region = frame[y:y_end, x:x_end]
            
            if face_region.size == 0:
                logger.warning("Extracted face region is empty")
                return None
            
            return face_region
            
        except Exception as e:
            logger.error(f"Face region extraction failed: {e}")
            return None
    
    def calculate_face_area(self, detection: FaceDetection, frame_width: int, frame_height: int) -> int:
        """
        计算人脸区域的像素面积
        
        Args:
            detection: 人脸检测结果
            frame_width: 帧宽度
            frame_height: 帧高度
            
        Returns:
            人脸区域的像素面积
        """
        x, y, w, h = detection.to_pixel_coords(frame_width, frame_height)
        return w * h
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
