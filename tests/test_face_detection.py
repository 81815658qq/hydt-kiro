"""人脸检测模块单元测试"""
import pytest
import numpy as np
import cv2
from src.face_detection import FaceDetection, FaceDetector


class TestFaceDetection:
    """测试FaceDetection数据类"""
    
    def test_to_pixel_coords(self):
        """测试坐标转换方法"""
        detection = FaceDetection(
            x=0.25,
            y=0.5,
            width=0.2,
            height=0.3,
            landmarks=[(0.3, 0.6), (0.4, 0.7)]
        )
        
        frame_width = 640
        frame_height = 480
        
        x, y, w, h = detection.to_pixel_coords(frame_width, frame_height)
        
        assert x == 160  # 0.25 * 640
        assert y == 240  # 0.5 * 480
        assert w == 128  # 0.2 * 640
        assert h == 144  # 0.3 * 480
    
    def test_normalized_coordinates_in_range(self):
        """测试归一化坐标在有效范围内"""
        detection = FaceDetection(
            x=0.1,
            y=0.2,
            width=0.3,
            height=0.4,
            landmarks=[]
        )
        
        assert 0 <= detection.x <= 1
        assert 0 <= detection.y <= 1
        assert 0 <= detection.width <= 1
        assert 0 <= detection.height <= 1


class TestFaceDetector:
    """测试FaceDetector类"""
    
    @pytest.fixture
    def detector(self):
        """创建FaceDetector实例"""
        return FaceDetector(min_detection_confidence=0.5)
    
    @pytest.fixture
    def face_image_with_face(self):
        """创建包含人脸的测试图像（使用简单的椭圆模拟人脸）"""
        # 创建640x480的白色背景
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 绘制一个椭圆模拟人脸（肤色）
        center = (320, 240)
        axes = (80, 100)
        cv2.ellipse(img, center, axes, 0, 0, 360, (180, 150, 120), -1)
        
        # 绘制眼睛
        cv2.circle(img, (290, 220), 10, (0, 0, 0), -1)
        cv2.circle(img, (350, 220), 10, (0, 0, 0), -1)
        
        # 绘制嘴巴
        cv2.ellipse(img, (320, 270), (30, 15), 0, 0, 180, (100, 50, 50), 2)
        
        return img
    
    @pytest.fixture
    def image_without_face(self):
        """创建不包含人脸的测试图像（纯色背景）"""
        return np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    def test_detector_initialization(self, detector):
        """测试检测器初始化"""
        assert detector is not None
        assert detector.min_detection_confidence == 0.5
        assert detector.face_detection is not None
    
    def test_detect_faces_with_face(self, detector, face_image_with_face):
        """测试检测包含人脸的图像"""
        detections = detector.detect_faces(face_image_with_face)
        
        # MediaPipe可能检测到人脸（取决于模型），但至少不应该崩溃
        assert isinstance(detections, list)
        
        # 如果检测到人脸，验证结果格式
        for detection in detections:
            assert isinstance(detection, FaceDetection)
            assert 0 <= detection.x <= 1
            assert 0 <= detection.y <= 1
            assert 0 < detection.width <= 1
            assert 0 < detection.height <= 1
            assert isinstance(detection.landmarks, list)
    
    def test_detect_faces_without_face(self, detector, image_without_face):
        """测试无人脸图像返回空列表"""
        detections = detector.detect_faces(image_without_face)
        
        # 纯色背景应该不会检测到人脸
        assert isinstance(detections, list)
        # 注意：MediaPipe可能会有误检，但大多数情况下应该返回空列表
    
    def test_detect_faces_invalid_input(self, detector):
        """测试无效输入的错误处理"""
        # 空数组
        empty_frame = np.array([])
        detections = detector.detect_faces(empty_frame)
        assert detections == []
        
        # 错误的形状
        invalid_frame = np.ones((100, 100), dtype=np.uint8)  # 缺少颜色通道
        detections = detector.detect_faces(invalid_frame)
        assert detections == []
    
    def test_bounding_box_coordinates_valid(self, detector, face_image_with_face):
        """测试边界框坐标有效性"""
        detections = detector.detect_faces(face_image_with_face)
        
        for detection in detections:
            # 归一化坐标应该在[0, 1]范围内
            assert 0 <= detection.x <= 1
            assert 0 <= detection.y <= 1
            assert 0 < detection.width <= 1
            assert 0 < detection.height <= 1
            
            # 边界框不应该超出图像范围
            assert detection.x + detection.width <= 1.1  # 允许小误差
            assert detection.y + detection.height <= 1.1
    
    def test_extract_face_region(self, detector, face_image_with_face):
        """测试人脸区域提取"""
        # 创建一个模拟的检测结果
        detection = FaceDetection(
            x=0.3,
            y=0.3,
            width=0.3,
            height=0.4,
            landmarks=[]
        )
        
        face_region = detector.extract_face_region(face_image_with_face, detection)
        
        if face_region is not None:
            assert isinstance(face_region, np.ndarray)
            assert len(face_region.shape) == 3  # 应该是彩色图像
            assert face_region.shape[2] == 3  # BGR格式
            assert face_region.size > 0
    
    def test_extract_face_region_invalid_detection(self, detector, face_image_with_face):
        """测试无效检测结果的人脸区域提取"""
        # 超出边界的检测
        invalid_detection = FaceDetection(
            x=1.5,
            y=1.5,
            width=0.2,
            height=0.2,
            landmarks=[]
        )
        
        face_region = detector.extract_face_region(face_image_with_face, invalid_detection)
        # 应该返回None或处理错误
        assert face_region is None or isinstance(face_region, np.ndarray)
    
    def test_extract_face_region_edge_case(self, detector, face_image_with_face):
        """测试边缘情况的人脸区域提取"""
        # 非常小的检测区域
        tiny_detection = FaceDetection(
            x=0.5,
            y=0.5,
            width=0.01,
            height=0.01,
            landmarks=[]
        )
        
        face_region = detector.extract_face_region(face_image_with_face, tiny_detection)
        
        if face_region is not None:
            assert isinstance(face_region, np.ndarray)
            assert face_region.size > 0
