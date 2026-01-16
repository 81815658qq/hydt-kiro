"""错误处理测试

测试各模块的错误处理能力，包括：
- 人脸检测失败场景
- 特征提取失败场景
- 摄像头断开场景
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.video_capture import VideoCapture, CameraConnectionError
from src.face_detection import FaceDetector
from src.face_feature_extractor import FaceFeatureExtractor
from src.greeter_service import GreeterService


class TestCameraErrorHandling:
    """测试摄像头错误处理"""
    
    def test_camera_connection_failure(self):
        """测试摄像头连接失败场景"""
        with patch('cv2.VideoCapture') as mock_capture:
            # 模拟摄像头无法打开
            mock_instance = Mock()
            mock_instance.isOpened.return_value = False
            mock_capture.return_value = mock_instance
            
            # 应该抛出CameraConnectionError
            with pytest.raises(CameraConnectionError):
                VideoCapture(camera_index=0)
    
    def test_camera_reconnect_on_read_failure(self):
        """测试读取失败时的重连逻辑"""
        with patch('cv2.VideoCapture') as mock_capture:
            mock_instance = Mock()
            
            # 第一次打开成功
            mock_instance.isOpened.return_value = True
            
            # 第一次读取失败，第二次成功
            mock_instance.read.side_effect = [
                (False, None),  # 第一次读取失败
                (True, np.zeros((480, 640, 3), dtype=np.uint8))  # 重连后成功
            ]
            
            mock_capture.return_value = mock_instance
            
            video_capture = VideoCapture(camera_index=0)
            
            # 第一次读取应该触发重连并返回帧
            frame = video_capture.read_frame()
            
            # 应该成功读取到帧
            assert frame is not None
            assert frame.shape == (480, 640, 3)
    
    def test_camera_release_with_exception(self):
        """测试释放摄像头时的异常处理"""
        with patch('cv2.VideoCapture') as mock_capture:
            mock_instance = Mock()
            mock_instance.isOpened.return_value = True
            mock_instance.release.side_effect = Exception("Release failed")
            mock_capture.return_value = mock_instance
            
            video_capture = VideoCapture(camera_index=0)
            
            # 释放时不应该抛出异常
            video_capture.release()  # 应该正常完成


class TestFaceDetectionErrorHandling:
    """测试人脸检测错误处理"""
    
    def test_face_detection_with_invalid_frame(self):
        """测试使用无效帧进行人脸检测"""
        detector = FaceDetector()
        
        # 测试None帧 - 应该返回空列表而不是抛出异常
        result = detector.detect_faces(None)
        assert result == []
        
        # 测试空数组
        empty_frame = np.array([])
        result = detector.detect_faces(empty_frame)
        assert result == []
    
    def test_face_detection_with_corrupted_frame(self):
        """测试使用损坏的帧进行人脸检测"""
        detector = FaceDetector()
        
        # 创建一个形状不正确的数组
        corrupted_frame = np.zeros((10, 10), dtype=np.uint8)  # 缺少颜色通道
        
        # 应该返回空列表而不是崩溃
        result = detector.detect_faces(corrupted_frame)
        assert isinstance(result, list)
    
    def test_extract_face_region_with_invalid_coordinates(self):
        """测试使用无效坐标提取人脸区域"""
        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 创建一个坐标超出范围的检测结果
        from src.face_detection import FaceDetection
        invalid_detection = FaceDetection(
            x=1.5,  # 超出范围
            y=1.5,
            width=0.5,
            height=0.5,
            landmarks=[]
        )
        
        # 应该返回None而不是崩溃
        result = detector.extract_face_region(frame, invalid_detection)
        assert result is None


class TestFeatureExtractionErrorHandling:
    """测试特征提取错误处理"""
    
    def test_feature_extraction_with_none_image(self):
        """测试使用None图像提取特征"""
        extractor = FaceFeatureExtractor()
        
        result = extractor.extract_features(None)
        assert result is None
    
    def test_feature_extraction_with_empty_image(self):
        """测试使用空图像提取特征"""
        extractor = FaceFeatureExtractor()
        
        empty_image = np.array([])
        result = extractor.extract_features(empty_image)
        assert result is None
    
    def test_feature_extraction_with_tiny_image(self):
        """测试使用极小图像提取特征"""
        extractor = FaceFeatureExtractor()
        
        # 创建一个1x1的图像
        tiny_image = np.zeros((1, 1, 3), dtype=np.uint8)
        
        # 应该能够处理并返回特征（可能质量不高）
        result = extractor.extract_features(tiny_image)
        
        # 应该返回128维特征向量或None
        assert result is None or (result is not None and len(result) == 128)
    
    def test_similarity_with_none_features(self):
        """测试使用None特征计算相似度"""
        extractor = FaceFeatureExtractor()
        
        valid_features = np.random.rand(128)
        
        # 测试第一个参数为None
        result = extractor.compute_similarity(None, valid_features)
        assert result == 0.0
        
        # 测试第二个参数为None
        result = extractor.compute_similarity(valid_features, None)
        assert result == 0.0
        
        # 测试两个都为None
        result = extractor.compute_similarity(None, None)
        assert result == 0.0
    
    def test_similarity_with_mismatched_dimensions(self):
        """测试使用维度不匹配的特征计算相似度"""
        extractor = FaceFeatureExtractor()
        
        features1 = np.random.rand(128)
        features2 = np.random.rand(64)  # 不同维度
        
        result = extractor.compute_similarity(features1, features2)
        assert result == 0.0


class TestGreeterServiceErrorHandling:
    """测试迎宾服务错误处理"""
    
    def test_greeter_service_with_camera_failure(self):
        """测试摄像头失败时的服务初始化"""
        with patch('src.greeter_service.VideoCapture') as mock_video_capture:
            mock_video_capture.side_effect = CameraConnectionError("Camera not found")
            
            # 应该抛出CameraConnectionError
            with pytest.raises(CameraConnectionError):
                GreeterService(camera_index=99)
    
    def test_process_frame_with_detection_failure(self):
        """测试人脸检测失败时的帧处理"""
        with patch('src.greeter_service.VideoCapture') as mock_video_capture, \
             patch('src.greeter_service.FaceDetector') as mock_detector:
            
            # 模拟摄像头正常
            mock_vc_instance = Mock()
            mock_vc_instance.isOpened.return_value = True
            mock_vc_instance.read_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_video_capture.return_value = mock_vc_instance
            
            # 模拟人脸检测失败
            mock_det_instance = Mock()
            mock_det_instance.detect_faces.side_effect = Exception("Detection failed")
            mock_detector.return_value = mock_det_instance
            
            service = GreeterService(camera_index=0)
            
            # 应该能够处理异常并返回帧
            frame = service.process_frame()
            assert frame is not None
    
    def test_process_frame_with_feature_extraction_failure(self):
        """测试特征提取失败时的帧处理"""
        with patch('src.greeter_service.VideoCapture') as mock_video_capture, \
             patch('src.greeter_service.FaceDetector') as mock_detector, \
             patch('src.greeter_service.FaceFeatureExtractor') as mock_extractor:
            
            # 模拟摄像头正常
            mock_vc_instance = Mock()
            mock_vc_instance.isOpened.return_value = True
            mock_vc_instance.read_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_video_capture.return_value = mock_vc_instance
            
            # 模拟检测到人脸
            from src.face_detection import FaceDetection
            mock_det_instance = Mock()
            mock_det_instance.detect_faces.return_value = [
                FaceDetection(x=0.2, y=0.2, width=0.3, height=0.3, landmarks=[])
            ]
            mock_det_instance.extract_face_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_detector.return_value = mock_det_instance
            
            # 模拟特征提取失败
            mock_ext_instance = Mock()
            mock_ext_instance.extract_features.side_effect = Exception("Extraction failed")
            mock_extractor.return_value = mock_ext_instance
            
            service = GreeterService(camera_index=0)
            
            # 应该能够处理异常并返回帧
            frame = service.process_frame()
            assert frame is not None
    
    def test_get_statistics_with_error(self):
        """测试获取统计信息时的错误处理"""
        with patch('src.greeter_service.VideoCapture') as mock_video_capture, \
             patch('src.greeter_service.FaceDatabase') as mock_database:
            
            # 模拟摄像头正常
            mock_vc_instance = Mock()
            mock_vc_instance.isOpened.return_value = True
            mock_video_capture.return_value = mock_vc_instance
            
            # 模拟数据库初始化正常，但查询时失败
            mock_db_instance = Mock()
            mock_db_instance.get_total_visitors.return_value = 0  # 初始化时正常
            mock_database.return_value = mock_db_instance
            
            service = GreeterService(camera_index=0)
            
            # 现在让get_total_visitors抛出异常
            mock_db_instance.get_total_visitors.side_effect = Exception("Database error")
            
            # 应该返回默认值而不是崩溃
            stats = service.get_statistics()
            assert isinstance(stats, dict)
            assert "total_visitors" in stats
            assert stats["total_visitors"] == 0  # 默认值


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
