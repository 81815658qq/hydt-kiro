"""VideoCapture模块的单元测试和属性测试"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, settings, strategies as st
from src.video_capture import VideoCapture, CameraConnectionError


class TestVideoCapture:
    """VideoCapture类的单元测试"""
    
    def test_camera_connection_success(self):
        """测试摄像头连接成功场景 - 需求：1.1"""
        with patch('cv2.VideoCapture') as mock_capture_class:
            # 模拟成功的摄像头连接
            mock_capture = Mock()
            mock_capture.isOpened.return_value = True
            mock_capture_class.return_value = mock_capture
            
            # 创建VideoCapture实例
            video_capture = VideoCapture(camera_index=0)
            
            # 验证摄像头已打开
            assert video_capture.is_opened() is True
            mock_capture_class.assert_called_once_with(0)
    
    def test_camera_connection_failure(self):
        """测试摄像头不存在时的错误处理 - 需求：1.1, 1.3"""
        with patch('cv2.VideoCapture') as mock_capture_class:
            # 模拟摄像头连接失败
            mock_capture = Mock()
            mock_capture.isOpened.return_value = False
            mock_capture_class.return_value = mock_capture
            
            # 验证抛出CameraConnectionError异常
            with pytest.raises(CameraConnectionError) as exc_info:
                VideoCapture(camera_index=99)
            
            assert "无法打开摄像头" in str(exc_info.value)
    
    def test_read_frame_success(self):
        """测试成功读取视频帧"""
        with patch('cv2.VideoCapture') as mock_capture_class:
            # 模拟成功读取帧
            mock_capture = Mock()
            mock_capture.isOpened.return_value = True
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_capture.read.return_value = (True, test_frame)
            mock_capture_class.return_value = mock_capture
            
            video_capture = VideoCapture(camera_index=0)
            frame = video_capture.read_frame()
            
            # 验证返回的帧不为None
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)
    
    def test_read_frame_failure(self):
        """测试读取帧失败的情况"""
        with patch('cv2.VideoCapture') as mock_capture_class:
            # 模拟读取失败
            mock_capture = Mock()
            mock_capture.isOpened.return_value = True
            mock_capture.read.return_value = (False, None)
            mock_capture_class.return_value = mock_capture
            
            video_capture = VideoCapture(camera_index=0)
            frame = video_capture.read_frame()
            
            # 验证返回None
            assert frame is None
    
    def test_get_fps(self):
        """测试帧率获取功能 - 需求：1.1, 1.3"""
        with patch('cv2.VideoCapture') as mock_capture_class:
            # 模拟FPS获取
            mock_capture = Mock()
            mock_capture.isOpened.return_value = True
            mock_capture.get.return_value = 30.0
            mock_capture_class.return_value = mock_capture
            
            video_capture = VideoCapture(camera_index=0)
            fps = video_capture.get_fps()
            
            # 验证FPS值
            assert fps == 30.0
            assert fps >= 15  # 满足最小FPS要求
    
    def test_get_fps_default_when_zero(self):
        """测试当摄像头返回0 FPS时使用默认值"""
        with patch('cv2.VideoCapture') as mock_capture_class:
            # 模拟FPS返回0
            mock_capture = Mock()
            mock_capture.isOpened.return_value = True
            mock_capture.get.return_value = 0.0
            mock_capture_class.return_value = mock_capture
            
            video_capture = VideoCapture(camera_index=0)
            fps = video_capture.get_fps()
            
            # 验证使用默认FPS
            assert fps == 30.0
    
    def test_get_resolution(self):
        """测试获取摄像头分辨率"""
        with patch('cv2.VideoCapture') as mock_capture_class:
            # 模拟分辨率获取
            mock_capture = Mock()
            mock_capture.isOpened.return_value = True
            mock_capture.get.side_effect = [640.0, 480.0]  # width, height
            mock_capture_class.return_value = mock_capture
            
            video_capture = VideoCapture(camera_index=0)
            width, height = video_capture.get_resolution()
            
            # 验证分辨率
            assert width == 640
            assert height == 480
    
    def test_release(self):
        """测试资源释放"""
        with patch('cv2.VideoCapture') as mock_capture_class:
            mock_capture = Mock()
            mock_capture.isOpened.return_value = True
            mock_capture_class.return_value = mock_capture
            
            video_capture = VideoCapture(camera_index=0)
            video_capture.release()
            
            # 验证release被调用
            mock_capture.release.assert_called_once()
    
    def test_context_manager(self):
        """测试上下文管理器支持"""
        with patch('cv2.VideoCapture') as mock_capture_class:
            mock_capture = Mock()
            mock_capture.isOpened.return_value = True
            mock_capture_class.return_value = mock_capture
            
            # 使用with语句
            with VideoCapture(camera_index=0) as video_capture:
                assert video_capture.is_opened() is True
            
            # 验证资源被自动释放
            mock_capture.release.assert_called_once()
    
    def test_is_opened_when_closed(self):
        """测试摄像头关闭后is_opened返回False"""
        with patch('cv2.VideoCapture') as mock_capture_class:
            mock_capture = Mock()
            # 初始化时调用一次，然后两次is_opened调用
            mock_capture.isOpened.side_effect = [True, True, False]
            mock_capture_class.return_value = mock_capture
            
            video_capture = VideoCapture(camera_index=0)
            assert video_capture.is_opened() is True
            
            # 模拟关闭
            assert video_capture.is_opened() is False


class TestVideoCaptureProperties:
    """VideoCapture类的基于属性的测试"""
    
    @settings(max_examples=100)
    @given(
        width=st.integers(min_value=320, max_value=1920),
        height=st.integers(min_value=240, max_value=1080),
        num_frames=st.integers(min_value=5, max_value=20)
    )
    def test_video_frame_resolution_consistency(self, width, height, num_frames):
        """
        属性 1：视频帧分辨率一致性
        
        对于任何摄像头配置，连续读取的所有视频帧应该具有相同的分辨率。
        这验证了需求1.4：视频处理器应保持一致的分辨率。
        
        验证需求：1.4
        Feature: welcome-greeter, Property 1: 视频帧分辨率一致性
        """
        with patch('cv2.VideoCapture') as mock_capture_class:
            # 模拟摄像头返回固定分辨率的帧
            mock_capture = Mock()
            mock_capture.isOpened.return_value = True
            mock_capture.get.side_effect = lambda prop: {
                3: float(width),   # CAP_PROP_FRAME_WIDTH
                4: float(height),  # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 30.0)
            
            # 创建多个不同的帧，但分辨率相同
            frames = []
            for i in range(num_frames):
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                frames.append((True, frame))
            
            mock_capture.read.side_effect = frames
            mock_capture_class.return_value = mock_capture
            
            # 创建VideoCapture实例
            video_capture = VideoCapture(camera_index=0)
            
            # 获取摄像头声明的分辨率
            declared_width, declared_height = video_capture.get_resolution()
            
            # 读取所有帧并验证分辨率一致性
            resolutions = []
            for _ in range(num_frames):
                frame = video_capture.read_frame()
                if frame is not None:
                    # 获取实际帧的分辨率 (height, width, channels)
                    frame_height, frame_width = frame.shape[:2]
                    resolutions.append((frame_width, frame_height))
            
            # 属性验证：
            # 1. 所有帧的分辨率应该相同
            if len(resolutions) > 0:
                first_resolution = resolutions[0]
                for resolution in resolutions:
                    assert resolution == first_resolution, \
                        f"帧分辨率不一致：期望 {first_resolution}，实际 {resolution}"
                
                # 2. 帧的分辨率应该与摄像头声明的分辨率一致
                assert first_resolution == (declared_width, declared_height), \
                    f"帧分辨率与声明不一致：声明 ({declared_width}, {declared_height})，实际 {first_resolution}"
            
            video_capture.release()
