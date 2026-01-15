"""迎宾服务属性测试

Feature: welcome-greeter, Property 11: 日志记录完整性
验证需求：7.5
"""
import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, patch
from hypothesis import given, settings, strategies as st, HealthCheck
import tempfile
import shutil

from src.greeter_service import GreeterService
from src.logger import get_logger


class TestGreeterServiceProperties:
    """GreeterService属性测试"""
    
    def _create_temp_storage_dir(self):
        """创建临时存储目录"""
        return Path(tempfile.mkdtemp())
    
    def _cleanup_temp_dir(self, temp_path):
        """清理临时目录"""
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    def _create_temp_log_file(self):
        """创建临时日志文件"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log', encoding='utf-8')
        temp_file.close()
        return Path(temp_file.name)
    
    def _cleanup_temp_file(self, temp_file):
        """清理临时文件"""
        try:
            temp_file.unlink()
        except:
            pass
    
    def _mock_video_capture(self):
        """创建模拟VideoCapture"""
        mock_vc = patch('src.greeter_service.VideoCapture')
        mock_vc_instance = mock_vc.start()
        mock_instance = Mock()
        mock_instance.read_frame.return_value = None
        mock_instance.is_opened.return_value = True
        mock_instance.get_fps.return_value = 30.0
        mock_instance.get_resolution.return_value = (640, 480)
        mock_vc_instance.return_value = mock_instance
        return mock_vc
    
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        camera_index=st.integers(min_value=0, max_value=5),
        similarity_threshold=st.floats(min_value=0.5, max_value=0.95)
    )
    def test_property_11_initialization_logging(
        self, 
        camera_index, 
        similarity_threshold
    ):
        """
        属性 11：日志记录完整性 - 初始化日志
        
        对于任何有效的初始化参数，系统应该记录初始化相关的日志条目
        
        验证需求：7.5
        Feature: welcome-greeter, Property 11: 日志记录完整性
        """
        # 创建临时目录和文件
        temp_storage_dir = self._create_temp_storage_dir()
        temp_log_file = self._create_temp_log_file()
        mock_vc = self._mock_video_capture()
        
        # 配置日志记录到临时文件
        logger = get_logger('src.greeter_service')
        
        # 添加临时文件处理器
        file_handler = logging.FileHandler(temp_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        try:
            # 创建服务
            service = GreeterService(
                camera_index=camera_index,
                similarity_threshold=similarity_threshold,
                storage_dir=str(temp_storage_dir)
            )
            
            # 停止服务以确保日志被写入
            service.stop()
            
            # 强制刷新日志
            file_handler.flush()
            
            # 读取日志内容
            log_content = temp_log_file.read_text(encoding='utf-8')
            
            # 验证关键初始化日志存在
            assert "Initializing GreeterService" in log_content or "初始化" in log_content
            assert "initialized successfully" in log_content or "成功" in log_content
            
            # 验证配置参数被记录
            assert str(camera_index) in log_content
            assert str(similarity_threshold) in log_content
            
        finally:
            # 清理handler
            logger.removeHandler(file_handler)
            file_handler.close()
            mock_vc.stop()
            self._cleanup_temp_dir(temp_storage_dir)
            self._cleanup_temp_file(temp_log_file)
    
    def test_property_11_new_visitor_logging(self):
        """
        属性 11：日志记录完整性 - 新访客添加日志
        
        对于任何新访客添加操作，系统应该记录相应的日志条目
        
        验证需求：7.5
        Feature: welcome-greeter, Property 11: 日志记录完整性
        """
        # 创建临时目录和文件
        temp_storage_dir = self._create_temp_storage_dir()
        temp_log_file = self._create_temp_log_file()
        mock_vc = self._mock_video_capture()
        
        # 配置日志
        logger = get_logger('src.face_database')
        file_handler = logging.FileHandler(temp_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        try:
            service = GreeterService(
                camera_index=0,
                similarity_threshold=0.7,
                storage_dir=str(temp_storage_dir)
            )
            
            # 清空日志文件以便只记录新操作
            temp_log_file.write_text('', encoding='utf-8')
            
            # 模拟添加新访客（通过直接调用数据库方法）
            import numpy as np
            features = np.random.rand(128)
            face_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            blessing = "测试祝福"
            
            visitor = service.face_database.add_visitor(features, face_image, blessing)
            
            # 强制刷新日志
            file_handler.flush()
            
            service.stop()
            
            # 读取日志内容
            log_content = temp_log_file.read_text(encoding='utf-8')
            
            # 验证新访客添加日志存在
            assert "Added new visitor" in log_content or visitor.visitor_id in log_content
            
        finally:
            logger.removeHandler(file_handler)
            file_handler.close()
            mock_vc.stop()
            self._cleanup_temp_dir(temp_storage_dir)
            self._cleanup_temp_file(temp_log_file)
    
    def test_property_11_error_logging(self):
        """
        属性 11：日志记录完整性 - 错误日志
        
        对于任何错误情况，系统应该记录错误日志
        
        验证需求：7.5
        Feature: welcome-greeter, Property 11: 日志记录完整性
        """
        # 创建临时目录和文件
        temp_storage_dir = self._create_temp_storage_dir()
        temp_log_file = self._create_temp_log_file()
        
        # 配置日志
        logger = get_logger('src.greeter_service')
        file_handler = logging.FileHandler(temp_log_file, encoding='utf-8')
        file_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        try:
            # 模拟摄像头连接失败的情况
            mock_vc = self._mock_video_capture()
            
            service = GreeterService(
                camera_index=0,
                similarity_threshold=0.7,
                storage_dir=str(temp_storage_dir)
            )
            
            # 清空日志文件
            temp_log_file.write_text('', encoding='utf-8')
            
            # 尝试处理帧（会失败因为返回None）
            result = service.process_frame()
            
            # 强制刷新日志
            file_handler.flush()
            
            service.stop()
            
            # 读取日志内容
            log_content = temp_log_file.read_text(encoding='utf-8')
            
            # 验证警告或错误日志存在
            assert "WARNING" in log_content or "ERROR" in log_content or "Failed" in log_content or "失败" in log_content
            
            mock_vc.stop()
            
        finally:
            logger.removeHandler(file_handler)
            file_handler.close()
            self._cleanup_temp_dir(temp_storage_dir)
            self._cleanup_temp_file(temp_log_file)
    
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_operations=st.integers(min_value=1, max_value=10)
    )
    def test_property_11_multiple_operations_logging(
        self,
        num_operations
    ):
        """
        属性 11：日志记录完整性 - 多操作日志
        
        对于任何数量的操作序列，每个关键操作都应该有对应的日志记录
        
        验证需求：7.5
        Feature: welcome-greeter, Property 11: 日志记录完整性
        """
        # 创建临时目录和文件
        temp_storage_dir = self._create_temp_storage_dir()
        temp_log_file = self._create_temp_log_file()
        mock_vc = self._mock_video_capture()
        
        # 配置日志
        logger = get_logger('src.face_database')
        file_handler = logging.FileHandler(temp_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        try:
            service = GreeterService(
                camera_index=0,
                similarity_threshold=0.7,
                storage_dir=str(temp_storage_dir)
            )
            
            # 清空日志文件
            temp_log_file.write_text('', encoding='utf-8')
            
            # 执行多个操作
            import numpy as np
            for i in range(num_operations):
                features = np.random.rand(128)
                face_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                blessing = f"祝福{i}"
                service.face_database.add_visitor(features, face_image, blessing)
            
            # 强制刷新日志
            file_handler.flush()
            
            service.stop()
            
            # 读取日志内容
            log_content = temp_log_file.read_text(encoding='utf-8')
            
            # 验证日志条目数量合理（至少应该有一些日志）
            log_lines = [line for line in log_content.split('\n') if line.strip()]
            assert len(log_lines) >= num_operations, f"Expected at least {num_operations} log entries, got {len(log_lines)}"
            
        finally:
            logger.removeHandler(file_handler)
            file_handler.close()
            mock_vc.stop()
            self._cleanup_temp_dir(temp_storage_dir)
            self._cleanup_temp_file(temp_log_file)
    
    def test_property_11_stop_logging(self):
        """
        属性 11：日志记录完整性 - 停止服务日志
        
        对于任何服务停止操作，系统应该记录停止日志
        
        验证需求：7.5
        Feature: welcome-greeter, Property 11: 日志记录完整性
        """
        # 创建临时目录和文件
        temp_storage_dir = self._create_temp_storage_dir()
        temp_log_file = self._create_temp_log_file()
        mock_vc = self._mock_video_capture()
        
        # 配置日志
        logger = get_logger('src.greeter_service')
        file_handler = logging.FileHandler(temp_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        try:
            service = GreeterService(
                camera_index=0,
                similarity_threshold=0.7,
                storage_dir=str(temp_storage_dir)
            )
            
            # 清空日志文件
            temp_log_file.write_text('', encoding='utf-8')
            
            # 停止服务
            service.stop()
            
            # 强制刷新日志
            file_handler.flush()
            
            # 读取日志内容
            log_content = temp_log_file.read_text(encoding='utf-8')
            
            # 验证停止日志存在
            assert "Stopping" in log_content or "stopped" in log_content or "停止" in log_content
            
        finally:
            logger.removeHandler(file_handler)
            file_handler.close()
            mock_vc.stop()
            self._cleanup_temp_dir(temp_storage_dir)
            self._cleanup_temp_file(temp_log_file)

