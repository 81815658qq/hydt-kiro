"""FastAPI端点测试

测试FastAPI应用的各个端点功能。
需求：6.2, 6.3, 6.4
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np


def test_index_returns_html():
    """测试根路径返回HTML
    
    需求：6.2 - API服务器应提供视频流端点
    """
    # 创建模拟的GreeterService
    mock_service = Mock()
    mock_service.start.return_value = None
    mock_service.stop.return_value = None
    
    # 在导入前patch
    with patch('src.greeter_service.GreeterService', return_value=mock_service):
        # 导入app
        from main import app
        
        # 使用TestClient，它会触发lifespan
        with TestClient(app) as client:
            response = client.get("/")
            
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            assert "迎宾器系统" in response.text
            assert "video-stream" in response.text


def test_statistics_endpoint_returns_json():
    """测试统计端点返回JSON
    
    需求：6.3 - API服务器应提供统计数据端点
    """
    # 创建模拟的GreeterService
    mock_service = Mock()
    mock_service.get_statistics.return_value = {
        "total_visitors": 5,
        "camera_fps": 30.0,
        "camera_resolution": (640, 480)
    }
    mock_service.start.return_value = None
    mock_service.stop.return_value = None
    
    with patch('src.greeter_service.GreeterService', return_value=mock_service):
        from main import app
        import main as main_module
        
        with TestClient(app) as client:
            # 手动设置 greeter_service 为 mock
            main_module.greeter_service = mock_service
            
            response = client.get("/api/statistics")
            
            assert response.status_code == 200
            assert "application/json" in response.headers["content-type"]
            
            data = response.json()
            assert "total_visitors" in data
            assert "camera_fps" in data
            assert "camera_resolution" in data
            assert data["total_visitors"] == 5


def test_video_feed_returns_correct_content_type():
    """测试视频流端点返回正确的Content-Type
    
    需求：6.4 - 客户端请求视频流时，API服务器应以MJPEG格式流式传输
    """
    # 创建模拟的GreeterService
    mock_service = Mock()
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_service.process_frame.return_value = mock_frame
    mock_service.start.return_value = None
    mock_service.stop.return_value = None
    
    with patch('src.greeter_service.GreeterService', return_value=mock_service):
        from main import app
        
        with TestClient(app) as client:
            response = client.get("/video_feed")
            
            assert response.status_code == 200
            assert "multipart/x-mixed-replace" in response.headers["content-type"]


def test_statistics_without_service():
    """测试服务未初始化时统计端点返回503"""
    # Mock GreeterService抛出异常
    with patch('src.greeter_service.GreeterService', side_effect=Exception("Camera not available")):
        from main import app
        import main as main_module
        
        with TestClient(app) as client:
            # 手动设置 greeter_service 为 None
            main_module.greeter_service = None
            
            response = client.get("/api/statistics")
            
            assert response.status_code == 503
            data = response.json()
            assert "error" in data


def test_video_feed_without_service():
    """测试服务未初始化时视频流端点返回503"""
    # Mock GreeterService抛出异常
    with patch('src.greeter_service.GreeterService', side_effect=Exception("Camera not available")):
        from main import app
        import main as main_module
        
        with TestClient(app) as client:
            # 手动设置 greeter_service 为 None
            main_module.greeter_service = None
            
            response = client.get("/video_feed")
            
            assert response.status_code == 503
            data = response.json()
            assert "error" in data
