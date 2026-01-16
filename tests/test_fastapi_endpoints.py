"""FastAPI端点测试"""
import pytest
from unittest.mock import Mock, patch
import numpy as np


def test_simple():
    """简单测试"""
    assert True


def test_index_endpoint():
    """测试根路径端点"""
    # 创建模拟的GreeterService
    mock_service = Mock()
    mock_service.start.return_value = None
    mock_service.stop.return_value = None
    
    with patch('src.greeter_service.GreeterService', return_value=mock_service):
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "迎宾器系统" in response.text


def test_statistics_endpoint():
    """测试统计端点"""
    # 创建模拟的GreeterService
    mock_service = Mock()
    mock_service.get_statistics.return_value = {
        "total_visitors": 5,
        "camera_fps": 30.0,
        "camera_resolution": (640, 480)
    }
    mock_service.start.return_value = None
    mock_service.stop.return_value = None
    
    # 需要patch main模块中的全局变量
    with patch('main.greeter_service', mock_service):
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        response = client.get("/api/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_visitors"] == 5


def test_video_feed_endpoint():
    """测试视频流端点"""
    # 创建模拟的GreeterService
    mock_service = Mock()
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_service.process_frame.return_value = mock_frame
    mock_service.start.return_value = None
    mock_service.stop.return_value = None
    
    with patch('src.greeter_service.GreeterService', return_value=mock_service):
        with patch('main.greeter_service', mock_service):
            from fastapi.testclient import TestClient
            from main import app
            
            client = TestClient(app)
            response = client.get("/video_feed")
            
            assert response.status_code == 200
            assert "multipart" in response.headers["content-type"]
