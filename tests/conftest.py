"""pytest 配置和共享fixtures"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """创建临时目录用于测试"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # 清理
    if temp_path.exists():
        shutil.rmtree(temp_path)

@pytest.fixture
def sample_image():
    """创建一个简单的测试图像（640x480 BGR格式）"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def sample_face_image():
    """创建一个简单的人脸区域图像（64x64 BGR格式）"""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
