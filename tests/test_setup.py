"""测试项目基础设施是否正确设置"""
import pytest
from pathlib import Path
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_project_structure():
    """测试项目目录结构是否存在"""
    project_root = Path(__file__).parent.parent
    
    assert (project_root / "src").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "pytest.ini").exists()
    assert (project_root / "README.md").exists()

def test_config_import():
    """测试配置模块是否可以导入"""
    from src import config
    
    assert hasattr(config, 'PROJECT_ROOT')
    assert hasattr(config, 'FACE_DATA_DIR')
    assert hasattr(config, 'LOG_DIR')
    assert hasattr(config, 'DEFAULT_CAMERA_INDEX')
    assert hasattr(config, 'SIMILARITY_THRESHOLD')

def test_logger_import():
    """测试日志模块是否可以导入"""
    from src.logger import setup_logger, logger
    
    assert logger is not None
    assert callable(setup_logger)

def test_directories_created():
    """测试必要的目录是否被创建"""
    from src.config import FACE_DATA_DIR, LOG_DIR
    
    assert FACE_DATA_DIR.exists()
    assert LOG_DIR.exists()

@pytest.mark.unit
def test_sample_fixtures(sample_image, sample_face_image, temp_dir):
    """测试共享fixtures是否正常工作"""
    import numpy as np
    
    # 测试sample_image
    assert isinstance(sample_image, np.ndarray)
    assert sample_image.shape == (480, 640, 3)
    
    # 测试sample_face_image
    assert isinstance(sample_face_image, np.ndarray)
    assert sample_face_image.shape == (64, 64, 3)
    
    # 测试temp_dir
    assert temp_dir.exists()
    assert temp_dir.is_dir()
