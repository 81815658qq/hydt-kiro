"""配置文件"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据存储目录
FACE_DATA_DIR = PROJECT_ROOT / "face_data"
FACE_DATA_DIR.mkdir(exist_ok=True)

# 日志配置
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "greeter.log"

# 摄像头配置
DEFAULT_CAMERA_INDEX = 0
MIN_FPS = 15

# 人脸检测配置
MIN_DETECTION_CONFIDENCE = 0.5

# 人脸识别配置
SIMILARITY_THRESHOLD = 0.7
FACE_IMAGE_SIZE = (64, 64)

# API配置
API_HOST = "0.0.0.0"
API_PORT = 8000

# 字体配置（用于中文渲染）
FONT_PATH = None  # 如果为None，系统会尝试查找系统字体
