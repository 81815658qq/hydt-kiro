# 迎宾器系统

基于计算机视觉的实时人脸检测和问候系统。

## 功能特性

- 🎥 实时视频采集和处理
- 👤 多人脸同时检测
- 🎊 吉祥祝福语显示
- 🔍 人脸识别和访客统计
- 🌐 Web API 服务
- 📊 访客数据持久化

## 技术栈

- Python 3.12
- FastAPI - Web框架
- OpenCV - 视频采集和图像处理
- MediaPipe - 人脸检测
- NumPy - 数值计算
- Pillow - 中文字体渲染

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── config.py          # 配置文件
│   └── logger.py          # 日志配置
├── tests/                 # 测试目录
│   ├── __init__.py
│   └── conftest.py        # pytest配置
├── face_data/             # 人脸数据存储（自动创建）
├── logs/                  # 日志文件（自动创建）
├── main.py               # 应用入口
├── pyproject.toml        # 项目配置
├── pytest.ini            # pytest配置
└── README.md             # 项目说明
```

## 安装

1. 确保已安装 Python 3.12+

2. 安装依赖：
```bash
pip install -e .
```

## 运行

```bash
python main.py
```

## 测试

运行所有测试：
```bash
pytest
```

运行单元测试：
```bash
pytest -m unit
```

运行属性测试：
```bash
pytest -m property
```

查看测试覆盖率：
```bash
pytest --cov=src --cov-report=html
```

## 开发

本项目遵循规范化的开发流程，使用基于属性的测试（Property-Based Testing）确保代码质量。

## 许可证

MIT License
