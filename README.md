# 迎宾器系统

基于计算机视觉的实时人脸检测和问候系统，使用Python、FastAPI、OpenCV和MediaPipe构建。

## 功能特点

- 🎥 实时视频流处理
- 👤 人脸检测与识别
- 🎊 个性化中文祝福语
- 📊 访客统计
- 💾 访客数据持久化
- 🌐 Web界面访问

## 系统要求

- Python 3.12+
- 摄像头设备
- Windows/Linux/macOS

## 安装指南

### 1. 克隆项目

```bash
git clone <repository-url>
cd welcome-greeter
```

### 2. 安装依赖

使用uv（推荐）：

```bash
# 安装uv
pip install uv

# 安装项目依赖
uv sync
```

或使用pip：

```bash
pip install -r requirements.txt
```

### 3. 配置

系统使用默认配置，可在`src/config.py`中修改：

- `DEFAULT_CAMERA_INDEX`: 摄像头索引（默认0）
- `SIMILARITY_THRESHOLD`: 人脸相似度阈值（默认0.7）
- `API_HOST`: API服务器地址（默认"0.0.0.0"）
- `API_PORT`: API服务器端口（默认8000）

## 使用方法

### 启动服务

```bash
# 使用uv运行
uv run python main.py

# 或直接运行
python main.py
```

### 访问Web界面

服务器启动后，打开浏览器访问：
- `http://localhost:8000` 或
- `http://127.0.0.1:8000`

**注意**：虽然服务器显示运行在 `http://0.0.0.0:8000`，但浏览器需要使用 `localhost` 或 `127.0.0.1` 访问。

## API文档

### 端点列表

#### GET /
返回Web界面HTML页面

#### GET /video_feed
返回MJPEG视频流

**响应类型**: `multipart/x-mixed-replace`

#### GET /api/statistics
获取访客统计数据

**响应示例**:
```json
{
  "total_visitors": 10,
  "camera_fps": 30.0,
  "camera_resolution": [640, 480]
}
```

#### GET /health
健康检查端点

**响应示例**:
```json
{
  "status": "healthy",
  "camera": "connected",
  "visitors": 10
}
```

## 项目结构

```
welcome-greeter/
├── src/
│   ├── app.py                    # FastAPI应用
│   ├── greeter_service.py        # 核心迎宾服务
│   ├── face_detection.py         # 人脸检测模块
│   ├── face_feature_extractor.py # 特征提取模块
│   ├── face_database.py          # 访客数据库
│   ├── blessing_generator.py     # 祝福语生成器
│   ├── video_capture.py          # 视频采集
│   ├── video_renderer.py         # 视频渲染
│   ├── config.py                 # 配置文件
│   └── logger.py                 # 日志配置
├── tests/                        # 测试文件
├── face_data/                    # 访客数据存储
│   ├── features/                 # 特征向量
│   └── images/                   # 人脸图片
├── logs/                         # 日志文件
├── main.py                       # 应用入口
├── pyproject.toml                # 项目配置
└── README.md                     # 项目文档
```

## 开发指南

### 运行测试

```bash
# 运行所有测试
uv run pytest tests/ -v

# 运行特定测试文件
uv run pytest tests/test_face_detection.py -v

# 运行带覆盖率的测试
uv run pytest tests/ --cov=src --cov-report=html
```

### 代码质量

项目包含完整的测试套件：
- 单元测试
- 集成测试
- 属性测试（Property-Based Testing）
- 错误处理测试

## 技术栈

- **Web框架**: FastAPI
- **计算机视觉**: OpenCV, MediaPipe
- **数值计算**: NumPy
- **图像处理**: Pillow
- **测试框架**: pytest, Hypothesis
- **包管理**: uv

## 故障排除

### 无法访问Web界面

如果浏览器显示"无法访问此页面"：
1. 确认服务器已启动（终端显示"Uvicorn running on..."）
2. 使用 `http://localhost:8000` 或 `http://127.0.0.1:8000` 访问
3. **不要**直接访问 `http://0.0.0.0:8000`（这是服务器监听地址，不是访问地址）

### 摄像头无法打开

1. 检查摄像头是否被其他应用占用
2. 尝试更改`DEFAULT_CAMERA_INDEX`配置
3. 确认摄像头驱动正常安装

### 中文显示乱码

系统会自动查找系统中的中文字体，如果显示异常：
- Windows: 确保有`msyh.ttc`（微软雅黑）
- Linux: 安装中文字体包
- macOS: 系统自带中文字体

### 人脸识别不准确

调整`SIMILARITY_THRESHOLD`参数：
- 提高阈值（如0.8）：更严格的匹配
- 降低阈值（如0.6）：更宽松的匹配

## 许可证

[添加许可证信息]

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

[添加联系方式]
