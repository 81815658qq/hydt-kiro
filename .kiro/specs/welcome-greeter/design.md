# 设计文档

## 概述

迎宾器系统是一个基于Python的实时人脸检测和问候应用。系统使用OpenCV进行视频采集，MediaPipe进行人脸检测，并通过FastAPI提供Web服务。核心功能包括：多人脸实时检测、吉祥祝福语显示、人脸特征提取与识别、访客统计。

技术栈：
- Python 3.12
- FastAPI：Web框架
- OpenCV：视频采集和图像处理
- MediaPipe：人脸检测
- NumPy：数值计算和特征处理
- Pillow：中文字体渲染

## 架构

系统采用分层架构，主要包含以下层次：

```
┌─────────────────────────────────────┐
│      FastAPI Web Server Layer       │
│  (视频流端点、统计端点、静态页面)    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     Video Processing Layer          │
│  (视频采集、帧处理、渲染)            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    Face Detection & Recognition     │
│  (MediaPipe检测、特征提取、匹配)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Data Persistence Layer         │
│  (人脸数据库、特征存储、图片存储)    │
└─────────────────────────────────────┘
```

### 数据流

1. 摄像头 → VideoCapture → 原始帧
2. 原始帧 → FaceDetector → 人脸位置列表
3. 人脸位置 → FaceFeatureExtractor → 特征向量
4. 特征向量 → FaceDatabase → 匹配结果（新/已知访客）
5. 匹配结果 → BlessingGenerator → 祝福语
6. 原始帧 + 人脸位置 + 祝福语 → VideoRenderer → 渲染后的帧
7. 渲染后的帧 → FastAPI → MJPEG视频流

## 组件和接口

### 1. VideoCapture（视频采集器）

负责从USB摄像头采集视频流。

```python
class VideoCapture:
    def __init__(self, camera_index: int = 0):
        """初始化摄像头连接"""
        
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧图像，返回BGR格式的numpy数组"""
        
    def is_opened(self) -> bool:
        """检查摄像头是否正常打开"""
        
    def release(self):
        """释放摄像头资源"""
        
    def get_fps(self) -> float:
        """获取摄像头帧率"""
```

### 2. FaceDetector（人脸检测器）

使用MediaPipe检测视频帧中的所有人脸。

```python
class FaceDetection:
    x: float  # 边界框左上角x坐标（归一化0-1）
    y: float  # 边界框左上角y坐标（归一化0-1）
    width: float  # 边界框宽度（归一化0-1）
    height: float  # 边界框高度（归一化0-1）
    landmarks: List[Tuple[float, float]]  # 面部关键点

class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        """初始化MediaPipe人脸检测模型"""
        
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """检测帧中的所有人脸，返回人脸检测结果列表"""
        
    def extract_face_region(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """从帧中提取人脸区域图像"""
```

### 3. FaceFeatureExtractor（人脸特征提取器）

从人脸区域提取特征向量用于识别。

```python
class FaceFeatureExtractor:
    def __init__(self):
        """初始化特征提取器"""
        
    def extract_features(self, face_image: np.ndarray) -> np.ndarray:
        """从人脸图像提取特征向量（128维或更高）"""
        
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算两个特征向量的相似度（0-1之间，1表示完全相同）"""
```

实现策略：使用简化的特征提取方法
- 将人脸图像调整为固定大小（如64x64）
- 转换为灰度图
- 计算直方图或使用简单的CNN特征
- 使用余弦相似度进行匹配

### 4. FaceDatabase（人脸数据库）

存储和管理已识别的访客人脸数据。

```python
class VisitorRecord:
    visitor_id: str  # 唯一访客ID
    features: np.ndarray  # 特征向量
    face_image_path: str  # 人脸图片保存路径
    first_seen: datetime  # 首次出现时间
    blessing: str  # 分配的祝福语

class FaceDatabase:
    def __init__(self, storage_dir: str = "./face_data"):
        """初始化数据库，指定存储目录"""
        
    def find_matching_visitor(self, features: np.ndarray, threshold: float = 0.7) -> Optional[VisitorRecord]:
        """查找匹配的访客记录，相似度超过阈值则返回"""
        
    def add_visitor(self, features: np.ndarray, face_image: np.ndarray, blessing: str) -> VisitorRecord:
        """添加新访客记录，保存特征和人脸图片"""
        
    def get_total_visitors(self) -> int:
        """获取总访客数量"""
        
    def load_from_disk(self):
        """从磁盘加载已保存的访客数据"""
        
    def save_to_disk(self):
        """将访客数据保存到磁盘"""
```

存储格式：
- 特征向量：保存为numpy的.npy文件
- 人脸图片：保存为JPEG格式
- 元数据：保存为JSON文件

### 5. BlessingGenerator（祝福语生成器）

为访客分配吉祥祝福语。

```python
class BlessingGenerator:
    BLESSINGS = [
        "鸿运当头", "好运常在", "福星高照", "吉祥如意",
        "万事如意", "心想事成", "步步高升", "财源广进",
        "喜气洋洋", "笑口常开", "福寿安康", "事业有成",
        "前程似锦", "大吉大利", "五福临门","学业进步",
        "得偿所愿","鲲鹏之志","一举夺魁","得偿所愿"
    ]
    
    def __init__(self):
        """初始化祝福语生成器"""
        
    def get_blessing_for_visitor(self, visitor_id: str) -> str:
        """为访客ID分配一个固定的祝福语（基于哈希确保一致性）"""
        
    def get_random_blessing(self) -> str:
        """获取随机祝福语"""
```

### 6. VideoRenderer（视频渲染器）

在视频帧上绘制人脸框、祝福语和统计信息。

```python
class VideoRenderer:
    def __init__(self, font_path: str = None):
        """初始化渲染器，加载中文字体"""
        
    def render_frame(
        self,
        frame: np.ndarray,
        detections: List[FaceDetection],
        blessings: List[str],
        total_visitors: int
    ) -> np.ndarray:
        """在帧上渲染所有信息，返回渲染后的帧"""
        
    def draw_blessing_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 0)
    ):
        """在指定位置绘制中文祝福语"""
        
    def draw_face_box(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        color: Tuple[int, int, int] = (0, 255, 0)
    ):
        """绘制人脸边界框"""
```

### 7. GreeterService（迎宾服务）

核心业务逻辑，协调各组件完成迎宾功能。

```python
class GreeterService:
    def __init__(
        self,
        camera_index: int = 0,
        similarity_threshold: float = 0.7
    ):
        """初始化迎宾服务，创建所有组件"""
        
    def process_frame(self) -> Optional[np.ndarray]:
        """处理一帧：检测人脸、识别访客、渲染祝福语"""
        
    def get_statistics(self) -> dict:
        """获取统计信息（总访客数等）"""
        
    def start(self):
        """启动服务"""
        
    def stop(self):
        """停止服务并释放资源"""
```

处理流程：
1. 从摄像头读取帧
2. 使用FaceDetector检测所有人脸
3. 对每个检测到的人脸：
   - 提取人脸区域
   - 提取特征向量
   - 在FaceDatabase中查找匹配
   - 如果是新访客，保存数据并分配祝福语
   - 如果是已知访客，使用已有祝福语
4. 使用VideoRenderer渲染帧
5. 返回渲染后的帧

### 8. FastAPI Application（Web服务器）

提供HTTP接口访问视频流和统计数据。

```python
app = FastAPI()

@app.get("/")
async def index():
    """返回HTML页面显示视频流"""
    
@app.get("/video_feed")
async def video_feed():
    """返回MJPEG视频流"""
    
@app.get("/api/statistics")
async def get_statistics():
    """返回JSON格式的统计数据"""
    
@app.on_event("startup")
async def startup_event():
    """启动时初始化GreeterService"""
    
@app.on_event("shutdown")
async def shutdown_event():
    """关闭时释放资源"""
```

## 数据模型

### FaceDetection（人脸检测结果）

```python
@dataclass
class FaceDetection:
    x: float  # 归一化坐标 [0, 1]
    y: float
    width: float
    height: float
    landmarks: List[Tuple[float, float]]  # 面部关键点
    
    def to_pixel_coords(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """转换为像素坐标"""
        return (
            int(self.x * frame_width),
            int(self.y * frame_height),
            int(self.width * frame_width),
            int(self.height * frame_height)
        )
```

### VisitorRecord（访客记录）

```python
@dataclass
class VisitorRecord:
    visitor_id: str  # UUID格式
    features: np.ndarray  # 特征向量
    face_image_path: str  # 相对路径
    first_seen: datetime
    blessing: str  # 四字祝福语
    
    def to_dict(self) -> dict:
        """序列化为字典（用于JSON存储）"""
        
    @classmethod
    def from_dict(cls, data: dict) -> 'VisitorRecord':
        """从字典反序列化"""
```

### 文件系统结构

```
face_data/
├── metadata.json          # 所有访客的元数据
├── features/              # 特征向量目录
│   ├── {visitor_id}.npy
│   └── ...
└── images/                # 人脸图片目录
    ├── {visitor_id}.jpg
    └── ...
```

metadata.json格式：
```json
{
  "visitors": [
    {
      "visitor_id": "uuid-string",
      "face_image_path": "images/uuid-string.jpg",
      "first_seen": "2026-01-14T10:30:00",
      "blessing": "鸿运当头"
    }
  ],
  "total_count": 1
}
```

## 正确性属性

*属性是关于系统行为的形式化陈述，应该在所有有效执行中保持为真。属性是人类可读规范和机器可验证正确性保证之间的桥梁。*


### 属性 1：人脸检测完整性
*对于任何*检测到的人脸，检测结果应该包含有效的边界框坐标（归一化到0-1范围）和面部关键点列表。
**验证需求：2.2, 2.4**

### 属性 2：多人脸检测能力
*对于任何*包含多个人脸的视频帧，人脸检测器应该返回所有可见人脸的检测结果，检测数量应该大于等于实际人脸数量。
**验证需求：2.1, 2.3**

### 属性 3：祝福语分配一致性
*对于任何*访客ID，多次调用祝福语生成器应该返回相同的祝福语（幂等性）。
**验证需求：3.5**

### 属性 4：祝福语渲染完整性
*对于任何*视频帧和人脸检测结果列表，渲染后的帧应该在每个人脸上方包含对应的祝福语文本。
**验证需求：3.1, 3.3**

### 属性 5：特征提取一致性
*对于任何*有效的人脸图像，特征提取器应该返回固定维度的特征向量（非空且维度一致）。
**验证需求：4.1**

### 属性 6：相似度阈值判断正确性
*对于任何*两个特征向量，当相似度超过阈值时应该判断为匹配，低于阈值时应该判断为不匹配。
**验证需求：4.3, 4.4**

### 属性 7：新访客数据持久化
*对于任何*新访客，添加到数据库后应该能够通过特征匹配找回，且保存的人脸图片和特征数据应该存在于文件系统中。
**验证需求：4.5**

### 属性 8：访客计数唯一性
*对于任何*访客序列（包含新访客和重复访客），系统的总访客计数应该等于唯一访客的数量。
**验证需求：5.1, 5.2**

### 属性 9：数据持久化往返一致性
*对于任何*访客记录集合，保存到磁盘后再加载应该得到等价的数据（特征向量、元数据、图片路径）。
**验证需求：5.4**

### 属性 10：统计信息显示完整性
*对于任何*渲染的视频帧，应该包含当前总访客数量的文本显示。
**验证需求：5.3**

### 属性 11：日志记录完整性
*对于任何*关键操作（新访客添加、错误发生），系统应该在日志中记录相应的条目。
**验证需求：7.5**

## 错误处理

### 摄像头错误
- **连接失败**：系统启动时检测摄像头，如果无法打开则抛出`CameraConnectionError`异常，包含详细错误信息
- **读取失败**：视频帧读取失败时记录警告日志，跳过该帧继续处理
- **断开连接**：检测到摄像头断开时，尝试重新连接（最多3次），失败后抛出异常

### 人脸检测错误
- **检测失败**：MediaPipe检测异常时捕获异常，记录错误日志，返回空列表，继续处理下一帧
- **无效检测结果**：过滤掉边界框坐标无效的检测结果

### 特征提取错误
- **提取失败**：特征提取异常时记录错误，跳过该人脸，不影响其他人脸处理
- **图像无效**：人脸区域图像无效时返回None，调用方应检查并处理

### 数据库错误
- **文件系统错误**：保存/加载文件失败时抛出`DatabaseError`异常
- **数据损坏**：加载时发现数据格式错误，记录错误并跳过该记录
- **磁盘空间不足**：保存失败时抛出异常，提示用户清理空间

### API错误
- **服务未启动**：访问API时如果服务未初始化，返回503状态码
- **摄像头不可用**：视频流端点在摄像头不可用时返回错误页面
- **内部错误**：捕获所有未处理异常，返回500状态码和错误信息

### 错误恢复策略
1. **优雅降级**：单个组件失败不影响整体系统运行
2. **重试机制**：临时性错误（如网络、IO）自动重试
3. **日志记录**：所有错误都记录到日志文件，便于调试
4. **用户反馈**：关键错误通过API返回明确的错误信息

## 测试策略

### 单元测试
单元测试用于验证特定示例、边缘情况和错误条件：

1. **VideoCapture测试**
   - 测试摄像头连接成功场景
   - 测试摄像头不存在时的错误处理
   - 测试帧率获取功能

2. **FaceDetector测试**
   - 使用包含1个、2个、5个人脸的测试图像
   - 测试无人脸图像返回空列表
   - 测试边界框坐标在有效范围内

3. **FaceFeatureExtractor测试**
   - 测试相同人脸图像的特征一致性
   - 测试不同人脸的特征差异
   - 测试相似度计算范围在[0, 1]

4. **FaceDatabase测试**
   - 测试添加新访客
   - 测试查找已存在访客
   - 测试保存和加载功能
   - 测试空数据库场景

5. **BlessingGenerator测试**
   - 验证祝福语列表包含至少10个四字词语
   - 测试相同访客ID返回相同祝福语
   - 测试不同访客ID可能返回不同祝福语

6. **VideoRenderer测试**
   - 测试中文文本渲染不抛出异常
   - 测试边界框绘制
   - 测试统计信息显示

7. **GreeterService集成测试**
   - 测试完整处理流程（使用测试图像）
   - 测试新访客识别和保存
   - 测试已知访客识别
   - 测试统计数据正确性

8. **FastAPI端点测试**
   - 测试根路径返回HTML
   - 测试统计端点返回JSON
   - 测试视频流端点返回正确的Content-Type

### 基于属性的测试
基于属性的测试用于验证跨所有输入的通用属性。使用`hypothesis`库进行属性测试，每个测试至少运行100次迭代。

**测试配置**：
```python
from hypothesis import given, settings
import hypothesis.strategies as st

@settings(max_examples=100)
```

**属性测试列表**：

1. **属性测试 1：人脸检测完整性**
   - 生成随机检测结果
   - 验证边界框坐标在[0, 1]范围内
   - 验证关键点列表非空
   - **标签：Feature: welcome-greeter, Property 1: 人脸检测完整性**

2. **属性测试 2：多人脸检测能力**
   - 使用多个测试图像（包含不同数量人脸）
   - 验证检测数量合理性
   - **标签：Feature: welcome-greeter, Property 2: 多人脸检测能力**

3. **属性测试 3：祝福语分配一致性**
   - 生成随机访客ID
   - 多次调用祝福语生成器
   - 验证返回值始终相同
   - **标签：Feature: welcome-greeter, Property 3: 祝福语分配一致性**

4. **属性测试 4：祝福语渲染完整性**
   - 生成随机帧和检测结果
   - 渲染后验证帧包含祝福语文本（通过OCR或像素变化）
   - **标签：Feature: welcome-greeter, Property 4: 祝福语渲染完整性**

5. **属性测试 5：特征提取一致性**
   - 生成随机人脸图像
   - 验证特征向量维度固定
   - 验证特征向量非空
   - **标签：Feature: welcome-greeter, Property 5: 特征提取一致性**

6. **属性测试 6：相似度阈值判断正确性**
   - 生成随机特征向量对和阈值
   - 计算相似度
   - 验证判断逻辑正确（>= 阈值为匹配，< 阈值为不匹配）
   - **标签：Feature: welcome-greeter, Property 6: 相似度阈值判断正确性**

7. **属性测试 7：新访客数据持久化**
   - 生成随机访客数据
   - 添加到数据库
   - 验证能通过特征匹配找回
   - 验证文件存在
   - **标签：Feature: welcome-greeter, Property 7: 新访客数据持久化**

8. **属性测试 8：访客计数唯一性**
   - 生成随机访客序列（包含重复）
   - 逐个添加到数据库
   - 验证计数等于唯一访客数
   - **标签：Feature: welcome-greeter, Property 8: 访客计数唯一性**

9. **属性测试 9：数据持久化往返一致性**
   - 生成随机访客记录集合
   - 保存到临时目录
   - 加载并比较
   - 验证数据等价
   - **标签：Feature: welcome-greeter, Property 9: 数据持久化往返一致性**

10. **属性测试 10：统计信息显示完整性**
    - 生成随机帧和访客数量
    - 渲染统计信息
    - 验证帧包含数字文本
    - **标签：Feature: welcome-greeter, Property 10: 统计信息显示完整性**

11. **属性测试 11：日志记录完整性**
    - 执行随机操作序列
    - 验证日志文件包含对应条目
    - **标签：Feature: welcome-greeter, Property 11: 日志记录完整性**

### 测试数据
- 使用公开的人脸测试数据集（如LFW的部分图像）
- 创建包含0-5个人脸的合成测试图像
- 使用不同光照、角度、遮挡的测试场景

### 测试覆盖率目标
- 单元测试代码覆盖率：>80%
- 属性测试覆盖所有核心业务逻辑
- 集成测试覆盖主要用户场景

### 持续集成
- 每次提交运行所有单元测试
- 每日运行完整测试套件（包括属性测试）
- 使用GitHub Actions或类似CI工具
