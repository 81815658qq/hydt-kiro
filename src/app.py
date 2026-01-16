"""FastAPI Web应用 - 迎宾器系统

提供HTTP接口访问视频流和统计数据。
"""

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Optional

from src.greeter_service import GreeterService
from src.config import DEFAULT_CAMERA_INDEX, SIMILARITY_THRESHOLD, API_HOST, API_PORT
from src.logger import get_logger

logger = get_logger(__name__)

# 全局GreeterService实例
greeter_service: Optional[GreeterService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global greeter_service
    
    # 启动时初始化GreeterService
    logger.info("Starting FastAPI application...")
    try:
        greeter_service = GreeterService(
            camera_index=DEFAULT_CAMERA_INDEX,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        greeter_service.start()
        logger.info("GreeterService initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GreeterService: {e}")
        greeter_service = None
    
    yield
    
    # 关闭时释放资源
    logger.info("Shutting down FastAPI application...")
    if greeter_service:
        greeter_service.stop()
    logger.info("FastAPI application shut down")


# 创建FastAPI应用实例
app = FastAPI(
    title="迎宾器系统",
    description="基于计算机视觉的实时人脸检测和问候系统",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        if not greeter_service:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "GreeterService not initialized"
                }
            )
        
        # 检查摄像头是否正常
        camera_ok = greeter_service.video_capture.is_opened()
        
        if camera_ok:
            return JSONResponse(
                content={
                    "status": "healthy",
                    "camera": "connected",
                    "visitors": greeter_service.face_database.get_total_visitors()
                }
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "camera": "disconnected",
                    "message": "摄像头未连接"
                }
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回HTML页面显示视频流"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>迎宾器系统</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                width: 100%;
            }
            
            header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            
            h1 {
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .subtitle {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .video-container {
                background: white;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                margin-bottom: 20px;
            }
            
            #video-stream {
                width: 100%;
                border-radius: 10px;
                display: block;
            }
            
            .stats-container {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
            }
            
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            
            .stat-label {
                font-size: 0.9em;
                opacity: 0.9;
                margin-bottom: 10px;
            }
            
            .stat-value {
                font-size: 2.5em;
                font-weight: bold;
            }
            
            .loading {
                text-align: center;
                color: white;
                font-size: 1.2em;
                padding: 40px;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .loading::after {
                content: '...';
                animation: pulse 1.5s infinite;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎊 迎宾器系统 🎊</h1>
                <p class="subtitle">欢迎光临！祝您好运连连！</p>
            </header>
            
            <div class="video-container">
                <img id="video-stream" src="/video_feed" alt="视频流加载中...">
            </div>
            
            <div class="stats-container">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">总访客数</div>
                        <div class="stat-value" id="total-visitors">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">摄像头帧率</div>
                        <div class="stat-value" id="camera-fps">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">视频分辨率</div>
                        <div class="stat-value" id="camera-resolution">-</div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // 定期更新统计信息
            async function updateStatistics() {
                try {
                    const response = await fetch('/api/statistics');
                    const data = await response.json();
                    
                    document.getElementById('total-visitors').textContent = data.total_visitors;
                    document.getElementById('camera-fps').textContent = data.camera_fps.toFixed(1);
                    document.getElementById('camera-resolution').textContent = 
                        data.camera_resolution[0] + 'x' + data.camera_resolution[1];
                } catch (error) {
                    console.error('Failed to fetch statistics:', error);
                }
            }
            
            // 每2秒更新一次统计信息
            updateStatistics();
            setInterval(updateStatistics, 2000);
            
            // 处理视频流加载错误
            document.getElementById('video-stream').onerror = function() {
                this.alt = '视频流加载失败，请检查摄像头连接';
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def generate_frames():
    """生成视频帧的生成器函数"""
    global greeter_service
    
    if not greeter_service:
        logger.error("GreeterService not initialized")
        return
    
    consecutive_failures = 0
    max_consecutive_failures = 10
    
    while True:
        try:
            # 处理一帧
            frame = greeter_service.process_frame()
            
            if frame is None:
                consecutive_failures += 1
                logger.warning(f"Failed to process frame (consecutive failures: {consecutive_failures})")
                
                # 如果连续失败次数过多，停止生成
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures ({consecutive_failures}), stopping frame generation")
                    break
                
                continue
            
            # 成功处理帧，重置失败计数
            consecutive_failures = 0
            
            # 将帧编码为JPEG
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if not ret:
                    logger.warning("Failed to encode frame")
                    continue
                
                # 转换为字节流
                frame_bytes = buffer.tobytes()
                
                # 生成MJPEG格式的帧
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
            except Exception as e:
                logger.error(f"Error encoding frame: {e}")
                consecutive_failures += 1
                continue
                   
        except Exception as e:
            logger.error(f"Unexpected error generating frame: {e}")
            consecutive_failures += 1
            
            if consecutive_failures >= max_consecutive_failures:
                logger.error("Too many consecutive failures, stopping frame generation")
                break


@app.get("/video_feed")
async def video_feed():
    """返回MJPEG视频流"""
    try:
        if not greeter_service:
            logger.error("Video feed requested but GreeterService not initialized")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service Unavailable",
                    "message": "迎宾服务未初始化，请检查摄像头连接"
                }
            )
        
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        logger.error(f"Error in video_feed endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": f"视频流生成失败: {str(e)}"
            }
        )


@app.get("/api/statistics")
async def get_statistics():
    """返回JSON格式的统计数据"""
    try:
        if not greeter_service:
            logger.error("Statistics requested but GreeterService not initialized")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service Unavailable",
                    "message": "迎宾服务未初始化"
                }
            )
        
        stats = greeter_service.get_statistics()
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Error in statistics endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": f"获取统计数据失败: {str(e)}"
            }
        )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
