"""视频渲染器模块

负责在视频帧上绘制人脸框、祝福语和统计信息。
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from pathlib import Path

from src.face_detection import FaceDetection


class VideoRenderer:
    """视频渲染器，在视频帧上绘制人脸框、祝福语和统计信息"""
    
    def __init__(self, font_path: Optional[str] = None, font_size: int = 32):
        """初始化渲染器，加载中文字体
        
        Args:
            font_path: 中文字体文件路径，如果为None则尝试使用系统默认字体
            font_size: 字体大小
        """
        self.font_size = font_size
        self.font = self._load_font(font_path, font_size)
        self.stats_font = self._load_font(font_path, font_size // 2)
        
    def _load_font(self, font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
        """加载字体
        
        Args:
            font_path: 字体文件路径
            size: 字体大小
            
        Returns:
            PIL字体对象
        """
        if font_path and Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
        
        # 尝试常见的中文字体路径
        common_fonts = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",  # Windows 宋体
        ]
        
        for font in common_fonts:
            if Path(font).exists():
                return ImageFont.truetype(font, size)
        
        # 如果都找不到，使用默认字体
        return ImageFont.load_default()
    
    def render_frame(
        self,
        frame: np.ndarray,
        detections: List[FaceDetection],
        blessings: List[str],
        total_visitors: int
    ) -> np.ndarray:
        """在帧上渲染所有信息，返回渲染后的帧
        
        Args:
            frame: 原始视频帧（BGR格式）
            detections: 人脸检测结果列表
            blessings: 对应每个人脸的祝福语列表
            total_visitors: 总访客数量
            
        Returns:
            渲染后的帧（BGR格式）
        """
        # 复制帧以避免修改原始数据
        rendered_frame = frame.copy()
        
        # 绘制每个人脸的边界框和祝福语
        for detection, blessing in zip(detections, blessings):
            self.draw_face_box(rendered_frame, detection)
            
            # 计算祝福语位置（人脸上方）
            h, w = frame.shape[:2]
            x, y, width, height = detection.to_pixel_coords(w, h)
            text_position = (x, max(0, y - 40))
            
            self.draw_blessing_text(rendered_frame, blessing, text_position)
        
        # 不再绘制统计信息（已在网页下方显示）
        
        return rendered_frame
    
    def draw_blessing_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 0)
    ):
        """在指定位置绘制中文祝福语
        
        Args:
            frame: 视频帧（BGR格式）
            text: 祝福语文本
            position: 文本位置 (x, y)
            color: 文本颜色 (B, G, R)
        """
        # 将BGR转换为RGB用于PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制文本（PIL使用RGB颜色）
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=self.font, fill=rgb_color)
        
        # 转换回BGR并更新原始帧
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        frame[:] = frame_bgr
    
    def draw_face_box(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ):
        """绘制人脸边界框
        
        Args:
            frame: 视频帧（BGR格式）
            detection: 人脸检测结果
            color: 边界框颜色 (B, G, R)
            thickness: 线条粗细
        """
        h, w = frame.shape[:2]
        x, y, width, height = detection.to_pixel_coords(w, h)
        
        # 绘制矩形
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            color,
            thickness
        )
    
    def _draw_statistics(
        self,
        frame: np.ndarray,
        total_visitors: int,
        position: Optional[Tuple[int, int]] = None
    ):
        """绘制统计信息
        
        Args:
            frame: 视频帧（BGR格式）
            total_visitors: 总访客数量
            position: 文本位置，如果为None则使用默认位置（左上角）
        """
        if position is None:
            position = (10, 30)
        
        stats_text = f"学生总数: {total_visitors}"
        
        # 使用PIL绘制中文统计信息
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制文本（白色）
        draw.text(position, stats_text, font=self.stats_font, fill=(255, 255, 255))
        
        # 转换回BGR并更新原始帧
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        frame[:] = frame_bgr
