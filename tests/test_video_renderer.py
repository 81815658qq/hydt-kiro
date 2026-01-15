"""VideoRenderer单元测试"""

import pytest
import numpy as np
import cv2
from src.video_renderer import VideoRenderer
from src.face_detection import FaceDetection


class TestVideoRenderer:
    """VideoRenderer类的单元测试"""
    
    @pytest.fixture
    def renderer(self):
        """创建VideoRenderer实例"""
        return VideoRenderer()
    
    @pytest.fixture
    def test_frame(self):
        """创建测试用的视频帧（640x480，蓝色背景）"""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_detection(self):
        """创建示例人脸检测结果"""
        return FaceDetection(
            x=0.25,
            y=0.25,
            width=0.3,
            height=0.4,
            landmarks=[(0.4, 0.4), (0.5, 0.4)]
        )
    
    def test_renderer_initialization(self, renderer):
        """测试渲染器初始化"""
        assert renderer is not None
        assert renderer.font is not None
        assert renderer.stats_font is not None
    
    def test_draw_face_box(self, renderer, test_frame, sample_detection):
        """测试绘制人脸边界框"""
        original_frame = test_frame.copy()
        renderer.draw_face_box(test_frame, sample_detection)
        
        # 验证帧已被修改（绘制了边界框）
        assert not np.array_equal(test_frame, original_frame)
        
        # 验证帧的形状和类型没有改变
        assert test_frame.shape == original_frame.shape
        assert test_frame.dtype == original_frame.dtype
    
    def test_draw_blessing_text_no_exception(self, renderer, test_frame):
        """测试中文文本渲染不抛出异常"""
        try:
            renderer.draw_blessing_text(test_frame, "鸿运当头", (100, 100))
        except Exception as e:
            pytest.fail(f"绘制中文文本时抛出异常: {e}")
        
        # 验证帧的形状没有改变
        assert test_frame.shape == (480, 640, 3)
    
    def test_draw_blessing_text_modifies_frame(self, renderer, test_frame):
        """测试绘制祝福语会修改帧"""
        original_frame = test_frame.copy()
        renderer.draw_blessing_text(test_frame, "福星高照", (200, 200))
        
        # 验证帧已被修改
        assert not np.array_equal(test_frame, original_frame)
    
    def test_render_frame_with_single_face(self, renderer, test_frame, sample_detection):
        """测试渲染单个人脸"""
        detections = [sample_detection]
        blessings = ["吉祥如意"]
        total_visitors = 1
        
        rendered_frame = renderer.render_frame(
            test_frame,
            detections,
            blessings,
            total_visitors
        )
        
        # 验证返回的帧不为空
        assert rendered_frame is not None
        assert rendered_frame.shape == test_frame.shape
        
        # 验证帧已被修改（绘制了内容）
        assert not np.array_equal(rendered_frame, test_frame)
    
    def test_render_frame_with_multiple_faces(self, renderer, test_frame):
        """测试渲染多个人脸"""
        detections = [
            FaceDetection(0.1, 0.1, 0.2, 0.3, [(0.2, 0.2)]),
            FaceDetection(0.6, 0.1, 0.2, 0.3, [(0.7, 0.2)]),
        ]
        blessings = ["万事如意", "心想事成"]
        total_visitors = 5
        
        rendered_frame = renderer.render_frame(
            test_frame,
            detections,
            blessings,
            total_visitors
        )
        
        # 验证返回的帧不为空
        assert rendered_frame is not None
        assert rendered_frame.shape == test_frame.shape
    
    def test_render_frame_with_no_faces(self, renderer, test_frame):
        """测试渲染无人脸的帧"""
        detections = []
        blessings = []
        total_visitors = 0
        
        rendered_frame = renderer.render_frame(
            test_frame,
            detections,
            blessings,
            total_visitors
        )
        
        # 验证返回的帧不为空
        assert rendered_frame is not None
        assert rendered_frame.shape == test_frame.shape
    
    def test_statistics_display(self, renderer, test_frame):
        """测试统计信息显示"""
        detections = []
        blessings = []
        total_visitors = 42
        
        rendered_frame = renderer.render_frame(
            test_frame,
            detections,
            blessings,
            total_visitors
        )
        
        # 验证帧已被修改（显示了统计信息）
        assert not np.array_equal(rendered_frame, test_frame)
    
    def test_render_frame_does_not_modify_original(self, renderer, test_frame, sample_detection):
        """测试render_frame不修改原始帧"""
        original_frame = test_frame.copy()
        detections = [sample_detection]
        blessings = ["步步高升"]
        total_visitors = 1
        
        renderer.render_frame(test_frame, detections, blessings, total_visitors)
        
        # 验证原始帧未被修改
        assert np.array_equal(test_frame, original_frame)
    
    def test_draw_face_box_with_different_colors(self, renderer, test_frame, sample_detection):
        """测试使用不同颜色绘制边界框"""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        for color in colors:
            frame = test_frame.copy()
            renderer.draw_face_box(frame, sample_detection, color=color)
            # 验证绘制成功（帧被修改）
            assert not np.array_equal(frame, test_frame)
    
    def test_blessing_text_at_edge_positions(self, renderer, test_frame):
        """测试在边缘位置绘制祝福语"""
        positions = [(0, 0), (600, 0), (0, 450), (600, 450)]
        
        for pos in positions:
            frame = test_frame.copy()
            try:
                renderer.draw_blessing_text(frame, "财源广进", pos)
            except Exception as e:
                pytest.fail(f"在位置 {pos} 绘制文本时抛出异常: {e}")
