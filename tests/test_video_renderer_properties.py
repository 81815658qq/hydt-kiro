"""VideoRenderer属性测试

Feature: welcome-greeter
验证视频渲染器的正确性属性
"""

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from src.video_renderer import VideoRenderer
from src.face_detection import FaceDetection


# 生成策略
@st.composite
def frame_strategy(draw):
    """生成随机视频帧"""
    height = draw(st.integers(min_value=240, max_value=1080))
    width = draw(st.integers(min_value=320, max_value=1920))
    # 生成随机颜色的帧
    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return frame


@st.composite
def detection_strategy(draw):
    """生成随机人脸检测结果"""
    x = draw(st.floats(min_value=0.0, max_value=0.7))
    y = draw(st.floats(min_value=0.0, max_value=0.7))
    width = draw(st.floats(min_value=0.1, max_value=0.3))
    height = draw(st.floats(min_value=0.1, max_value=0.3))
    
    # 生成1-5个关键点
    num_landmarks = draw(st.integers(min_value=1, max_value=5))
    landmarks = [
        (draw(st.floats(min_value=0.0, max_value=1.0)),
         draw(st.floats(min_value=0.0, max_value=1.0)))
        for _ in range(num_landmarks)
    ]
    
    return FaceDetection(x=x, y=y, width=width, height=height, landmarks=landmarks)


@st.composite
def blessing_strategy(draw):
    """生成随机四字祝福语"""
    blessings = [
        "鸿运当头", "好运常在", "福星高照", "吉祥如意",
        "万事如意", "心想事成", "步步高升", "财源广进",
        "喜气洋洋", "笑口常开", "福寿安康", "事业有成",
        "前程似锦", "大吉大利", "五福临门", "学业进步"
    ]
    return draw(st.sampled_from(blessings))


class TestVideoRendererProperties:
    """VideoRenderer的属性测试"""
    
    @settings(max_examples=100)
    @given(
        frame=frame_strategy(),
        detections=st.lists(detection_strategy(), min_size=1, max_size=5),
        total_visitors=st.integers(min_value=0, max_value=1000)
    )
    def test_property_4_blessing_rendering_completeness(
        self, frame, detections, total_visitors
    ):
        """属性 4：祝福语渲染完整性
        
        Feature: welcome-greeter, Property 4: 祝福语渲染完整性
        验证需求：3.1, 3.3
        
        对于任何视频帧和人脸检测结果列表，渲染后的帧应该在每个人脸上方
        包含对应的祝福语文本。
        """
        renderer = VideoRenderer()
        
        # 为每个检测结果生成祝福语
        blessings = [f"祝福{i}" for i in range(len(detections))]
        
        # 渲染帧
        rendered_frame = renderer.render_frame(
            frame,
            detections,
            blessings,
            total_visitors
        )
        
        # 验证：渲染后的帧不为空
        assert rendered_frame is not None
        
        # 验证：帧的形状保持不变
        assert rendered_frame.shape == frame.shape
        
        # 验证：帧的数据类型保持不变
        assert rendered_frame.dtype == frame.dtype
        
        # 验证：渲染后的帧与原始帧不同（已绘制内容）
        # 注意：如果检测列表为空，可能相同
        if len(detections) > 0:
            assert not np.array_equal(rendered_frame, frame)
    
    @settings(max_examples=100)
    @given(
        frame=frame_strategy(),
        detections=st.lists(detection_strategy(), min_size=1, max_size=3)
    )
    def test_blessing_text_renders_without_error(self, frame, detections):
        """测试祝福语文本渲染不会抛出异常
        
        对于任何帧和检测结果，绘制祝福语应该成功完成而不抛出异常。
        """
        renderer = VideoRenderer()
        blessings = [f"测试{i}" for i in range(len(detections))]
        
        try:
            rendered_frame = renderer.render_frame(
                frame,
                detections,
                blessings,
                total_visitors=0
            )
            assert rendered_frame is not None
        except Exception as e:
            pytest.fail(f"渲染祝福语时抛出异常: {e}")
    
    @settings(max_examples=100)
    @given(
        frame=frame_strategy(),
        detection=detection_strategy(),
        blessing=blessing_strategy()
    )
    def test_chinese_blessing_renders_correctly(self, frame, detection, blessing):
        """测试中文祝福语正确渲染
        
        对于任何中文祝福语，应该能够成功渲染到帧上。
        """
        renderer = VideoRenderer()
        detections = [detection]
        blessings = [blessing]
        
        try:
            rendered_frame = renderer.render_frame(
                frame,
                detections,
                blessings,
                total_visitors=1
            )
            
            # 验证渲染成功
            assert rendered_frame is not None
            assert rendered_frame.shape == frame.shape
            
            # 验证帧已被修改
            assert not np.array_equal(rendered_frame, frame)
            
        except Exception as e:
            pytest.fail(f"渲染中文祝福语 '{blessing}' 时抛出异常: {e}")
    
    @settings(max_examples=100)
    @given(
        frame=frame_strategy(),
        num_faces=st.integers(min_value=0, max_value=10)
    )
    def test_render_handles_variable_face_count(self, frame, num_faces):
        """测试渲染器处理不同数量的人脸
        
        对于任何人脸数量（包括0），渲染器应该正确处理。
        """
        renderer = VideoRenderer()
        # 生成指定数量的检测结果
        detections = [
            FaceDetection(
                x=0.1 + i * 0.1,
                y=0.1,
                width=0.1,
                height=0.15,
                landmarks=[(0.15, 0.15)]
            )
            for i in range(num_faces)
        ]
        blessings = [f"祝福{i}" for i in range(num_faces)]
        
        try:
            rendered_frame = renderer.render_frame(
                frame,
                detections,
                blessings,
                total_visitors=num_faces
            )
            
            assert rendered_frame is not None
            assert rendered_frame.shape == frame.shape
            
        except Exception as e:
            pytest.fail(f"渲染 {num_faces} 个人脸时抛出异常: {e}")
    
    @settings(max_examples=100)
    @given(
        frame=frame_strategy(),
        detection=detection_strategy()
    )
    def test_face_box_drawing_preserves_frame_properties(self, frame, detection):
        """测试绘制人脸框保持帧的属性
        
        对于任何帧和检测结果，绘制边界框后帧的形状和类型应该保持不变。
        """
        renderer = VideoRenderer()
        original_shape = frame.shape
        original_dtype = frame.dtype
        
        frame_copy = frame.copy()
        renderer.draw_face_box(frame_copy, detection)
        
        # 验证形状和类型不变
        assert frame_copy.shape == original_shape
        assert frame_copy.dtype == original_dtype
    
    @settings(max_examples=100)
    @given(
        frame=frame_strategy(),
        detections=st.lists(detection_strategy(), min_size=1, max_size=5),
        total_visitors=st.integers(min_value=0, max_value=1000)
    )
    def test_render_frame_is_deterministic(self, frame, detections, total_visitors):
        """测试渲染是确定性的
        
        对于相同的输入，多次渲染应该产生相同的结果。
        """
        renderer = VideoRenderer()
        blessings = [f"祝福{i}" for i in range(len(detections))]
        
        # 第一次渲染
        rendered1 = renderer.render_frame(
            frame.copy(),
            detections,
            blessings,
            total_visitors
        )
        
        # 第二次渲染
        rendered2 = renderer.render_frame(
            frame.copy(),
            detections,
            blessings,
            total_visitors
        )
        
        # 验证两次渲染结果相同
        assert np.array_equal(rendered1, rendered2)
    
    @settings(max_examples=100)
    @given(
        frame=frame_strategy(),
        detections=st.lists(detection_strategy(), min_size=0, max_size=5),
        total_visitors=st.integers(min_value=0, max_value=1000)
    )
    def test_property_10_statistics_display_completeness(
        self, frame, detections, total_visitors
    ):
        """属性 10：统计信息显示完整性
        
        Feature: welcome-greeter, Property 10: 统计信息显示完整性
        验证需求：5.3
        
        对于任何渲染的视频帧，应该包含当前总访客数量的文本显示。
        """
        renderer = VideoRenderer()
        blessings = [f"祝福{i}" for i in range(len(detections))]
        
        # 渲染帧
        rendered_frame = renderer.render_frame(
            frame.copy(),
            detections,
            blessings,
            total_visitors
        )
        
        # 验证：渲染后的帧不为空
        assert rendered_frame is not None
        
        # 验证：帧的形状保持不变
        assert rendered_frame.shape == frame.shape
        
        # 验证：渲染后的帧与原始帧不同（已绘制统计信息）
        # 统计信息应该总是被绘制，即使没有检测到人脸
        assert not np.array_equal(rendered_frame, frame), \
            "渲染后的帧应该包含统计信息，与原始帧不同"
        
        # 验证：统计信息区域（左上角）有变化
        # 检查左上角区域（统计信息通常显示在这里）
        stats_region_original = frame[0:50, 0:300]
        stats_region_rendered = rendered_frame[0:50, 0:300]
        
        # 统计信息区域应该有变化
        assert not np.array_equal(stats_region_original, stats_region_rendered), \
            "统计信息区域应该被修改以显示访客数量"
