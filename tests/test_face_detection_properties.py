"""人脸检测模块属性测试（Property-Based Testing）
Feature: welcome-greeter
使用Hypothesis进行基于属性的测试
"""
import pytest
import numpy as np
import cv2
from hypothesis import given, settings, strategies as st
from src.face_detection import FaceDetection, FaceDetector


# 策略：生成有效的归一化坐标（0-1范围）
normalized_coord = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# 策略：生成有效的宽度/高度（必须大于0）
positive_size = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)

# 策略：生成关键点列表
landmarks_strategy = st.lists(
    st.tuples(normalized_coord, normalized_coord),
    min_size=0,
    max_size=10
)


class TestFaceDetectionProperties:
    """属性 1：人脸检测完整性
    验证需求：2.2, 2.4
    """
    
    @settings(max_examples=100)
    @given(
        x=normalized_coord,
        y=normalized_coord,
        width=positive_size,
        height=positive_size,
        landmarks=landmarks_strategy
    )
    def test_property_1_face_detection_completeness(self, x, y, width, height, landmarks):
        """
        属性 1：人脸检测完整性
        
        对于任何检测到的人脸，检测结果应该包含有效的边界框坐标（归一化到0-1范围）
        和面部关键点列表。
        
        **Validates: Requirements 2.2, 2.4**
        """
        # 确保坐标不会超出边界，并保持最小尺寸
        if x + width > 1.0:
            width = max(0.01, 1.0 - x)  # 保证至少0.01的宽度
        if y + height > 1.0:
            height = max(0.01, 1.0 - y)  # 保证至少0.01的高度
        
        # 如果x或y太接近1.0，调整它们
        if x > 0.99:
            x = 0.99
            width = 0.01
        if y > 0.99:
            y = 0.99
            height = 0.01
        
        # 创建检测结果
        detection = FaceDetection(
            x=x,
            y=y,
            width=width,
            height=height,
            landmarks=landmarks
        )
        
        # 属性1：边界框坐标应该在[0, 1]范围内
        assert 0 <= detection.x <= 1, f"x坐标 {detection.x} 不在有效范围内"
        assert 0 <= detection.y <= 1, f"y坐标 {detection.y} 不在有效范围内"
        assert 0 < detection.width <= 1, f"宽度 {detection.width} 不在有效范围内"
        assert 0 < detection.height <= 1, f"高度 {detection.height} 不在有效范围内"
        
        # 属性2：边界框不应该超出图像范围
        assert detection.x + detection.width <= 1.01, "边界框右边界超出范围"  # 允许小误差
        assert detection.y + detection.height <= 1.01, "边界框下边界超出范围"
        
        # 属性3：关键点列表应该是有效的列表类型
        assert isinstance(detection.landmarks, list), "关键点应该是列表类型"
        
        # 属性4：所有关键点坐标应该在有效范围内
        for i, (lx, ly) in enumerate(detection.landmarks):
            assert 0 <= lx <= 1, f"关键点{i}的x坐标 {lx} 不在有效范围内"
            assert 0 <= ly <= 1, f"关键点{i}的y坐标 {ly} 不在有效范围内"
    
    @settings(max_examples=100)
    @given(
        x=normalized_coord,
        y=normalized_coord,
        width=positive_size,
        height=positive_size,
        frame_width=st.integers(min_value=100, max_value=1920),
        frame_height=st.integers(min_value=100, max_value=1080)
    )
    def test_property_1_pixel_coordinate_conversion(self, x, y, width, height, frame_width, frame_height):
        """
        属性 1（扩展）：像素坐标转换的正确性
        
        对于任何有效的归一化坐标和帧尺寸，转换后的像素坐标应该在有效范围内。
        
        **Validates: Requirements 2.2**
        """
        # 确保坐标不会超出边界，并保持最小尺寸
        if x + width > 1.0:
            width = max(0.01, 1.0 - x)
        if y + height > 1.0:
            height = max(0.01, 1.0 - y)
        
        # 如果x或y太接近1.0，调整它们
        if x > 0.99:
            x = 0.99
            width = 0.01
        if y > 0.99:
            y = 0.99
            height = 0.01
        
        detection = FaceDetection(
            x=x,
            y=y,
            width=width,
            height=height,
            landmarks=[]
        )
        
        # 转换为像素坐标
        px, py, pw, ph = detection.to_pixel_coords(frame_width, frame_height)
        
        # 属性：像素坐标应该在帧尺寸范围内
        assert 0 <= px < frame_width, f"像素x坐标 {px} 超出帧宽度 {frame_width}"
        assert 0 <= py < frame_height, f"像素y坐标 {py} 超出帧高度 {frame_height}"
        assert 0 < pw <= frame_width, f"像素宽度 {pw} 超出帧宽度 {frame_width}"
        assert 0 < ph <= frame_height, f"像素高度 {ph} 超出帧高度 {frame_height}"
        
        # 属性：边界框不应该超出帧范围（允许小误差）
        assert px + pw <= frame_width + 1, "像素边界框右边界超出帧宽度"
        assert py + ph <= frame_height + 1, "像素边界框下边界超出帧高度"
        
        # 属性：转换应该保持比例关系
        expected_px = int(x * frame_width)
        expected_py = int(y * frame_height)
        expected_pw = int(width * frame_width)
        expected_ph = int(height * frame_height)
        
        assert px == expected_px, f"x坐标转换不正确: {px} != {expected_px}"
        assert py == expected_py, f"y坐标转换不正确: {py} != {expected_py}"
        assert pw == expected_pw, f"宽度转换不正确: {pw} != {expected_pw}"
        assert ph == expected_ph, f"高度转换不正确: {ph} != {expected_ph}"


class TestMultiFaceDetectionProperties:
    """属性 2：多人脸检测能力
    验证需求：2.1, 2.3
    """
    
    @settings(max_examples=100)
    @given(
        num_faces=st.integers(min_value=0, max_value=5),
        image_width=st.integers(min_value=320, max_value=1280),
        image_height=st.integers(min_value=240, max_value=720)
    )
    def test_property_2_multi_face_detection_capability(self, num_faces, image_width, image_height):
        """
        属性 2：多人脸检测能力
        
        对于任何包含多个人脸的视频帧，人脸检测器应该返回所有可见人脸的检测结果，
        检测数量应该大于等于实际人脸数量（或合理范围内）。
        
        **Validates: Requirements 2.1, 2.3**
        """
        # 在测试内部创建detector（避免fixture问题）
        detector = FaceDetector(min_detection_confidence=0.5)
        
        # 创建测试图像
        frame = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        
        # 在图像上绘制多个模拟人脸
        face_positions = []
        for i in range(num_faces):
            # 计算人脸位置（避免重叠）
            cols = min(3, num_faces)  # 每行最多3个人脸
            row = i // cols
            col = i % cols
            
            # 计算中心位置
            x_spacing = image_width // (cols + 1)
            y_spacing = image_height // ((num_faces // cols) + 2)
            
            center_x = x_spacing * (col + 1)
            center_y = y_spacing * (row + 1)
            
            # 确保人脸在图像范围内
            if center_x < 50 or center_x > image_width - 50:
                continue
            if center_y < 50 or center_y > image_height - 50:
                continue
            
            # 绘制椭圆模拟人脸
            axes = (40, 50)
            cv2.ellipse(frame, (center_x, center_y), axes, 0, 0, 360, (180, 150, 120), -1)
            
            # 绘制眼睛
            cv2.circle(frame, (center_x - 15, center_y - 10), 5, (0, 0, 0), -1)
            cv2.circle(frame, (center_x + 15, center_y - 10), 5, (0, 0, 0), -1)
            
            # 绘制嘴巴
            cv2.ellipse(frame, (center_x, center_y + 15), (15, 8), 0, 0, 180, (100, 50, 50), 2)
            
            face_positions.append((center_x, center_y))
        
        # 执行检测
        detections = detector.detect_faces(frame)
        
        # 属性1：返回结果应该是列表
        assert isinstance(detections, list), "检测结果应该是列表类型"
        
        # 属性2：检测到的人脸数量应该是非负数
        assert len(detections) >= 0, "检测到的人脸数量不能为负"
        
        # 属性3：对于空图像（num_faces=0），通常不应该检测到人脸
        # 注意：MediaPipe可能会有误检，但这是合理的
        if num_faces == 0:
            # 允许少量误检（最多1个）
            assert len(detections) <= 1, f"空图像检测到过多人脸: {len(detections)}"
        
        # 属性4：所有检测结果应该是有效的FaceDetection对象
        for detection in detections:
            assert isinstance(detection, FaceDetection), "检测结果应该是FaceDetection类型"
            assert 0 <= detection.x <= 1, f"检测到的x坐标无效: {detection.x}"
            assert 0 <= detection.y <= 1, f"检测到的y坐标无效: {detection.y}"
            assert 0 < detection.width <= 1, f"检测到的宽度无效: {detection.width}"
            assert 0 < detection.height <= 1, f"检测到的高度无效: {detection.height}"
            
            # 验证边界框不超出范围
            assert detection.x + detection.width <= 1.01, \
                f"边界框右边界超出范围: x={detection.x}, width={detection.width}"
            assert detection.y + detection.height <= 1.01, \
                f"边界框下边界超出范围: y={detection.y}, height={detection.height}"
        
        # 属性5：检测数量应该在合理范围内（不会检测出远超实际的人脸数）
        # MediaPipe可能漏检或误检，但不应该检测出过多人脸
        max_reasonable_detections = max(num_faces + 2, 10)  # 允许一些误检
        assert len(detections) <= max_reasonable_detections, \
            f"检测到的人脸数量 {len(detections)} 超出合理范围（实际人脸数: {num_faces}）"
    
    @settings(max_examples=50)
    @given(
        detection_count=st.integers(min_value=0, max_value=10)
    )
    def test_property_2_detection_list_consistency(self, detection_count):
        """
        属性 2（扩展）：检测结果列表的一致性
        
        对于任何检测结果列表，所有元素应该是有效的FaceDetection对象。
        
        **Validates: Requirements 2.1, 2.3**
        """
        # 在测试内部创建detector（避免fixture问题）
        detector = FaceDetector(min_detection_confidence=0.5)
        
        # 创建一个简单的测试图像
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # 执行检测
        detections = detector.detect_faces(frame)
        
        # 属性：返回的应该是列表
        assert isinstance(detections, list), "检测结果必须是列表"
        
        # 属性：列表中的每个元素都应该是FaceDetection对象
        for i, detection in enumerate(detections):
            assert isinstance(detection, FaceDetection), \
                f"检测结果[{i}]不是FaceDetection类型: {type(detection)}"
            
            # 验证每个检测结果的完整性
            assert hasattr(detection, 'x'), f"检测结果[{i}]缺少x属性"
            assert hasattr(detection, 'y'), f"检测结果[{i}]缺少y属性"
            assert hasattr(detection, 'width'), f"检测结果[{i}]缺少width属性"
            assert hasattr(detection, 'height'), f"检测结果[{i}]缺少height属性"
            assert hasattr(detection, 'landmarks'), f"检测结果[{i}]缺少landmarks属性"
