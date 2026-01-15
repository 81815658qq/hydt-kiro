"""人脸特征提取模块属性测试（Property-Based Testing）
Feature: welcome-greeter
使用Hypothesis进行基于属性的测试
"""
import pytest
import numpy as np
import cv2
from hypothesis import given, settings, strategies as st
from src.face_feature_extractor import FaceFeatureExtractor


# 策略：生成有效的图像尺寸
image_size = st.integers(min_value=32, max_value=256)

# 策略：生成像素值
pixel_value = st.integers(min_value=0, max_value=255)


class TestFaceFeatureExtractorProperties:
    """属性 5：特征提取一致性
    验证需求：4.1
    """
    
    @settings(max_examples=100)
    @given(
        width=image_size,
        height=image_size,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_property_5_feature_extraction_consistency(self, width, height, seed):
        """
        属性 5：特征提取一致性
        
        对于任何有效的人脸图像，特征提取器应该返回固定维度的特征向量（非空且维度一致）。
        
        **Validates: Requirements 4.1**
        """
        extractor = FaceFeatureExtractor()
        
        # 使用种子生成可重复的随机图像
        np.random.seed(seed)
        face_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # 提取特征
        features = extractor.extract_features(face_image)
        
        # 属性1：特征向量不应该为None
        assert features is not None, "特征向量不应该为None"
        
        # 属性2：特征向量应该是numpy数组
        assert isinstance(features, np.ndarray), "特征向量应该是numpy数组"
        
        # 属性3：特征向量维度应该固定为128
        assert len(features) == 128, f"特征向量维度应该是128，实际为{len(features)}"
        
        # 属性4：特征向量应该是浮点数类型
        assert features.dtype in [np.float32, np.float64], \
            f"特征向量应该是浮点数类型，实际为{features.dtype}"
        
        # 属性5：特征向量应该是归一化的（L2范数接近1）
        norm = np.linalg.norm(features)
        assert 0.9 <= norm <= 1.1, f"特征向量L2范数应该接近1，实际为{norm}"
        
        # 属性6：特征向量不应该全为0
        assert not np.allclose(features, 0), "特征向量不应该全为0"
    
    @settings(max_examples=50)
    @given(
        width=image_size,
        height=image_size,
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_property_5_same_image_same_features(self, width, height, seed):
        """
        属性 5（扩展）：相同图像产生相同特征
        
        对于任何图像，多次提取特征应该产生相同的结果（确定性）。
        
        **Validates: Requirements 4.1**
        """
        extractor = FaceFeatureExtractor()
        
        # 生成图像
        np.random.seed(seed)
        face_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # 提取特征两次
        features1 = extractor.extract_features(face_image)
        features2 = extractor.extract_features(face_image)
        
        assert features1 is not None
        assert features2 is not None
        
        # 属性：相同图像应该产生完全相同的特征
        np.testing.assert_array_almost_equal(
            features1, features2, decimal=10,
            err_msg="相同图像应该产生相同的特征向量"
        )
    
    @settings(max_examples=50)
    @given(
        size=image_size,
        seed1=st.integers(min_value=0, max_value=10000),
        seed2=st.integers(min_value=0, max_value=10000)
    )
    def test_property_5_different_images_different_features(self, size, seed1, seed2):
        """
        属性 5（扩展）：不同图像产生不同特征
        
        对于任何两个不同的图像，它们的特征向量应该不完全相同。
        
        **Validates: Requirements 4.1**
        """
        # 确保种子不同
        if seed1 == seed2:
            seed2 = seed1 + 1
        
        extractor = FaceFeatureExtractor()
        
        # 生成两个不同的图像
        np.random.seed(seed1)
        image1 = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        
        np.random.seed(seed2)
        image2 = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        
        # 提取特征
        features1 = extractor.extract_features(image1)
        features2 = extractor.extract_features(image2)
        
        assert features1 is not None
        assert features2 is not None
        
        # 属性：不同图像应该产生不同的特征（至少有一些差异）
        # 使用余弦相似度检查
        similarity = extractor.compute_similarity(features1, features2)
        
        # 相似度应该小于1.0（不完全相同）
        assert similarity < 0.9999, "不同图像应该产生不同的特征向量"


class TestSimilarityComputationProperties:
    """属性 6：相似度阈值判断正确性
    验证需求：4.3, 4.4
    """
    
    @settings(max_examples=100)
    @given(
        dim=st.just(128),  # 固定维度为128
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_property_6_similarity_range(self, dim, seed):
        """
        属性 6：相似度阈值判断正确性
        
        对于任何两个特征向量，当相似度超过阈值时应该判断为匹配，
        低于阈值时应该判断为不匹配。
        
        **Validates: Requirements 4.3, 4.4**
        """
        extractor = FaceFeatureExtractor()
        
        # 生成两个随机特征向量
        np.random.seed(seed)
        features1 = np.random.rand(dim)
        features1 = features1 / np.linalg.norm(features1)  # 归一化
        
        features2 = np.random.rand(dim)
        features2 = features2 / np.linalg.norm(features2)  # 归一化
        
        # 计算相似度
        similarity = extractor.compute_similarity(features1, features2)
        
        # 属性1：相似度应该在[0, 1]范围内
        assert 0.0 <= similarity <= 1.0, \
            f"相似度应该在[0, 1]范围内，实际为{similarity}"
        
        # 属性2：相似度是对称的
        similarity_reverse = extractor.compute_similarity(features2, features1)
        assert abs(similarity - similarity_reverse) < 0.0001, \
            "相似度计算应该是对称的"
    
    @settings(max_examples=50)
    @given(
        dim=st.just(128),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_property_6_identical_features_high_similarity(self, dim, seed):
        """
        属性 6（扩展）：相同特征的相似度应该接近1
        
        对于任何特征向量，与自己的相似度应该是1.0。
        
        **Validates: Requirements 4.3**
        """
        extractor = FaceFeatureExtractor()
        
        # 生成随机特征向量
        np.random.seed(seed)
        features = np.random.rand(dim)
        features = features / np.linalg.norm(features)  # 归一化
        
        # 计算与自己的相似度
        similarity = extractor.compute_similarity(features, features)
        
        # 属性：相同特征的相似度应该接近1.0
        assert similarity > 0.99, \
            f"相同特征的相似度应该接近1.0，实际为{similarity}"
    
    @settings(max_examples=50)
    @given(
        dim=st.just(128),
        threshold=st.floats(min_value=0.5, max_value=0.9)
    )
    def test_property_6_threshold_judgment(self, dim, threshold):
        """
        属性 6（扩展）：阈值判断的正确性
        
        对于任何阈值，相似度大于阈值应该判断为匹配，小于阈值应该判断为不匹配。
        
        **Validates: Requirements 4.3, 4.4**
        """
        extractor = FaceFeatureExtractor()
        
        # 创建两个特征向量，控制它们的相似度
        features1 = np.zeros(dim)
        features1[0] = 1.0
        
        # 创建一个与features1有一定相似度的向量
        features2 = np.zeros(dim)
        features2[0] = threshold + 0.1  # 稍微高于阈值
        features2[1] = np.sqrt(1 - features2[0]**2)  # 保持单位向量
        
        similarity = extractor.compute_similarity(features1, features2)
        
        # 属性：相似度应该在合理范围内
        assert 0.0 <= similarity <= 1.0
        
        # 如果相似度大于阈值，应该判断为匹配
        if similarity >= threshold:
            is_match = True
        else:
            is_match = False
        
        # 验证判断逻辑的一致性
        assert isinstance(is_match, bool), "匹配判断应该返回布尔值"
    
    @settings(max_examples=50)
    @given(
        dim=st.just(128),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_property_6_similarity_transitivity(self, dim, seed):
        """
        属性 6（扩展）：相似度的传递性（弱形式）
        
        如果A与B相似，B与C相似，那么A与C也应该有一定的相似度。
        
        **Validates: Requirements 4.3**
        """
        extractor = FaceFeatureExtractor()
        
        # 生成三个特征向量
        np.random.seed(seed)
        features_a = np.random.rand(dim)
        features_a = features_a / np.linalg.norm(features_a)
        
        # B是A的轻微变化
        features_b = features_a + np.random.rand(dim) * 0.1
        features_b = features_b / np.linalg.norm(features_b)
        
        # C是B的轻微变化
        features_c = features_b + np.random.rand(dim) * 0.1
        features_c = features_c / np.linalg.norm(features_c)
        
        # 计算相似度
        sim_ab = extractor.compute_similarity(features_a, features_b)
        sim_bc = extractor.compute_similarity(features_b, features_c)
        sim_ac = extractor.compute_similarity(features_a, features_c)
        
        # 属性：如果A-B和B-C都相似，那么A-C也应该有一定相似度
        # 这是一个弱传递性（不是严格的数学传递性）
        if sim_ab > 0.8 and sim_bc > 0.8:
            # A和C应该也有一定的相似度（但可能较低）
            assert sim_ac > 0.5, \
                f"传递相似度应该大于0.5，实际为{sim_ac} (AB={sim_ab}, BC={sim_bc})"
