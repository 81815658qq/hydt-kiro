"""人脸特征提取模块单元测试"""
import pytest
import numpy as np
import cv2
from src.face_feature_extractor import FaceFeatureExtractor


class TestFaceFeatureExtractor:
    """测试FaceFeatureExtractor类"""
    
    @pytest.fixture
    def extractor(self):
        """创建FaceFeatureExtractor实例"""
        return FaceFeatureExtractor()
    
    @pytest.fixture
    def sample_face_image(self):
        """创建一个简单的人脸图像（64x64 BGR格式）"""
        # 创建一个带有简单图案的图像
        img = np.ones((64, 64, 3), dtype=np.uint8) * 200
        
        # 绘制一个椭圆模拟人脸
        cv2.ellipse(img, (32, 32), (20, 25), 0, 0, 360, (180, 150, 120), -1)
        
        # 绘制眼睛
        cv2.circle(img, (24, 28), 3, (0, 0, 0), -1)
        cv2.circle(img, (40, 28), 3, (0, 0, 0), -1)
        
        # 绘制嘴巴
        cv2.ellipse(img, (32, 40), (8, 4), 0, 0, 180, (100, 50, 50), 1)
        
        return img
    
    @pytest.fixture
    def different_face_image(self):
        """创建一个不同的人脸图像"""
        # 创建一个不同的图像
        img = np.ones((64, 64, 3), dtype=np.uint8) * 150
        
        # 绘制一个不同形状的椭圆
        cv2.ellipse(img, (32, 32), (25, 20), 0, 0, 360, (160, 140, 110), -1)
        
        # 绘制不同位置的眼睛
        cv2.circle(img, (22, 26), 4, (0, 0, 0), -1)
        cv2.circle(img, (42, 26), 4, (0, 0, 0), -1)
        
        # 绘制不同的嘴巴
        cv2.line(img, (24, 42), (40, 42), (100, 50, 50), 2)
        
        return img
    
    def test_extractor_initialization(self, extractor):
        """测试特征提取器初始化"""
        assert extractor is not None
        assert extractor.feature_size == (64, 64)
    
    def test_extract_features_returns_vector(self, extractor, sample_face_image):
        """测试特征提取返回特征向量"""
        features = extractor.extract_features(sample_face_image)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features) == 128  # 应该是128维
        assert features.dtype == np.float64 or features.dtype == np.float32
    
    def test_extract_features_consistency(self, extractor, sample_face_image):
        """测试相同人脸图像的特征一致性"""
        features1 = extractor.extract_features(sample_face_image)
        features2 = extractor.extract_features(sample_face_image)
        
        assert features1 is not None
        assert features2 is not None
        
        # 相同图像应该产生相同的特征
        np.testing.assert_array_almost_equal(features1, features2, decimal=5)
    
    def test_extract_features_different_images(self, extractor, sample_face_image, different_face_image):
        """测试不同人脸的特征差异"""
        features1 = extractor.extract_features(sample_face_image)
        features2 = extractor.extract_features(different_face_image)
        
        assert features1 is not None
        assert features2 is not None
        
        # 不同图像应该产生不同的特征
        # 使用余弦相似度检查
        similarity = extractor.compute_similarity(features1, features2)
        
        # 相似度应该小于1.0（不完全相同）
        assert similarity < 1.0
    
    def test_extract_features_invalid_input(self, extractor):
        """测试无效输入的错误处理"""
        # None输入
        features = extractor.extract_features(None)
        assert features is None
        
        # 空数组
        empty_image = np.array([])
        features = extractor.extract_features(empty_image)
        assert features is None
    
    def test_extract_features_grayscale_image(self, extractor):
        """测试灰度图像的特征提取"""
        # 创建灰度图像
        gray_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        
        features = extractor.extract_features(gray_image)
        
        assert features is not None
        assert len(features) == 128
    
    def test_compute_similarity_range(self, extractor, sample_face_image):
        """测试相似度计算范围"""
        features1 = extractor.extract_features(sample_face_image)
        features2 = extractor.extract_features(sample_face_image)
        
        similarity = extractor.compute_similarity(features1, features2)
        
        # 相似度应该在[0, 1]范围内
        assert 0.0 <= similarity <= 1.0
        
        # 相同特征的相似度应该接近1.0
        assert similarity > 0.95
    
    def test_compute_similarity_identical_features(self, extractor):
        """测试相同特征的相似度"""
        features = np.random.rand(128)
        features = features / np.linalg.norm(features)  # 归一化
        
        similarity = extractor.compute_similarity(features, features)
        
        # 相同特征的相似度应该是1.0
        assert abs(similarity - 1.0) < 0.01
    
    def test_compute_similarity_orthogonal_features(self, extractor):
        """测试正交特征的相似度"""
        # 创建两个正交的特征向量
        features1 = np.zeros(128)
        features1[0] = 1.0
        
        features2 = np.zeros(128)
        features2[1] = 1.0
        
        similarity = extractor.compute_similarity(features1, features2)
        
        # 正交向量的余弦相似度应该接近0.5（映射后）
        assert 0.4 <= similarity <= 0.6
    
    def test_compute_similarity_invalid_input(self, extractor):
        """测试无效输入的相似度计算"""
        features = np.random.rand(128)
        
        # None输入
        similarity = extractor.compute_similarity(None, features)
        assert similarity == 0.0
        
        similarity = extractor.compute_similarity(features, None)
        assert similarity == 0.0
        
        # 维度不匹配
        features_wrong_dim = np.random.rand(64)
        similarity = extractor.compute_similarity(features, features_wrong_dim)
        assert similarity == 0.0
    
    def test_feature_normalization(self, extractor, sample_face_image):
        """测试特征向量归一化"""
        features = extractor.extract_features(sample_face_image)
        
        assert features is not None
        
        # 检查L2范数接近1（归一化后）
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 0.01
    
    def test_extract_features_different_sizes(self, extractor):
        """测试不同尺寸图像的特征提取"""
        # 测试不同尺寸的图像
        sizes = [(32, 32), (64, 64), (128, 128), (100, 80)]
        
        for size in sizes:
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            features = extractor.extract_features(img)
            
            assert features is not None
            assert len(features) == 128  # 应该始终是128维
