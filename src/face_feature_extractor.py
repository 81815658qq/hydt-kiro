"""人脸特征提取模块"""
import numpy as np
import cv2
from typing import Optional
from src.logger import get_logger
from src.config import FACE_IMAGE_SIZE

logger = get_logger(__name__)


class FaceFeatureExtractor:
    """人脸特征提取器，使用简化的特征提取方法"""
    
    def __init__(self):
        """初始化特征提取器"""
        self.feature_size = FACE_IMAGE_SIZE
        logger.info(f"FaceFeatureExtractor initialized with feature size: {self.feature_size}")
    
    def extract_features(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        从人脸图像提取特征向量
        
        使用简化的方法：
        1. 调整图像大小到固定尺寸
        2. 转换为灰度图
        3. 计算直方图作为特征
        4. 归一化特征向量
        
        Args:
            face_image: BGR格式的人脸图像
            
        Returns:
            特征向量（128维），如果提取失败则返回None
        """
        try:
            if face_image is None or face_image.size == 0:
                logger.warning("Invalid face image provided")
                return None
            
            # 调整图像大小
            resized = cv2.resize(face_image, self.feature_size)
            
            # 转换为灰度图
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            
            # 归一化图像
            normalized = cv2.equalizeHist(gray)
            
            # 提取多种特征并组合
            features = []
            
            # 1. 直方图特征（64维）
            hist = cv2.calcHist([normalized], [0], None, [64], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)  # 归一化
            features.append(hist)
            
            # 2. 分块平均值特征（4x4=16维）
            h, w = normalized.shape
            block_h, block_w = h // 4, w // 4
            block_features = []
            for i in range(4):
                for j in range(4):
                    block = normalized[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    block_features.append(block.mean())
            block_features = np.array(block_features)
            block_features = block_features / 255.0  # 归一化到[0,1]
            features.append(block_features)
            
            # 3. 梯度特征（48维）
            # 计算Sobel梯度
            sobelx = cv2.Sobel(normalized, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(normalized, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2)
            
            # 分块梯度统计
            gradient_features = []
            for i in range(4):
                for j in range(3):
                    block = gradient_mag[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    gradient_features.append(block.mean())
                    gradient_features.append(block.std())
                    gradient_features.append(block.max())
            gradient_features = np.array(gradient_features)
            gradient_features = gradient_features / (gradient_features.max() + 1e-7)
            features.append(gradient_features)
            
            # 组合所有特征
            feature_vector = np.concatenate(features)
            
            # 确保特征向量是128维
            if len(feature_vector) < 128:
                # 填充到128维
                feature_vector = np.pad(feature_vector, (0, 128 - len(feature_vector)))
            elif len(feature_vector) > 128:
                # 截断到128维
                feature_vector = feature_vector[:128]
            
            # L2归一化
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm
            
            logger.debug(f"Extracted feature vector with shape: {feature_vector.shape}")
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        计算两个特征向量的相似度（余弦相似度）
        
        Args:
            features1: 第一个特征向量
            features2: 第二个特征向量
            
        Returns:
            相似度值（0-1之间，1表示完全相同）
        """
        try:
            if features1 is None or features2 is None:
                logger.warning("Cannot compute similarity with None features")
                return 0.0
            
            if len(features1) != len(features2):
                logger.warning(f"Feature dimension mismatch: {len(features1)} vs {len(features2)}")
                return 0.0
            
            # 余弦相似度
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # 将相似度限制在[0, 1]范围内
            # 余弦相似度范围是[-1, 1]，我们映射到[0, 1]
            similarity = (similarity + 1.0) / 2.0
            similarity = np.clip(similarity, 0.0, 1.0)
            
            logger.debug(f"Computed similarity: {similarity:.4f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0
