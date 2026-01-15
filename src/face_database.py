"""人脸数据库模块"""
import os
import json
import uuid
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VisitorRecord:
    """访客记录数据类"""
    visitor_id: str  # 唯一访客ID (UUID格式)
    features: np.ndarray  # 特征向量
    face_image_path: str  # 人脸图片保存路径（相对路径）
    first_seen: datetime  # 首次出现时间
    blessing: str  # 分配的祝福语
    
    def to_dict(self) -> dict:
        """
        序列化为字典（用于JSON存储）
        
        Returns:
            字典格式的访客记录（不包含特征向量，特征向量单独存储）
        """
        return {
            "visitor_id": self.visitor_id,
            "face_image_path": self.face_image_path,
            "first_seen": self.first_seen.isoformat(),
            "blessing": self.blessing
        }
    
    @classmethod
    def from_dict(cls, data: dict, features: np.ndarray) -> 'VisitorRecord':
        """
        从字典反序列化
        
        Args:
            data: 字典格式的访客记录
            features: 特征向量（从单独的文件加载）
            
        Returns:
            VisitorRecord实例
        """
        return cls(
            visitor_id=data["visitor_id"],
            features=features,
            face_image_path=data["face_image_path"],
            first_seen=datetime.fromisoformat(data["first_seen"]),
            blessing=data["blessing"]
        )


class FaceDatabase:
    """人脸数据库，存储和管理已识别的访客人脸数据"""
    
    def __init__(self, storage_dir: str = "./face_data"):
        """
        初始化数据库
        
        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = Path(storage_dir)
        self.features_dir = self.storage_dir / "features"
        self.images_dir = self.storage_dir / "images"
        self.metadata_file = self.storage_dir / "metadata.json"
        
        # 创建目录结构
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        
        # 访客记录列表
        self.visitors: List[VisitorRecord] = []
        
        # 加载已有数据
        self.load_from_disk()
        
        logger.info(f"FaceDatabase initialized with storage_dir: {storage_dir}")
        logger.info(f"Loaded {len(self.visitors)} existing visitors")
    
    def find_matching_visitor(
        self, 
        features: np.ndarray, 
        threshold: float = 0.7
    ) -> Optional[VisitorRecord]:
        """
        查找匹配的访客记录
        
        Args:
            features: 待匹配的特征向量
            threshold: 相似度阈值（默认0.7）
            
        Returns:
            匹配的访客记录，如果没有匹配则返回None
        """
        from src.face_feature_extractor import FaceFeatureExtractor
        
        extractor = FaceFeatureExtractor()
        best_match = None
        best_similarity = 0.0
        
        for visitor in self.visitors:
            similarity = extractor.compute_similarity(features, visitor.features)
            
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = visitor
        
        if best_match:
            logger.debug(f"Found matching visitor: {best_match.visitor_id} "
                        f"with similarity: {best_similarity:.4f}")
        else:
            logger.debug(f"No matching visitor found (threshold: {threshold})")
        
        return best_match
    
    def add_visitor(
        self, 
        features: np.ndarray, 
        face_image: np.ndarray, 
        blessing: str
    ) -> VisitorRecord:
        """
        添加新访客记录
        
        Args:
            features: 特征向量
            face_image: 人脸图片（BGR格式）
            blessing: 分配的祝福语
            
        Returns:
            新创建的访客记录
        """
        import cv2
        
        # 生成唯一ID
        visitor_id = str(uuid.uuid4())
        
        # 保存特征向量
        feature_path = self.features_dir / f"{visitor_id}.npy"
        np.save(feature_path, features)
        
        # 保存人脸图片
        image_filename = f"{visitor_id}.jpg"
        image_path = self.images_dir / image_filename
        cv2.imwrite(str(image_path), face_image)
        
        # 创建访客记录
        visitor = VisitorRecord(
            visitor_id=visitor_id,
            features=features,
            face_image_path=f"images/{image_filename}",
            first_seen=datetime.now(),
            blessing=blessing
        )
        
        # 添加到列表
        self.visitors.append(visitor)
        
        # 保存元数据
        self.save_to_disk()
        
        logger.info(f"Added new visitor: {visitor_id} with blessing: {blessing}")
        return visitor
    
    def get_total_visitors(self) -> int:
        """
        获取总访客数量
        
        Returns:
            访客总数
        """
        return len(self.visitors)
    
    def load_from_disk(self):
        """从磁盘加载已保存的访客数据"""
        try:
            if not self.metadata_file.exists():
                logger.info("No existing metadata file found, starting with empty database")
                return
            
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.visitors = []
            for visitor_data in data.get("visitors", []):
                visitor_id = visitor_data["visitor_id"]
                
                # 加载特征向量
                feature_path = self.features_dir / f"{visitor_id}.npy"
                if not feature_path.exists():
                    logger.warning(f"Feature file not found for visitor {visitor_id}, skipping")
                    continue
                
                features = np.load(feature_path)
                
                # 创建访客记录
                visitor = VisitorRecord.from_dict(visitor_data, features)
                self.visitors.append(visitor)
            
            logger.info(f"Loaded {len(self.visitors)} visitors from disk")
            
        except Exception as e:
            logger.error(f"Failed to load database from disk: {e}")
            self.visitors = []
    
    def save_to_disk(self):
        """将访客数据保存到磁盘"""
        try:
            metadata = {
                "visitors": [visitor.to_dict() for visitor in self.visitors],
                "total_count": len(self.visitors)
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved {len(self.visitors)} visitors to disk")
            
        except Exception as e:
            logger.error(f"Failed to save database to disk: {e}")
            raise
