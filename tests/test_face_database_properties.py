"""人脸数据库模块属性测试（Property-Based Testing）
Feature: welcome-greeter
使用Hypothesis进行基于属性的测试
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, settings, strategies as st
from src.face_database import FaceDatabase, VisitorRecord


# 策略：生成有效的特征向量
@st.composite
def feature_vectors(draw):
    """生成归一化的特征向量（非零向量）"""
    features = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=128,
        max_size=128
    ))
    features = np.array(features)
    norm = np.linalg.norm(features)
    
    # 确保不生成零向量（在实际应用中不会出现）
    # 如果生成了零向量，用一个小的随机向量替代
    if norm < 1e-6:
        features = np.random.rand(128) * 0.1 + 0.01
        norm = np.linalg.norm(features)
    
    # 归一化
    features = features / norm
    return features


# 策略：生成祝福语
blessings_strategy = st.sampled_from([
    "鸿运当头", "福星高照", "吉祥如意", "万事如意",
    "心想事成", "步步高升", "财源广进", "喜气洋洋"
])


class TestFaceDatabasePersistenceProperties:
    """属性 7：新访客数据持久化
    验证需求：4.5
    """
    
    @settings(max_examples=100)
    @given(
        features=feature_vectors(),
        blessing=blessings_strategy
    )
    def test_property_7_new_visitor_persistence(self, features, blessing):
        """
        属性 7：新访客数据持久化
        
        对于任何新访客，添加到数据库后应该能够通过特征匹配找回，
        且保存的人脸图片和特征数据应该存在于文件系统中。
        
        **Validates: Requirements 4.5**
        """
        temp_db_dir = tempfile.mkdtemp()
        try:
            db = FaceDatabase(storage_dir=temp_db_dir)
            
            # 创建测试图像
            face_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # 添加新访客
            visitor = db.add_visitor(
                features=features,
                face_image=face_image,
                blessing=blessing
            )
            
            # 属性1：访客记录应该被创建
            assert visitor is not None
            assert visitor.visitor_id is not None
            assert visitor.blessing == blessing
            
            # 属性2：特征文件应该存在
            feature_file = db.features_dir / f"{visitor.visitor_id}.npy"
            assert feature_file.exists(), "特征文件应该存在于文件系统中"
            
            # 属性3：图片文件应该存在
            image_file = db.images_dir / f"{visitor.visitor_id}.jpg"
            assert image_file.exists(), "图片文件应该存在于文件系统中"
            
            # 属性4：应该能够通过特征匹配找回访客
            found_visitor = db.find_matching_visitor(features, threshold=0.7)
            assert found_visitor is not None, "应该能够通过特征匹配找回访客"
            assert found_visitor.visitor_id == visitor.visitor_id
            
            # 属性5：加载的特征应该与原始特征相似
            loaded_features = np.load(feature_file)
            similarity_check = np.dot(features, loaded_features)
            assert similarity_check > 0.95, "加载的特征应该与原始特征高度相似"
            
        finally:
            shutil.rmtree(temp_db_dir, ignore_errors=True)


class TestVisitorCountUniquenessProperties:
    """属性 8：访客计数唯一性
    验证需求：5.1, 5.2
    """
    
    @settings(max_examples=50)
    @given(
        num_unique_visitors=st.integers(min_value=1, max_value=5),
        num_repeats=st.integers(min_value=0, max_value=3)
    )
    def test_property_8_visitor_count_uniqueness(self, num_unique_visitors, num_repeats):
        """
        属性 8：访客计数唯一性
        
        对于任何访客序列（包含新访客和重复访客），
        系统的总访客计数应该等于唯一访客的数量。
        
        **Validates: Requirements 5.1, 5.2**
        """
        temp_db_dir = tempfile.mkdtemp()
        try:
            db = FaceDatabase(storage_dir=temp_db_dir)
            
            # 创建唯一访客的特征
            unique_features = []
            for i in range(num_unique_visitors):
                features = np.random.rand(128)
                features = features / np.linalg.norm(features)
                unique_features.append(features)
            
            # 添加唯一访客
            face_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            for i, features in enumerate(unique_features):
                db.add_visitor(
                    features=features,
                    face_image=face_image,
                    blessing=f"祝福{i}"
                )
            
            # 属性1：初始计数应该等于唯一访客数
            assert db.get_total_visitors() == num_unique_visitors, \
                f"初始计数应该等于唯一访客数 {num_unique_visitors}"
            
            # 尝试重复添加（通过查找匹配）
            repeat_count = 0
            for _ in range(num_repeats):
                if unique_features:
                    # 随机选择一个已存在的特征
                    existing_features = unique_features[0]
                    found = db.find_matching_visitor(existing_features, threshold=0.7)
                    if found:
                        repeat_count += 1
            
            # 属性2：重复访客不应该增加计数
            assert db.get_total_visitors() == num_unique_visitors, \
                f"重复访客不应该增加计数，应该保持 {num_unique_visitors}"
        
        finally:
            shutil.rmtree(temp_db_dir, ignore_errors=True)
    
    @settings(max_examples=50)
    @given(
        num_visitors=st.integers(min_value=1, max_value=10)
    )
    def test_property_8_unique_visitor_ids(self, num_visitors):
        """
        属性 8（扩展）：访客ID唯一性
        
        对于任何数量的新访客，每个访客应该有唯一的ID。
        
        **Validates: Requirements 5.1**
        """
        temp_db_dir = tempfile.mkdtemp()
        try:
            db = FaceDatabase(storage_dir=temp_db_dir)
            
            visitor_ids = set()
            face_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            for i in range(num_visitors):
                features = np.random.rand(128)
                features = features / np.linalg.norm(features)
                
                visitor = db.add_visitor(
                    features=features,
                    face_image=face_image,
                    blessing=f"祝福{i}"
                )
                
                visitor_ids.add(visitor.visitor_id)
            
            # 属性：所有访客ID应该唯一
            assert len(visitor_ids) == num_visitors, \
                f"所有访客ID应该唯一，期望 {num_visitors} 个唯一ID"
        
        finally:
            shutil.rmtree(temp_db_dir, ignore_errors=True)


class TestDataPersistenceRoundTripProperties:
    """属性 9：数据持久化往返一致性
    验证需求：5.4
    """
    
    @settings(max_examples=50)
    @given(
        num_visitors=st.integers(min_value=1, max_value=5)
    )
    def test_property_9_persistence_round_trip(self, num_visitors):
        """
        属性 9：数据持久化往返一致性
        
        对于任何访客记录集合，保存到磁盘后再加载应该得到等价的数据
        （特征向量、元数据、图片路径）。
        
        **Validates: Requirements 5.4**
        """
        temp_db_dir = tempfile.mkdtemp()
        try:
            # 第一个数据库实例：添加访客
            db1 = FaceDatabase(storage_dir=temp_db_dir)
            
            original_visitors = []
            face_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            for i in range(num_visitors):
                features = np.random.rand(128)
                features = features / np.linalg.norm(features)
                
                visitor = db1.add_visitor(
                    features=features,
                    face_image=face_image,
                    blessing=f"祝福{i}"
                )
                original_visitors.append(visitor)
            
            # 第二个数据库实例：从磁盘加载
            db2 = FaceDatabase(storage_dir=temp_db_dir)
            
            # 属性1：访客数量应该一致
            assert db2.get_total_visitors() == num_visitors, \
                f"加载后的访客数量应该等于原始数量 {num_visitors}"
            
            # 属性2：每个访客的数据应该一致
            for original in original_visitors:
                # 通过ID查找
                loaded = None
                for v in db2.visitors:
                    if v.visitor_id == original.visitor_id:
                        loaded = v
                        break
                
                assert loaded is not None, f"访客 {original.visitor_id} 应该被加载"
                
                # 验证元数据
                assert loaded.blessing == original.blessing, "祝福语应该一致"
                assert loaded.face_image_path == original.face_image_path, "图片路径应该一致"
                
                # 验证特征向量
                assert len(loaded.features) == len(original.features), "特征向量维度应该一致"
                
                # 特征向量应该非常相似（允许浮点误差）
                similarity = np.dot(loaded.features, original.features)
                assert similarity > 0.999, \
                    f"加载的特征向量应该与原始特征向量几乎相同，相似度: {similarity}"
        
        finally:
            shutil.rmtree(temp_db_dir, ignore_errors=True)
    
    @settings(max_examples=30)
    @given(
        num_visitors=st.integers(min_value=1, max_value=3)
    )
    def test_property_9_metadata_consistency(self, num_visitors):
        """
        属性 9（扩展）：元数据文件一致性
        
        对于任何访客集合，元数据文件中的访客数量应该与实际访客数量一致。
        
        **Validates: Requirements 5.4**
        """
        temp_db_dir = tempfile.mkdtemp()
        try:
            db = FaceDatabase(storage_dir=temp_db_dir)
            
            face_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            for i in range(num_visitors):
                features = np.random.rand(128)
                features = features / np.linalg.norm(features)
                
                db.add_visitor(
                    features=features,
                    face_image=face_image,
                    blessing=f"祝福{i}"
                )
            
            # 读取元数据文件
            import json
            with open(db.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 属性：元数据中的计数应该与实际一致
            assert metadata["total_count"] == num_visitors, \
                "元数据中的total_count应该与实际访客数量一致"
            assert len(metadata["visitors"]) == num_visitors, \
                "元数据中的visitors列表长度应该与实际访客数量一致"
        
        finally:
            shutil.rmtree(temp_db_dir, ignore_errors=True)
