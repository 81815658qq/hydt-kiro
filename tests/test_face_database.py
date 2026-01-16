"""人脸数据库模块单元测试"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from src.face_database import FaceDatabase, VisitorRecord


# ===== VisitorRecord 测试 =====

def test_visitor_record_creation():
    """测试创建访客记录"""
    features = np.random.rand(128)
    visitor = VisitorRecord(
        visitor_id="test-id-123",
        features=features,
        face_image_path="images/test-id-123.jpg",
        first_seen=datetime.now(),
        blessing="鸿运当头"
    )
    
    assert visitor.visitor_id == "test-id-123"
    assert visitor.blessing == "鸿运当头"
    assert len(visitor.features) == 128
    assert visitor.face_image_path == "images/test-id-123.jpg"


def test_visitor_record_to_dict():
    """测试序列化为字典"""
    features = np.random.rand(128)
    now = datetime.now()
    visitor = VisitorRecord(
        visitor_id="test-id-456",
        features=features,
        face_image_path="images/test-id-456.jpg",
        first_seen=now,
        blessing="福星高照"
    )
    
    data = visitor.to_dict()
    
    assert data["visitor_id"] == "test-id-456"
    assert data["blessing"] == "福星高照"
    assert data["face_image_path"] == "images/test-id-456.jpg"
    assert "first_seen" in data
    assert "features" not in data


def test_visitor_record_from_dict():
    """测试从字典反序列化"""
    features = np.random.rand(128)
    data = {
        "visitor_id": "test-id-789",
        "face_image_path": "images/test-id-789.jpg",
        "first_seen": "2026-01-15T10:30:00",
        "blessing": "吉祥如意"
    }
    
    visitor = VisitorRecord.from_dict(data, features)
    
    assert visitor.visitor_id == "test-id-789"
    assert visitor.blessing == "吉祥如意"
    assert len(visitor.features) == 128
    assert isinstance(visitor.first_seen, datetime)


# ===== FaceDatabase 测试 =====

@pytest.fixture
def temp_db_dir():
    """创建临时数据库目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_face_image():
    """创建示例人脸图像"""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_features():
    """创建示例特征向量"""
    features = np.random.rand(128)
    features = features / np.linalg.norm(features)
    return features


def test_database_initialization(temp_db_dir):
    """测试数据库初始化"""
    db = FaceDatabase(storage_dir=temp_db_dir)
    
    assert db.storage_dir == Path(temp_db_dir)
    assert db.features_dir.exists()
    assert db.images_dir.exists()
    assert db.get_total_visitors() == 0


def test_add_new_visitor(temp_db_dir, sample_face_image, sample_features):
    """测试添加新访客"""
    db = FaceDatabase(storage_dir=temp_db_dir)
    
    visitor = db.add_visitor(
        features=sample_features,
        face_image=sample_face_image,
        blessing="鸿运当头"
    )
    
    assert visitor is not None
    assert visitor.blessing == "鸿运当头"
    assert db.get_total_visitors() == 1
    
    feature_file = db.features_dir / f"{visitor.visitor_id}.npy"
    image_file = db.images_dir / f"{visitor.visitor_id}.jpg"
    assert feature_file.exists()
    assert image_file.exists()


def test_find_matching_visitor_exists(temp_db_dir, sample_face_image, sample_features):
    """测试查找已存在的访客"""
    db = FaceDatabase(storage_dir=temp_db_dir)
    
    original_visitor = db.add_visitor(
        features=sample_features,
        face_image=sample_face_image,
        blessing="福星高照"
    )
    
    found_visitor = db.find_matching_visitor(sample_features, threshold=0.7)
    
    assert found_visitor is not None
    assert found_visitor.visitor_id == original_visitor.visitor_id
    assert found_visitor.blessing == "福星高照"


def test_empty_database_find(temp_db_dir, sample_features):
    """测试在空数据库中查找"""
    db = FaceDatabase(storage_dir=temp_db_dir)
    
    found_visitor = db.find_matching_visitor(sample_features, threshold=0.7)
    
    assert found_visitor is None


def test_save_and_load_database(temp_db_dir, sample_face_image, sample_features):
    """测试保存和加载数据库"""
    db1 = FaceDatabase(storage_dir=temp_db_dir)
    visitor1 = db1.add_visitor(
        features=sample_features,
        face_image=sample_face_image,
        blessing="万事如意"
    )
    
    db2 = FaceDatabase(storage_dir=temp_db_dir)
    
    assert db2.get_total_visitors() == 1
    
    loaded_visitor = db2.visitors[0]
    assert loaded_visitor.visitor_id == visitor1.visitor_id
    assert loaded_visitor.blessing == "万事如意"
    assert len(loaded_visitor.features) == 128


def test_multiple_visitors(temp_db_dir, sample_face_image):
    """测试添加多个访客"""
    db = FaceDatabase(storage_dir=temp_db_dir)
    
    blessings = ["鸿运当头", "福星高照", "吉祥如意"]
    
    for blessing in blessings:
        features = np.random.rand(128)
        features = features / np.linalg.norm(features)
        db.add_visitor(
            features=features,
            face_image=sample_face_image,
            blessing=blessing
        )
    
    assert db.get_total_visitors() == 3
    
    saved_blessings = [v.blessing for v in db.visitors]
    assert set(saved_blessings) == set(blessings)
