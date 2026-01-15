"""祝福语生成模块单元测试"""
import pytest
from src.blessing_generator import BlessingGenerator


def test_blessing_generator_initialization():
    """测试祝福语生成器初始化"""
    generator = BlessingGenerator()
    
    assert generator is not None
    assert len(generator.BLESSINGS) >= 10


def test_blessings_are_four_characters():
    """验证祝福语列表包含至少10个四字词语"""
    generator = BlessingGenerator()
    
    # 验证至少有10个祝福语
    assert len(generator.BLESSINGS) >= 10, "祝福语列表应该包含至少10个词语"
    
    # 验证所有祝福语都是四个字
    for blessing in generator.BLESSINGS:
        assert len(blessing) == 4, f"祝福语 '{blessing}' 应该是四个字"


def test_same_visitor_id_returns_same_blessing():
    """测试相同访客ID返回相同祝福语"""
    generator = BlessingGenerator()
    
    visitor_id = "test-visitor-123"
    
    # 多次调用应该返回相同的祝福语
    blessing1 = generator.get_blessing_for_visitor(visitor_id)
    blessing2 = generator.get_blessing_for_visitor(visitor_id)
    blessing3 = generator.get_blessing_for_visitor(visitor_id)
    
    assert blessing1 == blessing2, "相同访客ID应该返回相同祝福语"
    assert blessing2 == blessing3, "相同访客ID应该返回相同祝福语"


def test_different_visitor_ids_may_return_different_blessings():
    """测试不同访客ID可能返回不同祝福语"""
    generator = BlessingGenerator()
    
    visitor_id1 = "visitor-001"
    visitor_id2 = "visitor-002"
    visitor_id3 = "visitor-003"
    
    blessing1 = generator.get_blessing_for_visitor(visitor_id1)
    blessing2 = generator.get_blessing_for_visitor(visitor_id2)
    blessing3 = generator.get_blessing_for_visitor(visitor_id3)
    
    # 至少验证返回的都是有效的祝福语
    assert blessing1 in generator.BLESSINGS
    assert blessing2 in generator.BLESSINGS
    assert blessing3 in generator.BLESSINGS
    
    # 注意：由于哈希冲突，不同ID可能返回相同祝福语，所以不强制要求不同


def test_get_all_blessings():
    """测试获取所有祝福语"""
    generator = BlessingGenerator()
    
    all_blessings = generator.get_all_blessings()
    
    assert len(all_blessings) == len(generator.BLESSINGS)
    assert all_blessings == generator.BLESSINGS


def test_get_blessing_count():
    """测试获取祝福语数量"""
    generator = BlessingGenerator()
    
    count = generator.get_blessing_count()
    
    assert count == len(generator.BLESSINGS)
    assert count >= 10


def test_get_random_blessing():
    """测试获取随机祝福语"""
    generator = BlessingGenerator()
    
    # 获取多个随机祝福语
    blessings = [generator.get_random_blessing() for _ in range(10)]
    
    # 验证所有返回的都是有效的祝福语
    for blessing in blessings:
        assert blessing in generator.BLESSINGS


def test_blessing_consistency_across_instances():
    """测试不同实例对相同ID返回相同祝福语"""
    generator1 = BlessingGenerator()
    generator2 = BlessingGenerator()
    
    visitor_id = "consistent-visitor"
    
    blessing1 = generator1.get_blessing_for_visitor(visitor_id)
    blessing2 = generator2.get_blessing_for_visitor(visitor_id)
    
    assert blessing1 == blessing2, "不同实例应该为相同ID返回相同祝福语"


def test_blessing_distribution():
    """测试祝福语分配的分布性"""
    generator = BlessingGenerator()
    
    # 生成100个不同的访客ID
    visitor_ids = [f"visitor-{i:03d}" for i in range(100)]
    
    # 获取每个ID的祝福语
    blessings = [generator.get_blessing_for_visitor(vid) for vid in visitor_ids]
    
    # 统计不同祝福语的数量
    unique_blessings = set(blessings)
    
    # 验证至少使用了多个不同的祝福语（不是所有人都得到同一个）
    assert len(unique_blessings) > 1, "应该分配多种不同的祝福语"
