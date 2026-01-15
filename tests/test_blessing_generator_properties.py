"""祝福语生成模块属性测试（Property-Based Testing）
Feature: welcome-greeter
使用Hypothesis进行基于属性的测试
"""
import pytest
from hypothesis import given, settings, strategies as st
from src.blessing_generator import BlessingGenerator


class TestBlessingAssignmentConsistencyProperties:
    """属性 3：祝福语分配一致性
    验证需求：3.5
    """
    
    @settings(max_examples=100)
    @given(
        visitor_id=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            min_codepoint=33,
            max_codepoint=126
        ))
    )
    def test_property_3_blessing_assignment_consistency(self, visitor_id):
        """
        属性 3：祝福语分配一致性
        
        对于任何访客ID，多次调用祝福语生成器应该返回相同的祝福语（幂等性）。
        
        **Validates: Requirements 3.5**
        """
        generator = BlessingGenerator()
        
        # 多次获取同一个访客ID的祝福语
        blessing1 = generator.get_blessing_for_visitor(visitor_id)
        blessing2 = generator.get_blessing_for_visitor(visitor_id)
        blessing3 = generator.get_blessing_for_visitor(visitor_id)
        
        # 属性1：相同ID应该返回相同祝福语（幂等性）
        assert blessing1 == blessing2, \
            f"相同访客ID '{visitor_id}' 应该返回相同祝福语"
        assert blessing2 == blessing3, \
            f"相同访客ID '{visitor_id}' 应该返回相同祝福语"
        
        # 属性2：返回的祝福语应该在祝福语列表中
        assert blessing1 in generator.BLESSINGS, \
            f"返回的祝福语 '{blessing1}' 应该在祝福语列表中"
        
        # 属性3：返回的祝福语应该是四个字
        assert len(blessing1) == 4, \
            f"返回的祝福语 '{blessing1}' 应该是四个字"
    
    @settings(max_examples=100)
    @given(
        visitor_id=st.text(min_size=1, max_size=100, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            min_codepoint=33,
            max_codepoint=126
        ))
    )
    def test_property_3_consistency_across_instances(self, visitor_id):
        """
        属性 3（扩展）：跨实例一致性
        
        对于任何访客ID，不同的生成器实例应该返回相同的祝福语。
        
        **Validates: Requirements 3.5**
        """
        generator1 = BlessingGenerator()
        generator2 = BlessingGenerator()
        
        blessing1 = generator1.get_blessing_for_visitor(visitor_id)
        blessing2 = generator2.get_blessing_for_visitor(visitor_id)
        
        # 属性：不同实例应该为相同ID返回相同祝福语
        assert blessing1 == blessing2, \
            f"不同实例应该为访客ID '{visitor_id}' 返回相同祝福语"
    
    @settings(max_examples=50)
    @given(
        num_calls=st.integers(min_value=2, max_value=20)
    )
    def test_property_3_idempotence_multiple_calls(self, num_calls):
        """
        属性 3（扩展）：多次调用幂等性
        
        对于任何访客ID，无论调用多少次，都应该返回相同的祝福语。
        
        **Validates: Requirements 3.5**
        """
        generator = BlessingGenerator()
        visitor_id = "test-visitor-idempotence"
        
        # 多次调用
        blessings = [generator.get_blessing_for_visitor(visitor_id) for _ in range(num_calls)]
        
        # 属性：所有调用应该返回相同的祝福语
        unique_blessings = set(blessings)
        assert len(unique_blessings) == 1, \
            f"多次调用应该返回相同祝福语，但得到了 {len(unique_blessings)} 个不同的祝福语"
    
    @settings(max_examples=50)
    @given(
        num_visitors=st.integers(min_value=10, max_value=100)
    )
    def test_property_3_distribution_coverage(self, num_visitors):
        """
        属性 3（扩展）：祝福语分布覆盖
        
        对于任何数量的不同访客，应该使用多种不同的祝福语（不是所有人都得到同一个）。
        
        **Validates: Requirements 3.2**
        """
        generator = BlessingGenerator()
        
        # 生成不同的访客ID
        visitor_ids = [f"visitor-{i:05d}" for i in range(num_visitors)]
        
        # 获取每个访客的祝福语
        blessings = [generator.get_blessing_for_visitor(vid) for vid in visitor_ids]
        
        # 属性：应该使用多种不同的祝福语
        unique_blessings = set(blessings)
        
        # 至少应该有2种不同的祝福语（理想情况下应该更多）
        assert len(unique_blessings) >= 2, \
            f"对于 {num_visitors} 个访客，应该使用多种祝福语，但只使用了 {len(unique_blessings)} 种"
        
        # 所有祝福语都应该在列表中
        for blessing in unique_blessings:
            assert blessing in generator.BLESSINGS, \
                f"祝福语 '{blessing}' 应该在祝福语列表中"
    
    @settings(max_examples=30)
    @given(
        visitor_id1=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            min_codepoint=33,
            max_codepoint=126
        )),
        visitor_id2=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            min_codepoint=33,
            max_codepoint=126
        ))
    )
    def test_property_3_deterministic_mapping(self, visitor_id1, visitor_id2):
        """
        属性 3（扩展）：确定性映射
        
        对于任何两个访客ID，如果ID相同则祝福语相同，如果ID不同则可能不同。
        
        **Validates: Requirements 3.5**
        """
        generator = BlessingGenerator()
        
        blessing1 = generator.get_blessing_for_visitor(visitor_id1)
        blessing2 = generator.get_blessing_for_visitor(visitor_id2)
        
        if visitor_id1 == visitor_id2:
            # 属性：相同ID应该返回相同祝福语
            assert blessing1 == blessing2, \
                "相同的访客ID应该返回相同的祝福语"
        # 注意：不同ID可能返回相同祝福语（哈希冲突），所以不验证不等性
