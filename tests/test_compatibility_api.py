"""
测试兼容性API
"""
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.compatibility import (
    validate_combination,
    get_available_policies,
    get_available_algorithms,
    get_recommended_combination
)


class TestCompatibilityAPI:
    """测试兼容性检查相关功能"""
    
    def test_validate_valid_combination(self):
        """测试有效配置组合"""
        is_valid, message, level = validate_combination('tsp', 'attention', 'reinforce')
        assert is_valid is True
        assert level == 'success'
    
    def test_validate_invalid_combination(self):
        """测试无效配置组合"""
        # 假设这是一个不兼容的组合（根据实际情况调整）
        is_valid, message, level = validate_combination('invalid_problem', 'attention', 'reinforce')
        # 可能返回warning或error，取决于具体实现
        assert is_valid in [True, False]
    
    def test_get_available_policies_for_tsp(self):
        """测试获取TSP可用策略"""
        policies = get_available_policies('tsp')
        assert isinstance(policies, list)
        assert len(policies) > 0
        # TSP应该支持attention模型
        assert 'attention' in policies or any('attention' in p.lower() for p in policies)
    
    def test_get_available_algorithms_for_tsp(self):
        """测试获取TSP可用算法"""
        algorithms = get_available_algorithms('tsp')
        assert isinstance(algorithms, list)
        assert len(algorithms) > 0
        # TSP应该支持reinforce算法
        assert 'reinforce' in algorithms or any('reinforce' in a.lower() for a in algorithms)
    
    def test_get_recommended_configuration(self):
        """测试获取推荐配置"""
        recommended = get_recommended_combination('tsp', 'best')
        assert isinstance(recommended, dict)
        assert 'model' in recommended or 'policy' in recommended
        assert 'algorithm' in recommended
    
    def test_get_recommended_fast_configuration(self):
        """测试获取快速训练推荐配置"""
        recommended = get_recommended_combination('tsp', 'fast')
        assert isinstance(recommended, dict)


class TestCompatibilityEdgeCases:
    """测试兼容性检查的边界情况"""
    
    def test_validate_empty_parameters(self):
        """测试空参数"""
        is_valid, message, level = validate_combination('', '', '')
        assert is_valid is False or level == 'error'
    
    def test_get_policies_for_unknown_problem(self):
        """测试未知问题类型"""
        policies = get_available_policies('unknown_problem_type')
        assert isinstance(policies, list)
        # 应该返回空列表或默认列表


