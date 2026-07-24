"""
测试ATSP问题集成
"""
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.problems import get_problem_class, get_problem_info
from modules.compatibility import validate_combination, get_available_policies, get_recommended_combination


class TestATSPProblem:
    """测试ATSP问题类"""
    
    def test_atsp_registered(self):
        """测试ATSP是否正确注册"""
        info = get_problem_info('atsp')
        assert info is not None
        assert info['name'] == 'ATSP'
        assert info['cn_name'] == '非对称旅行商问题'
        assert info['status'] == 'active'
        assert info['difficulty'] == 'hard'
    
    def test_atsp_class_creation(self):
        """测试ATSP类创建"""
        ATSPClass = get_problem_class('atsp')
        assert ATSPClass is not None
        
        # 创建实例
        atsp = ATSPClass({'num_loc': 50})
        assert atsp.get_problem_type() == 'atsp'
        assert atsp.get_problem_name() == 'Asymmetric Traveling Salesman Problem'
        assert atsp.num_loc == 50
    
    def test_atsp_validation(self):
        """测试ATSP参数验证"""
        ATSPClass = get_problem_class('atsp')
        
        # 有效配置
        atsp_valid = ATSPClass({'num_loc': 50})
        valid, msg = atsp_valid.validate_config()
        assert valid is True
        
        # 城市数量过少
        atsp_too_small = ATSPClass({'num_loc': 3})
        valid, msg = atsp_too_small.validate_config()
        assert valid is False
        assert '至少5个' in msg
        
        # 城市数量过多
        atsp_too_large = ATSPClass({'num_loc': 1500})
        valid, msg = atsp_too_large.validate_config()
        assert valid is False
        assert '不建议超过1000' in msg


class TestATSPCompatibility:
    """测试ATSP兼容性"""
    
    def test_atsp_with_matnet_reinforce(self):
        """测试ATSP + MatNet + REINFORCE（唯一支持配置）"""
        valid, msg, level = validate_combination('atsp', 'matnet', 'reinforce')
        assert valid is True
        assert level in ['success', 'info']
    
    def test_atsp_with_pomo_blocked(self):
        """测试ATSP + POMO被阻止"""
        valid, msg, level = validate_combination('atsp', 'pomo', 'ppo')
        assert valid is False or level == 'error'
        assert 'POMO' in msg
    
    def test_atsp_with_attention_blocked(self):
        """测试ATSP拒绝依赖locs的Attention Model"""
        valid, msg, level = validate_combination('atsp', 'attention', 'reinforce')
        assert valid is False
        assert level == 'error'
        assert 'matnet' in msg.lower()
    
    def test_atsp_available_policies(self):
        """测试ATSP仅开放能够处理cost_matrix的MatNet"""
        policies = get_available_policies('atsp')
        assert policies == ['matnet']
    
    def test_atsp_recommended_config(self):
        """测试ATSP推荐配置"""
        recommended = get_recommended_combination('atsp', 'best')
        assert recommended['policy'] == 'matnet'
        assert recommended['algorithm'] == 'reinforce'
        
        recommended_fast = get_recommended_combination('atsp', 'fast')
        assert recommended_fast['policy'] == 'matnet'
        assert recommended_fast['algorithm'] == 'reinforce'


class TestATSPFeatures:
    """测试ATSP特性"""
    
    def test_atsp_features(self):
        """测试ATSP特征列表"""
        ATSPClass = get_problem_class('atsp')
        atsp = ATSPClass({'num_loc': 50})
        features = atsp.get_problem_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
        # 应该包含非对称相关特征
        features_text = ' '.join(features)
        assert '不对称' in features_text or 'asymmetric' in features_text.lower()
    
    def test_atsp_description(self):
        """测试ATSP描述"""
        ATSPClass = get_problem_class('atsp')
        atsp = ATSPClass({'num_loc': 50})
        desc = atsp.get_problem_description()
        
        assert isinstance(desc, str)
        assert 'ATSP' in desc or '非对称' in desc
        assert '50' in desc  # 应该包含城市数量


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
