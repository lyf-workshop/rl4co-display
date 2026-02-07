"""
mTSP 集成测试
验证 mTSP 问题类型是否正确集成到系统中
"""

import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.problems import get_problem_class, PROBLEM_REGISTRY, get_problem_info


class TestMTSPIntegration:
    """mTSP 集成测试类"""
    
    def test_mtsp_in_registry(self):
        """测试 mTSP 是否在问题注册表中"""
        assert 'mtsp' in PROBLEM_REGISTRY, "mTSP 未在 PROBLEM_REGISTRY 中注册"
    
    def test_mtsp_info_exists(self):
        """测试 mTSP 元信息是否存在"""
        info = get_problem_info('mtsp')
        assert info is not None, "mTSP 元信息不存在"
        assert info['name'] == 'mTSP'
        assert info['full_name'] == 'Multiple Traveling Salesman Problem'
        assert info['cn_name'] == '多旅行商问题'
        assert info['category'] == 'routing'
        assert info['status'] == 'active'
    
    def test_mtsp_problem_creation(self):
        """测试 mTSP 问题实例创建"""
        config = {
            'problem_type': 'mtsp',
            'num_loc': 20,
            'num_agents': 3,
            'cost_type': 'minmax',
            'batch_size': 128,
        }
        
        problem_class = get_problem_class('mtsp')
        problem = problem_class(config)
        
        assert problem.get_problem_type() == 'mtsp'
        assert problem.get_problem_name() == 'Multiple Traveling Salesman Problem'
        assert problem.num_loc == 20
        assert problem.num_agents == 3
        assert problem.cost_type == 'minmax'
    
    def test_mtsp_parameter_validation(self):
        """测试 mTSP 参数验证"""
        # 有效配置
        valid_config = {
            'problem_type': 'mtsp',
            'num_loc': 20,
            'num_agents': 5,
            'cost_type': 'minmax',
        }
        
        problem_class = get_problem_class('mtsp')
        problem = problem_class(valid_config)
        valid, msg = problem.validate_config()
        assert valid, f"有效配置验证失败: {msg}"
        
        # 无效配置1: 代理数超过城市数
        invalid_config1 = {
            'problem_type': 'mtsp',
            'num_loc': 10,
            'num_agents': 15,  # > num_loc
            'cost_type': 'minmax',
        }
        problem1 = problem_class(invalid_config1)
        valid1, msg1 = problem1.validate_config()
        assert not valid1, "应该检测到代理数超过城市数的错误"
        
        # 无效配置2: 无效的cost_type
        invalid_config2 = {
            'problem_type': 'mtsp',
            'num_loc': 20,
            'num_agents': 5,
            'cost_type': 'invalid_type',
        }
        problem2 = problem_class(invalid_config2)
        valid2, msg2 = problem2.validate_config()
        assert not valid2, "应该检测到无效的cost_type"
        
        # 无效配置3: 代理数为0
        invalid_config3 = {
            'problem_type': 'mtsp',
            'num_loc': 20,
            'num_agents': 0,
            'cost_type': 'minmax',
        }
        problem3 = problem_class(invalid_config3)
        valid3, msg3 = problem3.validate_config()
        assert not valid3, "应该检测到代理数为0的错误"
    
    def test_mtsp_cost_types(self):
        """测试 mTSP 的两种成本类型"""
        problem_class = get_problem_class('mtsp')
        
        # minmax
        config_minmax = {
            'problem_type': 'mtsp',
            'num_loc': 20,
            'num_agents': 3,
            'cost_type': 'minmax',
        }
        problem_minmax = problem_class(config_minmax)
        assert problem_minmax.cost_type == 'minmax'
        
        # sum
        config_sum = {
            'problem_type': 'mtsp',
            'num_loc': 20,
            'num_agents': 3,
            'cost_type': 'sum',
        }
        problem_sum = problem_class(config_sum)
        assert problem_sum.cost_type == 'sum'
    
    def test_mtsp_env_params(self):
        """测试 mTSP 环境参数生成"""
        config = {
            'problem_type': 'mtsp',
            'num_loc': 50,
            'num_agents': 5,
            'cost_type': 'minmax',
        }
        
        problem_class = get_problem_class('mtsp')
        problem = problem_class(config)
        env_params = problem.get_env_params()
        
        assert 'num_loc' in env_params
        assert 'min_num_agents' in env_params
        assert 'max_num_agents' in env_params
        assert env_params['num_loc'] == 50
        assert env_params['min_num_agents'] == 5
        assert env_params['max_num_agents'] == 5
    
    def test_mtsp_problem_description(self):
        """测试 mTSP 问题描述"""
        config = {
            'problem_type': 'mtsp',
            'num_loc': 30,
            'num_agents': 4,
            'cost_type': 'minmax',
        }
        
        problem_class = get_problem_class('mtsp')
        problem = problem_class(config)
        description = problem.get_problem_description()
        
        assert '多旅行商问题' in description
        assert '30' in description
        assert '4' in description
    
    def test_mtsp_problem_features(self):
        """测试 mTSP 问题特征"""
        config = {
            'problem_type': 'mtsp',
            'num_loc': 20,
            'num_agents': 3,
            'cost_type': 'minmax',
        }
        
        problem_class = get_problem_class('mtsp')
        problem = problem_class(config)
        features = problem.get_problem_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
        # 检查关键特征
        assert any('代理' in str(f) for f in features)
        assert any('起点' in str(f) or 'depot' in str(f).lower() for f in features)
    
    def test_mtsp_visualization_functions(self):
        """测试 mTSP 可视化函数获取"""
        config = {
            'problem_type': 'mtsp',
            'num_loc': 20,
            'num_agents': 3,
            'cost_type': 'minmax',
        }
        
        problem_class = get_problem_class('mtsp')
        problem = problem_class(config)
        viz_funcs = problem.get_visualization_functions()
        
        assert isinstance(viz_funcs, dict)
        assert 'animation' in viz_funcs
        assert 'comparison' in viz_funcs
        assert callable(viz_funcs['animation'])
        assert callable(viz_funcs['comparison'])


class TestMTSPVisualization:
    """mTSP 可视化函数测试"""
    
    def test_extract_agent_routes(self):
        """测试代理路径提取函数"""
        from modules.rl_training.visualizations.mtsp_viz import extract_agent_routes
        
        # 测试案例1: 3个代理
        actions1 = [1, 2, 0, 3, 4, 0, 5, 6, 0]
        routes1 = extract_agent_routes(actions1)
        assert len(routes1) == 3
        assert routes1[0] == [1, 2]
        assert routes1[1] == [3, 4]
        assert routes1[2] == [5, 6]
        
        # 测试案例2: 2个代理，不均衡
        actions2 = [1, 0, 2, 3, 4, 0]
        routes2 = extract_agent_routes(actions2)
        assert len(routes2) == 2
        assert routes2[0] == [1]
        assert routes2[1] == [2, 3, 4]
        
        # 测试案例3: 单个代理
        actions3 = [1, 2, 3, 0]
        routes3 = extract_agent_routes(actions3)
        assert len(routes3) == 1
        assert routes3[0] == [1, 2, 3]
    
    def test_calculate_route_distance(self):
        """测试路径距离计算函数"""
        import numpy as np
        from modules.rl_training.visualizations.mtsp_viz import calculate_route_distance
        
        # 创建简单的测试坐标
        locs = np.array([
            [0.0, 0.0],  # depot
            [1.0, 0.0],  # city 1
            [0.0, 1.0],  # city 2
            [1.0, 1.0],  # city 3
        ])
        
        # 测试路径: depot -> 1 -> 2 -> depot
        route = [1, 2]
        dist = calculate_route_distance(locs, route, include_depot=True)
        expected = 1.0 + np.sqrt(2) + 1.0  # depot->1 + 1->2 + 2->depot
        assert abs(dist - expected) < 1e-6
        
        # 测试不包含depot的距离
        dist_no_depot = calculate_route_distance(locs, route, include_depot=False)
        expected_no_depot = np.sqrt(2)  # 只计算 1->2
        assert abs(dist_no_depot - expected_no_depot) < 1e-6


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])
