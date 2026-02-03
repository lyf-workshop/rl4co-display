"""
强化学习算法模块
提供统一的算法接口，支持多种RL算法（REINFORCE、PPO、A2C等）

使用方式:
    from modules.algorithms import get_algorithm_class, ALGORITHM_REGISTRY
    
    # 获取算法类
    AlgorithmClass = get_algorithm_class('reinforce')
    algorithm = AlgorithmClass(env, policy, config)
"""

from .base_algorithm import BaseAlgorithm
from .reinforce_algo import REINFORCEAlgorithm
from .ppo_algo import PPOAlgorithm
from .a2c_algo import A2CAlgorithm

# 算法注册表
ALGORITHM_REGISTRY = {
    'reinforce': REINFORCEAlgorithm,
    'ppo': PPOAlgorithm,
    'a2c': A2CAlgorithm,
    # 未来扩展：
    # 'dqn': DQNAlgorithm,
    # 'sac': SACAlgorithm,
}

# 算法元信息
ALGORITHM_INFO = {
    'reinforce': {
        'name': 'REINFORCE',
        'full_name': 'Policy Gradient with REINFORCE',
        'cn_name': 'REINFORCE策略梯度',
        'type': 'policy_gradient',
        'difficulty': 'easy',
        'status': 'active',
        'description': '最基础的策略梯度算法，易于实现和理解',
        'advantages': ['简单易懂', '易于实现', '适合入门'],
        'disadvantages': ['方差大', '训练不稳定', '收敛慢'],
        'suitable_for': ['小规模问题', '教学演示', '算法原型'],
        'params': {
            'baseline': {'default': 'rollout', 'options': ['rollout', 'exponential', 'critic']},
            'entropy_coef': {'default': 0.0, 'range': [0.0, 0.1]},
        }
    },
    'ppo': {
        'name': 'PPO',
        'full_name': 'Proximal Policy Optimization',
        'cn_name': '近端策略优化',
        'type': 'policy_gradient',
        'difficulty': 'medium',
        'status': 'active',
        'description': '工业界首选的RL算法，稳定性强',
        'advantages': ['训练稳定', '超参数不敏感', '收敛快', '工业界验证'],
        'disadvantages': ['实现稍复杂', '计算开销较大'],
        'suitable_for': ['生产环境', '大规模问题', '实际应用'],
        'params': {
            'clip_ratio': {'default': 0.2, 'range': [0.1, 0.3]},
            'value_loss_coef': {'default': 0.5, 'range': [0.1, 1.0]},
            'entropy_coef': {'default': 0.01, 'range': [0.0, 0.1]},
            'epochs_per_update': {'default': 4, 'range': [1, 10]},
        }
    },
    'a2c': {
        'name': 'A2C',
        'full_name': 'Advantage Actor-Critic',
        'cn_name': '优势Actor-Critic',
        'type': 'actor_critic',
        'difficulty': 'medium',
        'status': 'active',
        'description': 'Actor-Critic方法，比REINFORCE更稳定',
        'advantages': ['方差小', '收敛快', '性能稳定'],
        'disadvantages': ['需要Critic网络', '实现复杂'],
        'suitable_for': ['中规模问题', '需要稳定训练'],
        'params': {
            'value_loss_coef': {'default': 0.5, 'range': [0.1, 1.0]},
            'entropy_coef': {'default': 0.01, 'range': [0.0, 0.1]},
        }
    },
}


def get_algorithm_class(algorithm_name: str):
    """
    根据算法名称获取对应的算法类
    
    参数:
        algorithm_name: 算法名称 (如 'reinforce', 'ppo', 'a2c')
    
    返回:
        算法类 (BaseAlgorithm的子类)
    
    异常:
        ValueError: 如果算法不支持
    """
    algorithm_name = algorithm_name.lower()
    
    if algorithm_name not in ALGORITHM_REGISTRY:
        available = ', '.join(ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"不支持的算法: {algorithm_name}\n"
            f"可用的算法: {available}"
        )
    
    return ALGORITHM_REGISTRY[algorithm_name]


def get_algorithm_info(algorithm_name: str = None):
    """
    获取算法元信息
    
    参数:
        algorithm_name: 算法名称，None则返回所有
    
    返回:
        dict: 算法信息字典
    """
    if algorithm_name is None:
        return ALGORITHM_INFO
    
    algorithm_name = algorithm_name.lower()
    return ALGORITHM_INFO.get(algorithm_name, {})


def list_available_algorithms():
    """
    列出所有可用的算法
    
    返回:
        list: [(算法名称, 状态, 中文名)]
    """
    algorithms = []
    for key, info in ALGORITHM_INFO.items():
        if key in ALGORITHM_REGISTRY:
            algorithms.append((key, info['status'], info['cn_name']))
    return algorithms


__all__ = [
    'BaseAlgorithm',
    'REINFORCEAlgorithm',
    'PPOAlgorithm',
    'A2CAlgorithm',
    'ALGORITHM_REGISTRY',
    'ALGORITHM_INFO',
    'get_algorithm_class',
    'get_algorithm_info',
    'list_available_algorithms',
]






