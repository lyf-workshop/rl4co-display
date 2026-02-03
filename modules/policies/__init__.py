"""
策略模型模块
提供统一的策略网络接口，支持多种模型（AM、POMO、SymNCO等）

使用方式:
    from modules.policies import get_policy_class, POLICY_REGISTRY
    
    # 获取策略类
    PolicyClass = get_policy_class('attention')
    policy = PolicyClass(env, config)
"""

from .base_policy import BasePolicy
from .attention_model_policy import AttentionModelPolicyWrapper
from .pomo_policy import POMOPolicyWrapper

# 策略注册表
POLICY_REGISTRY = {
    'attention': AttentionModelPolicyWrapper,
    'am': AttentionModelPolicyWrapper,  # 别名
    'pomo': POMOPolicyWrapper,
    # 未来扩展：
    # 'symnco': SymNCOPolicyWrapper,
    # 'mdam': MDAMPolicyWrapper,
    # 'matnet': MatNetPolicyWrapper,
}

# 策略元信息
POLICY_INFO = {
    'attention': {
        'name': 'AM',
        'full_name': 'Attention Model',
        'cn_name': '注意力模型',
        'type': 'constructive',
        'year': 2019,
        'status': 'active',
        'description': '基于Transformer的构造式模型，适合入门',
        'advantages': ['速度快', '易于理解', '经典方法'],
        'disadvantages': ['质量中等', '未利用对称性'],
        'suitable_for': ['TSP', 'CVRP', 'PCTSP', 'OP'],
        'params': {
            'embed_dim': {'default': 128, 'range': [64, 256]},
            'num_encoder_layers': {'default': 3, 'range': [1, 6]},
            'num_heads': {'default': 8, 'range': [4, 16]},
        }
    },
    'pomo': {
        'name': 'POMO',
        'full_name': 'Policy Optimization with Multiple Optima',
        'cn_name': '多起点优化',
        'type': 'constructive',
        'year': 2020,
        'status': 'active',
        'description': '同时从多个起点构建路径，质量更高',
        'advantages': ['质量高', '并行效率高', '利用对称性'],
        'disadvantages': ['计算量大', '显存占用高'],
        'suitable_for': ['TSP', 'CVRP'],
        'params': {
            'embed_dim': {'default': 128, 'range': [64, 256]},
            'num_encoder_layers': {'default': 6, 'range': [3, 9]},
            'num_heads': {'default': 8, 'range': [4, 16]},
            'num_starts': {'default': 50, 'range': [10, 100]},  # POMO特有
        }
    },
}


def get_policy_class(policy_name: str):
    """
    根据策略名称获取对应的策略类
    
    参数:
        policy_name: 策略名称 (如 'attention', 'pomo')
    
    返回:
        策略类 (BasePolicy的子类)
    
    异常:
        ValueError: 如果策略不支持
    """
    policy_name = policy_name.lower()
    
    if policy_name not in POLICY_REGISTRY:
        available = ', '.join(POLICY_REGISTRY.keys())
        raise ValueError(
            f"不支持的策略模型: {policy_name}\n"
            f"可用的策略: {available}"
        )
    
    return POLICY_REGISTRY[policy_name]


def get_policy_info(policy_name: str = None):
    """
    获取策略元信息
    
    参数:
        policy_name: 策略名称，None则返回所有
    
    返回:
        dict: 策略信息字典
    """
    if policy_name is None:
        return POLICY_INFO
    
    policy_name = policy_name.lower()
    return POLICY_INFO.get(policy_name, {})


def list_available_policies():
    """
    列出所有可用的策略
    
    返回:
        list: [(策略名称, 状态, 中文名)]
    """
    policies = []
    for key, info in POLICY_INFO.items():
        if key in POLICY_REGISTRY:
            policies.append((key, info['status'], info['cn_name']))
    return policies


__all__ = [
    'BasePolicy',
    'AttentionModelPolicyWrapper',
    'POMOPolicyWrapper',
    'POLICY_REGISTRY',
    'POLICY_INFO',
    'get_policy_class',
    'get_policy_info',
    'list_available_policies',
]






