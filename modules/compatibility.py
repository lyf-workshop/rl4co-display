"""
组件兼容性验证模块
检查问题、策略、算法的组合是否有效
"""

from typing import Tuple, Dict, List

# ============================================
# 兼容性规则定义
# ============================================

# 策略 → 问题兼容性
POLICY_PROBLEM_COMPATIBILITY = {
    'attention': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'pctsp', 'op'],  # AM通用
    'am': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'pctsp', 'op'],         # AM别名
    'pomo': ['tsp', 'mtsp', 'cvrp'],                                                 # POMO只适用对称问题（不支持ATSP）
}

# 算法 → 问题兼容性 (大部分算法通用)
ALGORITHM_PROBLEM_COMPATIBILITY = {
    'reinforce': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'pctsp', 'op'],
    'ppo': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'pctsp', 'op'],
    'a2c': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'pctsp', 'op'],
}

# 策略 → 算法兼容性
POLICY_ALGORITHM_COMPATIBILITY = {
    'attention': ['reinforce', 'ppo', 'a2c'],
    'am': ['reinforce', 'ppo', 'a2c'],
    'pomo': ['reinforce', 'ppo', 'a2c'],
}

# 警告组合 (技术上可行，但不推荐)
WARNING_COMBINATIONS = [
    {
        'problem': 'atsp',
        'policy': 'pomo',
        'message': 'POMO设计用于对称问题，不支持ATSP（非对称距离）。请使用Attention Model',
        'severity': 'error'
    },
    {
        'problem': 'atsp',
        'algorithm': 'reinforce',
        'message': 'ATSP问题复杂度高，建议使用PPO或A2C算法以获得更好的收敛性',
        'severity': 'info'
    },
    {
        'problem': 'sdvrp',
        'policy': 'pomo',
        'message': 'POMO在SDVRP上的效果未经充分验证，建议使用Attention Model',
        'severity': 'warning'
    },
    {
        'problem': 'vrptw',
        'policy': 'pomo',
        'message': 'POMO在VRPTW上的效果未经验证，建议使用Attention Model',
        'severity': 'warning'
    },
    {
        'problem': 'vrptw',
        'algorithm': 'reinforce',
        'message': 'VRPTW问题复杂度高，建议使用PPO或A2C算法以获得更稳定的训练',
        'severity': 'info'
    },
]

# 推荐组合 (根据问题类型)
RECOMMENDED_COMBINATIONS = {
    'tsp': {
        'best': {'policy': 'pomo', 'algorithm': 'ppo'},
        'fast': {'policy': 'attention', 'algorithm': 'ppo'},
        'simple': {'policy': 'attention', 'algorithm': 'reinforce'},
    },
    'atsp': {
        'best': {'policy': 'attention', 'algorithm': 'ppo'},      # ATSP只能用AM（POMO不支持非对称）
        'fast': {'policy': 'attention', 'algorithm': 'a2c'},
        'simple': {'policy': 'attention', 'algorithm': 'ppo'},     # ATSP不建议用REINFORCE
    },
    'mtsp': {
        'best': {'policy': 'pomo', 'algorithm': 'ppo'},           # mTSP支持POMO（对称问题）
        'fast': {'policy': 'attention', 'algorithm': 'ppo'},
        'simple': {'policy': 'attention', 'algorithm': 'reinforce'},
    },
    'cvrp': {
        'best': {'policy': 'pomo', 'algorithm': 'ppo'},
        'fast': {'policy': 'attention', 'algorithm': 'ppo'},
        'simple': {'policy': 'attention', 'algorithm': 'reinforce'},
    },
    'sdvrp': {
        'best': {'policy': 'attention', 'algorithm': 'ppo'},
        'fast': {'policy': 'attention', 'algorithm': 'a2c'},
        'simple': {'policy': 'attention', 'algorithm': 'reinforce'},
    },
    'vrptw': {
        'best': {'policy': 'attention', 'algorithm': 'ppo'},
        'fast': {'policy': 'attention', 'algorithm': 'a2c'},
        'simple': {'policy': 'attention', 'algorithm': 'ppo'},  # VRPTW不建议用REINFORCE
    },
}


# ============================================
# 验证函数
# ============================================

def validate_combination(
    problem: str,
    policy: str,
    algorithm: str
) -> Tuple[bool, str, str]:
    """
    验证问题、策略、算法的组合是否有效
    
    参数:
        problem: 问题类型 (tsp, cvrp, sdvrp等)
        policy: 策略模型 (attention, pomo等)
        algorithm: RL算法 (reinforce, ppo, a2c等)
    
    返回:
        (is_valid, message, level)
        - is_valid: bool - 组合是否有效
        - message: str - 提示信息
        - level: str - 'error', 'warning', 'success'
    """
    problem = problem.lower()
    policy = policy.lower()
    algorithm = algorithm.lower()
    
    # 1. 检查策略是否支持该问题
    if not is_policy_compatible_with_problem(policy, problem):
        return (
            False,
            f'❌ {policy.upper()}策略不适用于{problem.upper()}问题。'
            f'建议使用Attention Model。',
            'error'
        )
    
    # 2. 检查算法是否支持该问题
    if not is_algorithm_compatible_with_problem(algorithm, problem):
        return (
            False,
            f'❌ {algorithm.upper()}算法不适用于{problem.upper()}问题。',
            'error'
        )
    
    # 3. 检查策略和算法是否兼容
    if not is_policy_compatible_with_algorithm(policy, algorithm):
        return (
            False,
            f'❌ {policy.upper()}策略与{algorithm.upper()}算法不兼容。',
            'error'
        )
    
    # 4. 检查是否有警告
    warning = get_combination_warning(problem, policy, algorithm)
    if warning:
        return True, f'⚠️ {warning}', 'warning'
    
    # 5. 所有检查通过
    return True, f'✅ 配置有效: {problem.upper()} + {policy.upper()} + {algorithm.upper()}', 'success'


def is_policy_compatible_with_problem(policy: str, problem: str) -> bool:
    """检查策略是否支持该问题"""
    policy = policy.lower()
    problem = problem.lower()
    
    if policy not in POLICY_PROBLEM_COMPATIBILITY:
        return False
    
    return problem in POLICY_PROBLEM_COMPATIBILITY[policy]


def is_algorithm_compatible_with_problem(algorithm: str, problem: str) -> bool:
    """检查算法是否支持该问题"""
    algorithm = algorithm.lower()
    problem = problem.lower()
    
    if algorithm not in ALGORITHM_PROBLEM_COMPATIBILITY:
        return False
    
    return problem in ALGORITHM_PROBLEM_COMPATIBILITY[algorithm]


def is_policy_compatible_with_algorithm(policy: str, algorithm: str) -> bool:
    """检查策略和算法是否兼容"""
    policy = policy.lower()
    algorithm = algorithm.lower()
    
    if policy not in POLICY_ALGORITHM_COMPATIBILITY:
        return False
    
    return algorithm in POLICY_ALGORITHM_COMPATIBILITY[policy]


def get_combination_warning(problem: str, policy: str, algorithm: str) -> str:
    """获取组合的警告信息（如果有）"""
    problem = problem.lower()
    policy = policy.lower()
    algorithm = algorithm.lower()
    
    for warning in WARNING_COMBINATIONS:
        # 检查问题是否匹配
        if warning.get('problem') != problem:
            continue
        
        # 检查策略是否匹配（如果警告中指定了策略）
        if 'policy' in warning and warning['policy'] != policy:
            continue
        
        # 检查算法是否匹配（如果警告中指定了算法）
        if 'algorithm' in warning and warning['algorithm'] != algorithm:
            continue
        
        # 如果所有条件都匹配，返回警告消息
        return warning['message']
    
    return ""


def get_available_policies(problem: str) -> List[str]:
    """获取问题类型支持的所有策略"""
    problem = problem.lower()
    available = []
    
    for policy, problems in POLICY_PROBLEM_COMPATIBILITY.items():
        if problem in problems:
            available.append(policy)
    
    return available


def get_available_algorithms(problem: str, policy: str = None) -> List[str]:
    """获取问题类型（和策略）支持的所有算法"""
    problem = problem.lower()
    
    # 先获取问题支持的算法
    problem_algorithms = []
    for algorithm, problems in ALGORITHM_PROBLEM_COMPATIBILITY.items():
        if problem in problems:
            problem_algorithms.append(algorithm)
    
    # 如果指定了策略，进一步筛选
    if policy:
        policy = policy.lower()
        if policy in POLICY_ALGORITHM_COMPATIBILITY:
            policy_algorithms = POLICY_ALGORITHM_COMPATIBILITY[policy]
            # 取交集
            return [alg for alg in problem_algorithms if alg in policy_algorithms]
    
    return problem_algorithms


def get_recommended_combination(problem: str, preference: str = 'best') -> Dict[str, str]:
    """
    获取推荐的组合
    
    参数:
        problem: 问题类型
        preference: 偏好 ('best', 'fast', 'simple')
    
    返回:
        {'policy': '...', 'algorithm': '...'}
    """
    problem = problem.lower()
    
    if problem not in RECOMMENDED_COMBINATIONS:
        # 默认推荐
        return {'policy': 'attention', 'algorithm': 'reinforce'}
    
    if preference not in RECOMMENDED_COMBINATIONS[problem]:
        preference = 'best'
    
    return RECOMMENDED_COMBINATIONS[problem][preference]


def get_compatibility_info() -> Dict:
    """
    获取完整的兼容性信息（用于前端）
    
    返回:
        {
            'policy_problem': {...},
            'algorithm_problem': {...},
            'policy_algorithm': {...},
            'warnings': [...],
            'recommendations': {...}
        }
    """
    return {
        'policy_problem': POLICY_PROBLEM_COMPATIBILITY,
        'algorithm_problem': ALGORITHM_PROBLEM_COMPATIBILITY,
        'policy_algorithm': POLICY_ALGORITHM_COMPATIBILITY,
        'warnings': WARNING_COMBINATIONS,
        'recommendations': RECOMMENDED_COMBINATIONS,
    }


# ============================================
# 便捷验证函数
# ============================================

def validate_config(config: Dict) -> Tuple[bool, str, str]:
    """
    验证完整的训练配置
    
    参数:
        config: 训练配置字典
    
    返回:
        (is_valid, message, level)
    """
    problem = config.get('problem', 'tsp')
    policy = config.get('model', 'attention')
    algorithm = config.get('algorithm', 'reinforce')
    
    return validate_combination(problem, policy, algorithm)


def suggest_alternative(problem: str, policy: str, algorithm: str) -> Dict[str, str]:
    """
    如果当前组合无效，建议替代方案
    
    返回:
        {'policy': '...', 'algorithm': '...', 'reason': '...'}
    """
    is_valid, message, level = validate_combination(problem, policy, algorithm)
    
    if is_valid:
        return {'policy': policy, 'algorithm': algorithm, 'reason': '当前组合有效'}
    
    # 获取推荐组合
    recommended = get_recommended_combination(problem, 'best')
    
    return {
        'policy': recommended['policy'],
        'algorithm': recommended['algorithm'],
        'reason': f'原组合无效：{message}。建议使用推荐组合。'
    }


# ============================================
# 前端API接口辅助函数
# ============================================

def get_frontend_constraints(problem: str) -> Dict:
    """
    为前端提供约束信息
    
    参数:
        problem: 问题类型
    
    返回:
        {
            'available_policies': [...],
            'available_algorithms': [...],
            'recommended': {...},
            'warnings': [...]
        }
    """
    problem = problem.lower()
    
    return {
        'available_policies': get_available_policies(problem),
        'available_algorithms': get_available_algorithms(problem),
        'recommended': get_recommended_combination(problem, 'best'),
        'warnings': [
            w for w in WARNING_COMBINATIONS 
            if w['problem'] == problem
        ]
    }


if __name__ == '__main__':
    # 测试示例
    print("=== 兼容性验证测试 ===\n")
    
    # 测试1: 有效组合
    print("测试1: TSP + POMO + PPO")
    valid, msg, level = validate_combination('tsp', 'pomo', 'ppo')
    print(f"结果: {msg} [{level}]\n")
    
    # 测试2: 无效组合
    print("测试2: PCTSP + POMO + REINFORCE")
    valid, msg, level = validate_combination('pctsp', 'pomo', 'reinforce')
    print(f"结果: {msg} [{level}]\n")
    
    # 测试3: 警告组合
    print("测试3: SDVRP + POMO + PPO")
    valid, msg, level = validate_combination('sdvrp', 'pomo', 'ppo')
    print(f"结果: {msg} [{level}]\n")
    
    # 测试4: 获取可用选项
    print("测试4: TSP可用的策略和算法")
    print(f"策略: {get_available_policies('tsp')}")
    print(f"算法: {get_available_algorithms('tsp')}\n")
    
    # 测试5: 获取推荐组合
    print("测试5: CVRP的推荐组合")
    print(f"最佳: {get_recommended_combination('cvrp', 'best')}")
    print(f"快速: {get_recommended_combination('cvrp', 'fast')}")
    print(f"简单: {get_recommended_combination('cvrp', 'simple')}\n")



