"""
组件兼容性验证模块
检查问题、策略、算法的组合是否有效

基于 RL4CO 官方文档（v0.6.0+）更新
数据来源：https://github.com/ai4co/rl4co
最后更新：2024年2月
"""

from typing import Tuple, Dict, List

# ============================================
# 环境分类（基于 RL4CO 官方）
# ============================================

# 路由问题（Routing Problems）- 本系统已集成
ROUTING_PROBLEMS_INTEGRATED = [
    'tsp',      # Traveling Salesman Problem
    'atsp',     # Asymmetric TSP
    'mtsp',     # Multiple TSP
    'cvrp',     # Capacitated VRP
    'sdvrp',    # Split Delivery VRP
    'vrptw',    # VRP with Time Windows
    'op',       # Orienteering Problem
    'pdp',      # Pickup and Delivery Problem
]

# 调度问题（Scheduling Problems）- 本系统已集成
SCHEDULING_PROBLEMS_INTEGRATED = [
    'ffsp',     # Flexible Flow Shop Problem
]

# 所有已集成的问题
ALL_INTEGRATED_PROBLEMS = ROUTING_PROBLEMS_INTEGRATED + SCHEDULING_PROBLEMS_INTEGRATED

# RL4CO 官方支持但本系统未集成的环境（供将来扩展）
ROUTING_PROBLEMS_AVAILABLE = [
    'pctsp',    # Prize Collecting TSP
    'spctsp',   # Stochastic PCTSP
    'cvrptw',   # Capacitated VRP with Time Windows
    'svrp',     # Skill VRP
    'dpp',      # Dial-a-Ride Problem
    'mdpp',     # Multi-Depot PDP
    'mdcpdp',   # Multi-Depot Capacitated PDP
    'mtvrp',    # Multi-Task VRP (16 variants)
]

SCHEDULING_PROBLEMS_AVAILABLE = [
    'ffsp',     # Flexible Flow Shop Problem
    'fjsp',     # Flexible Job Shop Problem
    'jssp',     # Job Shop Scheduling Problem
    'smtwtp',   # Single Machine Total Weighted Tardiness
]

# ============================================
# 兼容性规则定义（基于官方文档）
# ============================================

# 策略 → 问题兼容性
POLICY_PROBLEM_COMPATIBILITY = {
    # Attention Model：支持所有路由问题（需要对应的 init_embedding）
    'attention': ROUTING_PROBLEMS_INTEGRATED,
    'am': ROUTING_PROBLEMS_INTEGRATED,  # AM 别名
    
    # POMO：仅适用对称路由问题（利用旋转对称性）
    'pomo': ['tsp', 'mtsp', 'cvrp'],
    
    # Pointer Network：基础路由问题（历史方法，性能有限）
    'ptrnet': ['tsp', 'cvrp'],
    'ptr': ['tsp', 'cvrp'],  # PtrNet 别名
    
    # MatNet：专为非对称和调度问题设计（矩阵注意力）
    'matnet': ['atsp', 'ffsp'],  # ATSP和FFSP
}

# 算法 → 问题兼容性（官方文档：REINFORCE, PPO, A2C 都是通用算法）
ALGORITHM_PROBLEM_COMPATIBILITY = {
    'reinforce': ALL_INTEGRATED_PROBLEMS,  # REINFORCE 通用
    'ppo': ALL_INTEGRATED_PROBLEMS,        # PPO 通用，复杂问题推荐
    'a2c': ALL_INTEGRATED_PROBLEMS,        # A2C 通用，快速收敛
}

# 策略 → 算法兼容性
POLICY_ALGORITHM_COMPATIBILITY = {
    'attention': ['reinforce', 'ppo', 'a2c'],
    'am': ['reinforce', 'ppo', 'a2c'],
    'pomo': ['reinforce', 'ppo', 'a2c'],
    'ptrnet': ['reinforce'],  # PtrNet 通常只使用 REINFORCE（经典组合）
    'ptr': ['reinforce'],
    'matnet': ['reinforce', 'ppo', 'a2c'],  # MatNet支持所有通用算法
}

# 警告组合 (技术上可行，但不推荐)
WARNING_COMBINATIONS = [
    {
        'problem': 'atsp',
        'policy': 'pomo',
        'message': 'POMO设计用于对称问题，不支持ATSP（非对称距离）。请使用MatNet',
        'severity': 'error'
    },
    {
        'problem': 'ffsp',
        'policy': 'attention',
        'message': 'Attention Model不支持FFSP调度问题。请使用MatNet',
        'severity': 'error'
    },
    {
        'problem': 'ffsp',
        'policy': 'pomo',
        'message': 'POMO不支持FFSP调度问题。请使用MatNet',
        'severity': 'error'
    },
    {
        'problem': 'ffsp',
        'policy': 'ptrnet',
        'message': 'PtrNet不支持FFSP调度问题。请使用MatNet',
        'severity': 'error'
    },
    {
        'problem': 'ffsp',
        'algorithm': 'reinforce',
        'message': 'FFSP是复杂调度问题，建议使用PPO或A2C算法以获得更稳定的训练和更好的收敛性',
        'severity': 'info'
    },
    {
        'policy': 'ptrnet',
        'message': 'PtrNet 是2015年的经典方法，性能不如现代方法（AM/POMO）。推荐用于学习和研究，实际应用建议使用 Attention Model 或 POMO',
        'severity': 'info'
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
    'pdp': {
        'best': {'policy': 'attention', 'algorithm': 'ppo', 'description': '最佳质量配置'},
        'fast': {'policy': 'attention', 'algorithm': 'a2c', 'description': '快速训练配置'},
        'simple': {'policy': 'attention', 'algorithm': 'reinforce', 'description': '简单易用配置'},
    },
    'op': {
        'best': {'policy': 'attention', 'algorithm': 'ppo', 'description': '最佳质量配置'},
        'fast': {'policy': 'attention', 'algorithm': 'a2c', 'description': '快速训练配置'},
        'simple': {'policy': 'attention', 'algorithm': 'reinforce', 'description': '简单易用配置'},
    },
    'ffsp': {
        'best': {'policy': 'matnet', 'algorithm': 'ppo', 'description': '最佳质量配置（复杂调度推荐）'},
        'fast': {'policy': 'matnet', 'algorithm': 'a2c', 'description': '快速收敛配置'},
        'simple': {'policy': 'matnet', 'algorithm': 'reinforce', 'description': '简单易用配置'},
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


def print_compatibility_matrix():
    """打印完整的兼容性矩阵（用于文档生成）"""
    print("\n" + "=" * 80)
    print("RL4CO Display - 兼容性矩阵")
    print("基于 RL4CO 官方文档 v0.6.0+")
    print("=" * 80)
    
    # 打印已集成的问题
    print(f"\n📦 已集成的问题类型 ({len(ALL_INTEGRATED_PROBLEMS)}个):")
    print(f"  路由问题: {', '.join(ROUTING_PROBLEMS_INTEGRATED)}")
    print(f"  调度问题: {', '.join(SCHEDULING_PROBLEMS_INTEGRATED)}")
    
    # 打印策略支持
    print(f"\n🧠 策略模型支持:")
    for policy, problems in POLICY_PROBLEM_COMPATIBILITY.items():
        print(f"  {policy.upper()}: 支持 {len(problems)} 个问题")
        print(f"    → {', '.join(problems[:8])}...")
    
    # 打印算法支持
    print(f"\n🎮 算法支持:")
    for algorithm, problems in ALGORITHM_PROBLEM_COMPATIBILITY.items():
        print(f"  {algorithm.upper()}: 支持 {len(problems)} 个问题 (通用)")
    
    # 打印推荐配置
    print(f"\n⭐ 推荐配置 ({len(RECOMMENDED_COMBINATIONS)}个问题):")
    for problem, configs in RECOMMENDED_COMBINATIONS.items():
        best = configs['best']
        print(f"  {problem.upper()}: {best['policy'].upper()} + {best['algorithm'].upper()}")
    
    print("\n" + "=" * 80)


def validate_system_consistency():
    """验证系统配置的一致性"""
    print("\n" + "=" * 80)
    print("系统一致性检查")
    print("=" * 80)
    
    errors = []
    warnings = []
    
    # 检查1: 推荐配置是否都在兼容性列表中
    for problem, configs in RECOMMENDED_COMBINATIONS.items():
        for config_type, config in configs.items():
            policy = config['policy']
            algorithm = config['algorithm']
            
            if not is_policy_compatible_with_problem(policy, problem):
                errors.append(f"{problem.upper()}: 推荐的策略 {policy} 不在兼容列表中")
            
            if not is_algorithm_compatible_with_problem(algorithm, problem):
                errors.append(f"{problem.upper()}: 推荐的算法 {algorithm} 不在兼容列表中")
    
    # 检查2: 警告组合是否有效
    for warning in WARNING_COMBINATIONS:
        problem = warning.get('problem')
        policy = warning.get('policy')
        algorithm = warning.get('algorithm')
        
        if problem and problem not in ALL_INTEGRATED_PROBLEMS:
            warnings.append(f"警告组合引用了未集成的问题: {problem}")
    
    # 输出结果
    if not errors and not warnings:
        print("✅ 所有检查通过，配置一致！")
    else:
        if errors:
            print(f"\n❌ 发现 {len(errors)} 个错误:")
            for err in errors:
                print(f"  - {err}")
        if warnings:
            print(f"\n⚠️  发现 {len(warnings)} 个警告:")
            for warn in warnings:
                print(f"  - {warn}")
    
    print("\n" + "=" * 80)
    
    return len(errors) == 0


if __name__ == '__main__':
    # 打印兼容性矩阵
    print_compatibility_matrix()
    
    # 验证一致性
    validate_system_consistency()
    
    print("\n" + "=" * 80)
    print("兼容性验证测试")
    print("=" * 80)
    
    # 测试1: 有效组合
    print("\n测试1: TSP + POMO + PPO")
    valid, msg, level = validate_combination('tsp', 'pomo', 'ppo')
    print(f"结果: {msg} [{level}]")
    
    # 测试2: 无效组合
    print("\n测试2: ATSP + POMO + REINFORCE")
    valid, msg, level = validate_combination('atsp', 'pomo', 'reinforce')
    print(f"结果: {msg} [{level}]")
    
    # 测试3: PtrNet
    print("\n测试3: TSP + PtrNet + REINFORCE")
    valid, msg, level = validate_combination('tsp', 'ptrnet', 'reinforce')
    print(f"结果: {msg} [{level}]")
    
    # 测试4: 获取可用选项
    print("\n测试5: TSP可用的策略和算法")
    print(f"策略: {get_available_policies('tsp')}")
    print(f"算法: {get_available_algorithms('tsp')}")
    
    # 测试6: 获取推荐组合
    print("\n测试6: CVRP的推荐组合")
    print(f"最佳: {get_recommended_combination('cvrp', 'best')}")
    print(f"快速: {get_recommended_combination('cvrp', 'fast')}")
    print(f"简单: {get_recommended_combination('cvrp', 'simple')}")
    
    print("\n" + "=" * 80)



