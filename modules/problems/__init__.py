"""
组合优化问题类型模块
提供统一的问题类型注册和调用接口

模块结构:
    - base_problem.py: 问题基类
    - tsp.py: 旅行商问题
    - cvrp.py: 车辆路径问题
    - pctsp.py: 奖励收集TSP (待实现)
    - op.py: 定向问题 (待实现)
    ...

使用方式:
    from modules.problems import get_problem_class, PROBLEM_REGISTRY
    
    # 获取问题类
    problem_class = get_problem_class('tsp')
    problem = problem_class(config)
    
    # 获取环境
    env = problem.create_environment()
    
    # 获取可视化函数
    viz_funcs = problem.get_visualization_functions()
"""

from .base_problem import BaseProblem
from .tsp import TSProblem
from .cvrp import CVRProblem
from .sdvrp import SDVRProblem
from .vrptw import VRPTWProblem
from .atsp import ATSProblem

# 问题类型注册表
PROBLEM_REGISTRY = {
    'tsp': TSProblem,
    'atsp': ATSProblem,
    'cvrp': CVRProblem,
    'sdvrp': SDVRProblem,
    'vrptw': VRPTWProblem,
    # 未来扩展：
    # 'pctsp': PCTSProblem,
    # 'op': OProblem,
    # 'jssp': JSSProblem,
    # 'bpp': BPProblem,
}

# 问题类型元信息
PROBLEM_INFO = {
    'tsp': {
        'name': 'TSP',
        'full_name': 'Traveling Salesman Problem',
        'cn_name': '旅行商问题',
        'category': 'routing',
        'difficulty': 'medium',
        'status': 'active',
        'description': '寻找访问所有城市的最短路径',
        'params': ['num_loc'],
        'features': ['无容量约束', '单条路径', '经典NP-hard'],
    },
    'atsp': {
        'name': 'ATSP',
        'full_name': 'Asymmetric Traveling Salesman Problem',
        'cn_name': '非对称旅行商问题',
        'category': 'routing',
        'difficulty': 'hard',
        'status': 'active',
        'description': '在非对称距离矩阵下寻找最短路径',
        'params': ['num_loc'],
        'features': ['无容量约束', '单条路径', '距离不对称', '有向图问题', '比TSP更复杂'],
    },
    'cvrp': {
        'name': 'CVRP',
        'full_name': 'Capacitated Vehicle Routing Problem',
        'cn_name': '带容量约束的车辆路径问题',
        'category': 'routing',
        'difficulty': 'hard',
        'status': 'active',
        'description': '在容量约束下规划车辆配送路径',
        'params': ['num_loc', 'vehicle_capacity'],
        'features': ['容量约束', '多次返回仓库', '实用性强'],
    },
    'sdvrp': {
        'name': 'SDVRP',
        'full_name': 'Split Delivery Vehicle Routing Problem',
        'cn_name': '分割配送车辆路径问题',
        'category': 'routing',
        'difficulty': 'hard',
        'status': 'active',
        'description': '允许分割配送的车辆路径规划',
        'params': ['num_loc', 'vehicle_capacity', 'max_split_per_customer'],
        'features': ['分割配送', '灵活方案', '减少返仓', '大宗物流'],
    },
    'vrptw': {
        'name': 'VRPTW',
        'full_name': 'Vehicle Routing Problem with Time Windows',
        'cn_name': '带时间窗的车辆路径问题',
        'category': 'routing',
        'difficulty': 'very_hard',
        'status': 'active',
        'description': '在容量和时间窗约束下规划配送路径',
        'params': ['num_loc', 'vehicle_capacity', 'time_window_width', 'service_time', 'max_time'],
        'features': ['容量约束', '时间窗约束', '实时配送', '快递外卖', '最复杂VRP'],
    },
    'pctsp': {
        'name': 'PCTSP',
        'full_name': 'Prize Collecting TSP',
        'cn_name': '奖励收集旅行商问题',
        'category': 'routing',
        'difficulty': 'medium',
        'status': 'planned',
        'description': '收集足够奖励的前提下最小化路径',
        'params': ['num_loc', 'prize_required'],
        'features': ['奖励约束', '部分访问', '灵活规划'],
    },
    'op': {
        'name': 'OP',
        'full_name': 'Orienteering Problem',
        'cn_name': '定向问题',
        'category': 'routing',
        'difficulty': 'medium',
        'status': 'planned',
        'description': '在时间预算内最大化收集的奖励',
        'params': ['num_loc', 'max_length'],
        'features': ['时间约束', '奖励最大化', '旅游规划'],
    },
}


def get_problem_class(problem_type: str):
    """
    根据问题类型获取对应的问题类
    
    参数:
        problem_type: 问题类型名称 (如 'tsp', 'cvrp')
    
    返回:
        问题类 (BaseProblem的子类)
    
    异常:
        ValueError: 如果问题类型不支持
    """
    problem_type = problem_type.lower()
    
    if problem_type not in PROBLEM_REGISTRY:
        available = ', '.join(PROBLEM_REGISTRY.keys())
        raise ValueError(
            f"不支持的问题类型: {problem_type}\n"
            f"可用的问题类型: {available}"
        )
    
    return PROBLEM_REGISTRY[problem_type]


def get_problem_info(problem_type: str = None):
    """
    获取问题类型的元信息
    
    参数:
        problem_type: 问题类型名称，None则返回所有
    
    返回:
        dict: 问题信息字典
    """
    if problem_type is None:
        return PROBLEM_INFO
    
    problem_type = problem_type.lower()
    return PROBLEM_INFO.get(problem_type, {})


def list_available_problems():
    """
    列出所有可用的问题类型
    
    返回:
        list: [(问题类型, 状态, 中文名)]
    """
    problems = []
    for key, info in PROBLEM_INFO.items():
        if key in PROBLEM_REGISTRY:
            problems.append((key, info['status'], info['cn_name']))
    return problems


def list_problems_by_category():
    """
    按类别列出问题类型
    
    返回:
        dict: {类别: [问题列表]}
    """
    categories = {}
    for key, info in PROBLEM_INFO.items():
        category = info.get('category', 'other')
        if category not in categories:
            categories[category] = []
        categories[category].append({
            'type': key,
            'name': info['name'],
            'cn_name': info['cn_name'],
            'status': info['status'],
        })
    return categories


# 导出所有公共接口
__all__ = [
    'BaseProblem',
    'TSProblem',
    'ATSProblem',
    'CVRProblem',
    'SDVRProblem',
    'VRPTWProblem',
    'PROBLEM_REGISTRY',
    'PROBLEM_INFO',
    'get_problem_class',
    'get_problem_info',
    'list_available_problems',
    'list_problems_by_category',
]

