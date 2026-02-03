"""
ATSP (Asymmetric Traveling Salesman Problem) - 非对称旅行商问题
"""

from typing import Dict, Any, Callable
from .base_problem import BaseProblem


class ATSProblem(BaseProblem):
    """
    非对称旅行商问题类
    
    目标：找到访问所有城市并返回起点的最短路径
    特点：
        - 无容量约束
        - 单条闭合路径
        - 距离矩阵不对称（从A到B的距离 ≠ 从B到A的距离）
        - 比TSP更具挑战性
    
    应用场景：
        - 城市交通（考虑单行道）
        - 航空路线（考虑风向）
        - 网络路由（考虑链路方向性）
    """
    
    def get_problem_type(self) -> str:
        return 'atsp'
    
    def get_problem_name(self) -> str:
        return 'Asymmetric Traveling Salesman Problem'
    
    def create_environment(self):
        """
        创建ATSP环境
        
        返回:
            ATSPEnv: RL4CO的ATSP环境实例
        """
        try:
            from rl4co.envs import ATSPEnv
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或版本过低，无法创建ATSP环境。\n"
                "请安装或升级: pip install rl4co>=0.4.0"
            )
        
        env = ATSPEnv(generator_params=self.get_env_params())
        return env
    
    def get_visualization_functions(self) -> Dict[str, Callable]:
        """
        获取ATSP可视化函数
        
        ATSP的可视化与TSP类似，但需要显示方向性
        
        返回:
            dict: {
                'animation': 创建路线动画（带方向箭头）,
                'comparison': 创建对比图
            }
        """
        # ATSP可以复用TSP的可视化函数，因为都是单一路径
        # 但理想情况下应该显示方向箭头
        from modules.rl_training.visualizations.tsp_viz import (
            create_tsp_route_animation,
            create_tsp_comparison_plot
        )
        
        return {
            'animation': create_tsp_route_animation,  # 复用TSP动画
            'comparison': create_tsp_comparison_plot,  # 复用TSP对比图
        }
    
    def get_problem_description(self) -> str:
        return (
            f"非对称旅行商问题 (ATSP)\n"
            f"目标: 找到访问所有{self.num_loc}个城市的最短路径\n"
            f"约束: 每个城市访问一次，最后返回起点\n"
            f"特殊性: 距离矩阵不对称（考虑方向性）"
        )
    
    def get_problem_features(self) -> list:
        return [
            '无容量约束',
            '单条闭合路径',
            '距离不对称',
            '比TSP更复杂',
            '适用于有向图',
            '单行道、风向等实际场景',
        ]
    
    def get_env_params(self) -> Dict[str, Any]:
        """ATSP环境参数"""
        return {
            'num_loc': self.num_loc,
            # ATSP特有参数可以在这里添加
        }
    
    def _validate_problem_params(self):
        """ATSP参数验证"""
        # ATSP与TSP有相似的规模限制
        if self.num_loc > 1000:
            return False, "ATSP城市数量不建议超过1000（计算复杂度过高）"
        
        if self.num_loc < 5:
            return False, "ATSP城市数量建议至少5个以体现非对称性"
        
        return True, ""
