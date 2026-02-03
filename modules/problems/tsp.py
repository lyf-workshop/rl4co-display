"""
TSP (Traveling Salesman Problem) - 旅行商问题
"""

from typing import Dict, Any, Callable
from .base_problem import BaseProblem


class TSProblem(BaseProblem):
    """
    旅行商问题类
    
    目标：找到访问所有城市并返回起点的最短路径
    特点：
        - 无容量约束
        - 单条闭合路径
        - 经典NP-hard问题
    """
    
    def get_problem_type(self) -> str:
        return 'tsp'
    
    def get_problem_name(self) -> str:
        return 'Traveling Salesman Problem'
    
    def create_environment(self):
        """
        创建TSP环境
        
        返回:
            TSPEnv: RL4CO的TSP环境实例
        """
        try:
            from rl4co.envs import TSPEnv
        except ImportError:
            raise ImportError(
                "RL4CO库未安装，无法创建TSP环境。\n"
                "请安装: pip install rl4co"
            )
        
        env = TSPEnv(generator_params=self.get_env_params())
        return env
    
    def get_visualization_functions(self) -> Dict[str, Callable]:
        """
        获取TSP可视化函数
        
        返回:
            dict: {
                'animation': 创建路线动画,
                'comparison': 创建对比图
            }
        """
        from modules.rl_training.visualizations.tsp_viz import (
            create_tsp_route_animation,
            create_tsp_comparison_plot
        )
        
        return {
            'animation': create_tsp_route_animation,
            'comparison': create_tsp_comparison_plot,
        }
    
    def get_problem_description(self) -> str:
        return (
            f"旅行商问题 (TSP)\n"
            f"目标: 找到访问所有{self.num_loc}个城市的最短路径\n"
            f"约束: 每个城市访问一次，最后返回起点"
        )
    
    def get_problem_features(self) -> list:
        return [
            '无容量约束',
            '单条闭合路径',
            '经典NP-hard问题',
            '广泛应用于物流和调度',
        ]
    
    def get_env_params(self) -> Dict[str, Any]:
        """TSP环境参数"""
        return {
            'num_loc': self.num_loc,
        }
    
    def _validate_problem_params(self):
        """TSP参数验证"""
        if self.num_loc > 1000:
            return False, "TSP城市数量不建议超过1000"
        return True, ""


