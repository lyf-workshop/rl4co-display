"""
CVRP (Capacitated Vehicle Routing Problem) - 带容量约束的车辆路径问题
"""

from typing import Dict, Any, Callable
from .base_problem import BaseProblem


class CVRProblem(BaseProblem):
    """
    车辆路径问题类
    
    目标：在满足容量约束的前提下，规划最短配送路径
    特点：
        - 有容量约束
        - 可多次返回仓库
        - 实用性强（物流配送）
    """
    
    def _init_problem_params(self):
        """初始化CVRP特定参数"""
        self.vehicle_capacity = float(self.config.get('vehicle_capacity', 1.0))
        self.num_vehicles = int(self.config.get('num_vehicles', 1))
    
    def get_problem_type(self) -> str:
        return 'cvrp'
    
    def get_problem_name(self) -> str:
        return 'Capacitated Vehicle Routing Problem'
    
    def create_environment(self):
        """
        创建CVRP环境
        
        返回:
            CVRPEnv: RL4CO的CVRP环境实例
        """
        try:
            from rl4co.envs import CVRPEnv
        except ImportError:
            raise ImportError(
                "RL4CO库未安装，无法创建CVRP环境。\n"
                "请安装: pip install rl4co"
            )
        
        env = CVRPEnv(generator_params=self.get_env_params())
        return env
    
    def get_visualization_functions(self) -> Dict[str, Callable]:
        """
        获取CVRP可视化函数
        
        返回:
            dict: {
                'animation': 创建配送路线动画,
                'comparison': 创建对比图
            }
        """
        from modules.rl_training.visualizations.cvrp_viz import (
            create_cvrp_route_animation,
            create_cvrp_comparison_plot
        )
        
        return {
            'animation': create_cvrp_route_animation,
            'comparison': create_cvrp_comparison_plot,
        }
    
    def get_problem_description(self) -> str:
        return (
            f"车辆路径问题 (CVRP)\n"
            f"目标: 规划车辆配送路径，访问{self.num_loc}个客户\n"
            f"约束: 车辆容量={self.vehicle_capacity}，可多次返回仓库补货"
        )
    
    def get_problem_features(self) -> list:
        return [
            '容量约束',
            '多次返回仓库',
            '实用性强（物流配送）',
            '比TSP更复杂',
        ]
    
    def get_env_params(self) -> Dict[str, Any]:
        """CVRP环境参数"""
        return {
            'num_loc': self.num_loc,
            'vehicle_capacity': self.vehicle_capacity,
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取CVRP默认参数"""
        params = super().get_default_params()
        params.update({
            'vehicle_capacity': self.vehicle_capacity,
            'num_vehicles': self.num_vehicles,
        })
        return params
    
    def _validate_problem_params(self):
        """CVRP参数验证"""
        if self.vehicle_capacity <= 0:
            return False, "vehicle_capacity必须大于0"
        
        if self.num_vehicles < 1:
            return False, "num_vehicles必须大于等于1"
        
        if self.num_loc > 200:
            return False, "CVRP客户数量不建议超过200"
        
        return True, ""


