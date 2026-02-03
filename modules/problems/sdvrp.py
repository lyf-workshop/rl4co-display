"""
SDVRP (Split Delivery Vehicle Routing Problem) - 分割配送车辆路径问题
"""

from typing import Dict, Any, Callable
from .base_problem import BaseProblem


class SDVRProblem(BaseProblem):
    """
    分割配送车辆路径问题类
    
    目标：在满足容量约束的前提下，规划最短配送路径，允许对同一客户进行多次配送
    """
    
    def _init_problem_params(self):
        """初始化SDVRP特定参数"""
        self.vehicle_capacity = float(self.config.get('vehicle_capacity', 1.0))
        self.num_vehicles = int(self.config.get('num_vehicles', 1))
        self.allow_split = True
        self.max_split_per_customer = int(self.config.get('max_split_per_customer', 3))
    
    def get_problem_type(self) -> str:
        return 'sdvrp'
    
    def get_problem_name(self) -> str:
        return 'Split Delivery Vehicle Routing Problem'
    
    def create_environment(self):
        """创建SDVRP环境"""
        try:
            try:
                from rl4co.envs import SDVRPEnv
                env = SDVRPEnv(generator_params=self.get_env_params())
            except (ImportError, AttributeError):
                from rl4co.envs import CVRPEnv
                print("Warning: Using CVRP environment for SDVRP")
                env = CVRPEnv(generator_params=self.get_env_params())
                env.allow_split_delivery = True
            return env
        except ImportError:
            raise ImportError("RL4CO not installed")
    
    def get_visualization_functions(self) -> Dict[str, Callable]:
        """获取SDVRP可视化函数"""
        from modules.rl_training.visualizations.sdvrp_viz import (
            create_sdvrp_route_animation,
            create_sdvrp_comparison_plot,
            create_sdvrp_split_analysis
        )
        
        return {
            'animation': create_sdvrp_route_animation,
            'comparison': create_sdvrp_comparison_plot,
            'split_analysis': create_sdvrp_split_analysis,
        }
    
    def get_problem_description(self) -> str:
        return (
            f"分割配送车辆路径问题 (SDVRP)\n"
            f"目标: 规划车辆配送路径，访问{self.num_loc}个客户\n"
            f"特点: 允许分割配送（同一客户可多次配送）\n"
            f"约束: 车辆容量={self.vehicle_capacity}，最多分割{self.max_split_per_customer}次"
        )
    
    def get_problem_features(self) -> list:
        return [
            '允许分割配送',
            '客户需求可由多车完成',
            '减少返回仓库次数',
            '适用于大宗物流',
            '比CVRP更灵活',
        ]
    
    def get_env_params(self) -> Dict[str, Any]:
        """SDVRP环境参数"""
        params = {
            'num_loc': self.num_loc,
            'vehicle_capacity': self.vehicle_capacity,
        }
        if hasattr(self, 'max_split_per_customer'):
            params['max_split_per_customer'] = self.max_split_per_customer
        return params
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取SDVRP默认参数"""
        params = super().get_default_params()
        params.update({
            'vehicle_capacity': self.vehicle_capacity,
            'num_vehicles': self.num_vehicles,
            'allow_split': self.allow_split,
            'max_split_per_customer': self.max_split_per_customer,
        })
        return params
    
    def _validate_problem_params(self):
        """SDVRP参数验证"""
        if self.vehicle_capacity <= 0:
            return False, "vehicle_capacity must be greater than 0"
        
        if self.num_vehicles < 1:
            return False, "num_vehicles must be at least 1"
        
        if self.max_split_per_customer < 1:
            return False, "max_split_per_customer must be at least 1"
        
        if self.max_split_per_customer > 10:
            return False, "max_split_per_customer should not exceed 10"
        
        if self.num_loc > 200:
            return False, "num_loc should not exceed 200 for SDVRP"
        
        return True, ""
    
    def get_split_delivery_info(self) -> str:
        """获取分割配送说明"""
        return (
            "分割配送特性:\n"
            "- 同一客户可以被访问多次\n"
            "- 每次配送部分需求\n"
            "- 减少车辆返回仓库次数\n"
            f"- 每个客户最多分割{self.max_split_per_customer}次\n"
            "\n"
            "适用场景:\n"
            "- 客户需求 > 车辆容量\n"
            "- 大宗物流配送\n"
            "- 需要灵活配送方案"
        )


