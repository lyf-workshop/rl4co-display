"""
mTSP (Multiple Traveling Salesman Problem) - 多旅行商问题
"""

from typing import Dict, Any, Callable
from .base_problem import BaseProblem


class MTSProblem(BaseProblem):
    """
    多旅行商问题类
    
    目标：使用多个旅行商（车辆/代理）访问所有城市并返回起点，优化目标可以是：
        - minmax: 最小化最长路径（默认）
        - sum: 最小化总路径长度
    
    特点：
        - 多个代理同时工作
        - 所有代理从同一起点（depot）出发和返回
        - 每个城市只能被一个代理访问一次
        - NP-hard问题，是TSP的推广
    """
    
    def _init_problem_params(self):
        """初始化mTSP特定参数"""
        self.num_agents = int(self.config.get('num_agents', 5))
        self.cost_type = self.config.get('cost_type', 'minmax')
    
    def get_problem_type(self) -> str:
        return 'mtsp'
    
    def get_problem_name(self) -> str:
        return 'Multiple Traveling Salesman Problem'
    
    def create_environment(self):
        """
        创建mTSP环境
        
        返回:
            MTSPEnv: RL4CO的mTSP环境实例
        """
        try:
            from rl4co.envs import MTSPEnv
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或版本不支持mTSP，无法创建mTSP环境。\n"
                "请安装/升级: pip install -U rl4co"
            )
        
        env = MTSPEnv(
            generator_params=self.get_env_params(),
            cost_type=self.cost_type
        )
        return env
    
    def get_visualization_functions(self) -> Dict[str, Callable]:
        """
        获取mTSP可视化函数
        
        返回:
            dict: {
                'animation': 创建路线动画,
                'comparison': 创建对比图
            }
        """
        # 注意：mTSP可视化需要处理多条路径
        from modules.rl_training.visualizations.mtsp_viz import (
            create_mtsp_route_animation,
            create_mtsp_comparison_plot
        )
        
        return {
            'animation': create_mtsp_route_animation,
            'comparison': create_mtsp_comparison_plot,
        }
    
    def get_problem_description(self) -> str:
        cost_desc = "最长路径" if self.cost_type == 'minmax' else "总路径长度"
        return (
            f"多旅行商问题 (mTSP)\n"
            f"目标: 使用{self.num_agents}个代理访问所有{self.num_loc}个城市\n"
            f"约束: 每个城市访问一次，所有代理从depot出发并返回\n"
            f"优化目标: 最小化{cost_desc}"
        )
    
    def get_problem_features(self) -> list:
        return [
            f'{self.num_agents}个代理协同工作',
            '所有代理共享同一起点',
            '每个城市仅访问一次',
            '可选优化目标: minmax或sum',
            'TSP的多代理推广',
            '适用于多车辆调度场景',
        ]
    
    def get_env_params(self) -> Dict[str, Any]:
        """mTSP环境参数"""
        return {
            'num_loc': self.num_loc,
            'min_num_agents': self.num_agents,
            'max_num_agents': self.num_agents,  # 固定代理数量
        }
    
    def _validate_problem_params(self):
        """mTSP参数验证"""
        if self.num_loc > 1000:
            return False, "mTSP城市数量不建议超过1000"
        
        if self.num_agents < 1:
            return False, "代理数量至少为1"
        
        if self.num_agents > self.num_loc:
            return False, f"代理数量({self.num_agents})不能超过城市数量({self.num_loc})"
        
        if self.cost_type not in ['minmax', 'sum']:
            return False, "cost_type必须是'minmax'或'sum'"
        
        return True, ""
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取mTSP的默认配置"""
        config = super().get_default_config()
        config.update({
            'num_agents': 5,
            'cost_type': 'minmax',
        })
        return config
