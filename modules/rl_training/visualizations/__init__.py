"""
RL4CO 可视化模块
提供不同问题类型的可视化函数
"""

from .common import create_training_curve_plot
from .tsp_viz import create_tsp_route_animation, create_tsp_comparison_plot
from .cvrp_viz import create_cvrp_route_animation, create_cvrp_comparison_plot

__all__ = [
    'create_training_curve_plot',
    'create_tsp_route_animation',
    'create_tsp_comparison_plot',
    'create_cvrp_route_animation',
    'create_cvrp_comparison_plot',
]



