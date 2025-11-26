"""
RL4CO 训练模块
包含强化学习训练的核心逻辑和回调函数
"""

from .training_functions import real_rl4co_training, ProgressCallback, create_route_animation

__all__ = ['real_rl4co_training', 'ProgressCallback', 'create_route_animation']

