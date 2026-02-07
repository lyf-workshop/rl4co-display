"""
RL4CO 训练模块 - 模块化重构版
按问题类型组织训练逻辑和可视化函数

模块结构:
    - base_trainer.py: 通用训练基类和回调函数
    - tsp_trainer.py: TSP问题专用训练器
    - cvrp_trainer.py: CVRP问题专用训练器
    - sdvrp_trainer.py: SDVRP问题专用训练器
    - visualizations/: 各问题类型的可视化函数
        - common.py: 通用可视化
        - tsp_viz.py: TSP可视化
        - cvrp_viz.py: CVRP可视化
        - sdvrp_viz.py: SDVRP可视化
"""

import json
from .base_trainer import BaseTrainer, ProgressCallback
from .tsp_trainer import TSPTrainer, train_tsp
from .mtsp_trainer import MTSPTrainer, train_mtsp
from .cvrp_trainer import CVRPTrainer, train_cvrp
from .sdvrp_trainer import SDVRPTrainer, train_sdvrp
from .vrptw_trainer import VRPTWTrainer, train_vrptw

# 向后兼容：提供统一的训练入口
def real_rl4co_training(config, session_id, user_id, queue, training_status, get_background_db_func):
    """
    统一的强化学习训练入口函数（根据问题类型自动路由）
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列（用于推送进度）
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    problem_type = config.get('problem', 'tsp').lower()
    
    # 根据问题类型路由到对应的训练器
    if problem_type == 'tsp':
        train_tsp(config, session_id, user_id, queue, training_status, get_background_db_func)
    elif problem_type == 'atsp':
        # ATSP使用TSP训练器（距离矩阵不对称，但训练流程相同）
        train_tsp(config, session_id, user_id, queue, training_status, get_background_db_func)
    elif problem_type == 'mtsp':
        # mTSP - 多旅行商问题
        train_mtsp(config, session_id, user_id, queue, training_status, get_background_db_func)
    elif problem_type == 'cvrp':
        train_cvrp(config, session_id, user_id, queue, training_status, get_background_db_func)
    elif problem_type == 'sdvrp':
        train_sdvrp(config, session_id, user_id, queue, training_status, get_background_db_func)
    elif problem_type == 'vrptw':
        train_vrptw(config, session_id, user_id, queue, training_status, get_background_db_func)
    else:
        # 未支持的问题类型 - 先初始化状态再设置错误
        training_status[session_id] = {
            'status': 'error',
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': 0,
            'message': f'暂不支持的问题类型: {problem_type}'
        }
        queue.put(json.dumps({
            'type': 'error',
            'message': f'暂不支持的问题类型: {problem_type}，请选择 TSP、ATSP、mTSP、CVRP、SDVRP 或 VRPTW'
        }))


# 向后兼容：保留原有的动画创建函数（使用TSP可视化）
from .visualizations.tsp_viz import create_tsp_route_animation as create_route_animation

__all__ = [
    # 基类和通用组件
    'BaseTrainer',
    'ProgressCallback',
    
    # 具体问题训练器
    'TSPTrainer',
    'MTSPTrainer',
    'CVRPTrainer',
    'SDVRPTrainer',
    'VRPTWTrainer',
    
    # 训练入口函数
    'real_rl4co_training',
    'train_tsp',
    'train_mtsp',
    'train_cvrp',
    'train_sdvrp',
    'train_vrptw',
    
    # 向后兼容
    'create_route_animation',
]
