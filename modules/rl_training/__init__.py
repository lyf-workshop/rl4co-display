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
import logging

logger = logging.getLogger('rl4co_display')

# base_trainer 依赖 torch；在没有安装 torch 的环境（如纯测试环境）下优雅降级
try:
    from .base_trainer import BaseTrainer, ProgressCallback
except Exception as _base_err:
    logger.warning(f"base_trainer 导入失败（torch 未安装？）: {_base_err}")
    BaseTrainer = None
    ProgressCallback = None

# 以下 Trainer 依赖 rl4co → torchrl，在 Windows 上可能因原生 DLL 问题导入失败。
# 用 try/except 隔离，确保 Flask 应用和不依赖 rl4co 的测试不受影响。
try:
    from .tsp_trainer import TSPTrainer, train_tsp
    from .atsp_trainer import ATSPTrainer, train_atsp
    from .mtsp_trainer import MTSPTrainer, train_mtsp
    from .cvrp_trainer import CVRPTrainer, train_cvrp
    from .sdvrp_trainer import SDVRPTrainer, train_sdvrp
    from .vrptw_trainer import VRPTWTrainer, train_vrptw
    from .pdp_trainer import PDPTrainer, train_pdp
    from .op_trainer import OPTrainer, train_op
    from .pctsp_trainer import PCTSPTrainer, train_pctsp
    from .spctsp_trainer import SPCTSPTrainer, train_spctsp
    from .ffsp_trainer import FFSPTrainer, train_ffsp
    _RL4CO_TRAINERS_AVAILABLE = True
except Exception as e:
    _RL4CO_TRAINERS_AVAILABLE = False
    logger.warning(f"RL4CO 训练器导入失败（将使用模拟训练模式）: {e}")
    # 定义占位符，避免 NameError
    TSPTrainer = ATSPTrainer = MTSPTrainer = CVRPTrainer = None
    SDVRPTrainer = VRPTWTrainer = PDPTrainer = OPTrainer = None
    PCTSPTrainer = SPCTSPTrainer = FFSPTrainer = None
    train_tsp = train_atsp = train_mtsp = train_cvrp = train_sdvrp = None
    train_vrptw = train_pdp = train_op = train_pctsp = train_spctsp = train_ffsp = None

# 向后兼容：提供统一的训练入口
def real_rl4co_training(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    统一的强化学习训练入口函数（根据问题类型自动路由）

    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列（用于推送进度）
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
        pause_event: threading.Event，set=运行，clear=暂停（可选）
    """
    if not _RL4CO_TRAINERS_AVAILABLE:
        training_status[session_id] = {'status': 'error', 'progress': 0}
        queue.put(json.dumps({
            'type': 'error',
            'message': 'RL4CO 训练器加载失败（可能是 torchrl DLL 兼容性问题），请检查环境配置'
        }))
        return

    problem_type = config.get('problem', 'tsp').lower()

    # 根据问题类型路由到对应的训练器
    if problem_type == 'tsp':
        train_tsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    elif problem_type == 'atsp':
        # ATSP：MatNet 需要 ATSPEnv（提供 cost_matrix），attention 模型用 TSP 训练器即可
        policy = config.get('model', 'attention').lower()
        if policy in ('matnet',):
            from .atsp_trainer import train_atsp
            train_atsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
        else:
            train_tsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    elif problem_type == 'mtsp':
        # mTSP - 多旅行商问题
        train_mtsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    elif problem_type == 'cvrp':
        train_cvrp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    elif problem_type == 'sdvrp':
        train_sdvrp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    elif problem_type == 'vrptw':
        train_vrptw(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    elif problem_type == 'pdp':
        # PDP - 取送货问题
        train_pdp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    elif problem_type == 'op':
        # OP - 定向问题
        train_op(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    elif problem_type == 'pctsp':
        # PCTSP - 奖励收集旅行商问题
        train_pctsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    elif problem_type == 'spctsp':
        # SPCTSP - 随机奖励收集旅行商问题
        train_spctsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    elif problem_type == 'ffsp':
        # FFSP - 柔性流水车间调度问题
        train_ffsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
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
            'message': f'暂不支持的问题类型: {problem_type}，请选择 TSP、ATSP、mTSP、CVRP、SDVRP、VRPTW、PDP、OP、PCTSP、SPCTSP 或 FFSP'
        }))


# 向后兼容：保留原有的动画创建函数（使用TSP可视化）
try:
    from .visualizations.tsp_viz import create_tsp_route_animation as create_route_animation
except Exception:
    create_route_animation = None

__all__ = [
    # 基类和通用组件
    'BaseTrainer',
    'ProgressCallback',
    
    # 具体问题训练器
    'TSPTrainer',
    'ATSPTrainer',
    'MTSPTrainer',
    'CVRPTrainer',
    'SDVRPTrainer',
    'VRPTWTrainer',
    'PDPTrainer',
    'OPTrainer',
    'PCTSPTrainer',
    'SPCTSPTrainer',
    'FFSPTrainer',
    
    # 训练入口函数
    'real_rl4co_training',
    'train_tsp',
    'train_atsp',
    'train_mtsp',
    'train_cvrp',
    'train_sdvrp',
    'train_vrptw',
    'train_pdp',
    'train_op',
    'train_pctsp',
    'train_spctsp',
    'train_ffsp',
    
    # 向后兼容
    'create_route_animation',
]
