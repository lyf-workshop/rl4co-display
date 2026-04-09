"""
ATSP（非对称旅行商）专用训练器
MatNet 策略需要 ATSPEnv 提供的 cost_matrix；
attention / ptrnet 等坐标模型仍沿用 TSPTrainer。
"""

import os
import logging
import torch

logger = logging.getLogger('rl4co_display')

try:
    from rl4co.envs.routing.atsp.env import ATSPEnv
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    logger.warning("RL4CO 库未安装或版本过旧，ATSPEnv 不可用")

from .base_trainer import BaseTrainer
from .visualizations.tsp_viz import create_tsp_comparison_plot


class ATSPTrainer(BaseTrainer):
    """
    ATSP 训练器（MatNet 专用）

    使用 ATSPEnv，环境在 reset 时会随机生成非对称代价矩阵（cost_matrix），
    供 MatNet 的 init_embedding 使用。
    可视化沿用 TSP 对比图（坐标不可用时跳过）。
    """

    def initialize_environment(self):
        if not RL4CO_AVAILABLE:
            raise ImportError("RL4CO 库未安装，无法创建 ATSPEnv")

        env = ATSPEnv(generator_params={'num_loc': self.num_loc})
        self.send_message('info', f'✅ ATSPEnv 初始化完成 (num_loc={self.num_loc}，含 cost_matrix)')
        return env

    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """ATSP 没有 2D 坐标，跳过路线图，只保存 checkpoint 和训练曲线"""
        trainer.save_checkpoint(checkpoint_path)

        if self.bg_file_manager:
            try:
                self.bg_file_manager.save_file_record(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    filename=os.path.basename(checkpoint_path),
                    file_type='checkpoint',
                    file_path=checkpoint_path,
                )
            except Exception as e:
                logger.warning(f"保存 checkpoint 记录失败: {e}")

        self.send_message('info', f'检查点已保存: {checkpoint_path}')
        self.send_message('info', 'ℹ️ ATSP 为代价矩阵表示，跳过路线可视化')

        return {
            'plot_paths': [],
            'animation_paths': [],
            'training_curve': self.training_status[self.session_id].get('plot_url', ''),
            'checkpoint_path': checkpoint_path,
        }


def train_atsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """ATSP（MatNet）训练入口"""
    trainer = ATSPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()
