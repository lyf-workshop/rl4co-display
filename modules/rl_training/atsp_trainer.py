"""
ATSP（非对称旅行商）专用训练器
所有策略（MatNet / Attention）统一使用 ATSPEnv，
训练完成后生成费用矩阵热力图对比图。
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
from .visualizations.atsp_viz import create_atsp_comparison_plot, create_atsp_route_animation


class ATSPTrainer(BaseTrainer):
    """
    ATSP 训练器（支持 MatNet 和 Attention Model）

    统一使用 ATSPEnv，环境在 reset 时随机生成非对称代价矩阵（cost_matrix）。
    训练完成后生成费用矩阵热力图，对比随机策略与训练后贪心解的路径质量。
    """

    def initialize_environment(self):
        if not RL4CO_AVAILABLE:
            raise ImportError("RL4CO 库未安装，无法创建 ATSPEnv")

        env = ATSPEnv(generator_params={'num_loc': self.num_loc})
        self.send_message('info', f'✅ ATSPEnv 初始化完成 (num_loc={self.num_loc}，含 cost_matrix)')
        return env

    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """生成 ATSP 可视化结果（费用矩阵对比图 + 路径构建动态GIF）"""
        # ── 保存检查点 ──────────────────────────────────────────────────
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

        plot_paths = []
        animation_paths = []

        # ── 模型推理：获取 cost_matrix、随机解、贪心解 ──────────────────
        cost_matrix    = None
        actions_random = None
        actions_greedy = None
        try:
            policy  = model.policy.to(self.device)
            td_init = env.reset(batch_size=[1]).to(self.device)

            out_random = policy(
                td_init.clone(), phase="test", decode_type="sampling", return_actions=True
            )
            out_greedy = policy(
                td_init.clone(), phase="test", decode_type="greedy",   return_actions=True
            )

            cost_matrix    = td_init['cost_matrix'][0].cpu()
            actions_random = out_random['actions'][0].cpu()
            actions_greedy = out_greedy['actions'][0].cpu()

        except Exception as e:
            logger.error(f"ATSP 模型推理失败: {e}", exc_info=True)
            self.send_message('warning', f'⚠️ 模型推理失败，跳过可视化: {str(e)}')
            return {
                'plot_paths': plot_paths,
                'animation_paths': animation_paths,
                'training_curve': self.training_status[self.session_id].get('plot_url', ''),
                'checkpoint_path': checkpoint_path,
            }

        # ── 静态对比图（热力图）──────────────────────────────────────────
        try:
            self.send_message('info', '正在生成 ATSP 费用矩阵热力图...')
            plot_filename = f"atsp_comparison_{self.session_id[:8]}.png"
            plot_path = os.path.join(self.user_plots_dir, plot_filename)

            result = create_atsp_comparison_plot(
                cost_matrix, actions_random, actions_greedy, plot_path,
                title=f"ATSP训练前后对比（{self.num_loc}节点）"
            )

            if self.bg_file_manager:
                try:
                    self.bg_file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=plot_filename,
                        file_type='plot',
                        file_path=plot_path,
                    )
                except Exception as e:
                    logger.warning(f"保存对比图记录失败: {e}")

            plot_paths.append(f"/static/model_plots/user_{self.user_id}/{plot_filename}")
            self.send_message(
                'info',
                f'✅ ATSP热力图生成完成: 总费用 {result["cost_random"]:.4f} → '
                f'{result["cost_trained"]:.4f}，改进 {result["improvement"]:.2f}%'
            )
        except Exception as e:
            logger.error(f"生成 ATSP 对比图失败: {e}", exc_info=True)
            self.send_message('warning', f'⚠️ 对比图生成失败: {str(e)}')

        # ── 动态 GIF（路径构建过程）──────────────────────────────────────
        try:
            self.send_message('info', '正在生成 ATSP 路径构建动态图...')
            anim_filename = f"atsp_animation_{self.session_id[:8]}.gif"
            anim_path = os.path.join(self.user_plots_dir, anim_filename)

            create_atsp_route_animation(
                cost_matrix, actions_greedy, anim_path,
                title=f"ATSP路径构建过程（{self.num_loc}节点，训练后贪心解）",
                fps=2
            )

            if self.bg_file_manager:
                try:
                    self.bg_file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=anim_filename,
                        file_type='animation',
                        file_path=anim_path,
                    )
                except Exception as e:
                    logger.warning(f"保存动画记录失败: {e}")

            animation_paths.append(f"/static/model_plots/user_{self.user_id}/{anim_filename}")
            self.send_message('info', '✅ ATSP动态GIF生成完成')

        except Exception as e:
            logger.error(f"生成 ATSP 动画失败: {e}", exc_info=True)
            self.send_message('warning', f'⚠️ 动画生成失败: {str(e)}')

        return {
            'plot_paths': plot_paths,
            'animation_paths': animation_paths,
            'training_curve': self.training_status[self.session_id].get('plot_url', ''),
            'checkpoint_path': checkpoint_path,
        }


def train_atsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """ATSP（MatNet）训练入口"""
    trainer = ATSPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()
