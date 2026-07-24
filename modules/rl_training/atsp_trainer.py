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
        """为三个随机 ATSP 实例生成费用矩阵对比图和路径动画。"""
        trainer.save_checkpoint(checkpoint_path)
        self._save_file_record(os.path.basename(checkpoint_path), 'checkpoint', checkpoint_path)
        self.send_message('info', f'检查点已保存: {checkpoint_path}')

        plot_paths = []
        animation_paths = []
        num_test_instances = 3

        # 一次批量推理三个不同的随机费用矩阵，避免重复推理同一实例。
        try:
            device = next(model.parameters()).device
            model.eval()
            policy = model.policy.to(device)
            policy.eval()
            td_init = env.reset(batch_size=[num_test_instances]).to(device)

            untrained_policy = self.create_untrained_policy_copy(model)
            with torch.no_grad():
                out_random = self._run_policy(
                    untrained_policy,
                    td_init.clone(),
                    env,
                    phase='test',
                    decode_type='greedy',
                    return_actions=True,
                )
                out_greedy = self._run_policy(
                    policy,
                    td_init.clone(),
                    env,
                    phase='test',
                    decode_type='greedy',
                    return_actions=True,
                )

            cost_matrices = td_init['cost_matrix'].cpu()
            actions_random_batch = out_random['actions'].cpu()
            actions_greedy_batch = out_greedy['actions'].cpu()
        except Exception as e:
            logger.error(f"ATSP 模型推理失败: {e}", exc_info=True)
            self.send_message('warning', f'⚠️ 模型推理失败，跳过可视化: {str(e)}')
            return {
                'plot_paths': plot_paths,
                'animation_paths': animation_paths,
                'training_curve': self.training_status[self.session_id].get('plot_url', ''),
                'checkpoint_path': checkpoint_path,
            }

        for i in range(num_test_instances):
            instance_id = i + 1
            cost_matrix = cost_matrices[i]
            actions_random = actions_random_batch[i]
            actions_greedy = actions_greedy_batch[i]

            try:
                self.send_message('info', f'正在生成 ATSP 实例 {instance_id}/3 费用矩阵热力图...')
                plot_filename = (
                    f"atsp_comparison_{self.session_id[:8]}_inst{instance_id}.png"
                )
                plot_path = os.path.join(self.user_plots_dir, plot_filename)

                result = create_atsp_comparison_plot(
                    cost_matrix,
                    actions_random,
                    actions_greedy,
                    plot_path,
                    title=f"ATSP训练前后对比（实例{instance_id}，{self.num_loc}节点）",
                )

                self._save_file_record(plot_filename, 'plot', plot_path)
                plot_paths.append(
                    f"/static/model_plots/user_{self.user_id}/{plot_filename}"
                )
                self.send_message(
                    'info',
                    f'✅ ATSP实例 {instance_id} 热力图完成: '
                    f'总费用 {result["cost_random"]:.4f} → '
                    f'{result["cost_trained"]:.4f}，'
                    f'改进 {result["improvement"]:.2f}%',
                )
            except Exception as e:
                logger.error(
                    f"生成 ATSP 实例 {instance_id} 对比图失败: {e}",
                    exc_info=True,
                )
                self.send_message(
                    'warning',
                    f'⚠️ ATSP实例 {instance_id} 对比图生成失败: {str(e)}',
                )

            try:
                self.send_message('info', f'正在生成 ATSP 实例 {instance_id}/3 路径动态图...')
                anim_filename = (
                    f"atsp_animation_{self.session_id[:8]}_inst{instance_id}.gif"
                )
                anim_path = os.path.join(self.user_plots_dir, anim_filename)

                create_atsp_route_animation(
                    cost_matrix,
                    actions_greedy,
                    anim_path,
                    title=(
                        f"ATSP路径构建过程（实例{instance_id}，"
                        f"{self.num_loc}节点，训练后贪心解）"
                    ),
                    fps=2,
                )

                self._save_file_record(anim_filename, 'animation', anim_path)
                animation_paths.append(
                    f"/static/model_plots/user_{self.user_id}/{anim_filename}"
                )
                self.send_message('info', f'✅ ATSP实例 {instance_id} 动态GIF生成完成')
            except Exception as e:
                logger.error(
                    f"生成 ATSP 实例 {instance_id} 动画失败: {e}",
                    exc_info=True,
                )
                self.send_message(
                    'warning',
                    f'⚠️ ATSP实例 {instance_id} 动画生成失败: {str(e)}',
                )

        self.send_message(
            'info',
            f'🎉 ATSP可视化完成: {len(plot_paths)}张对比图，'
            f'{len(animation_paths)}个动画',
        )
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
