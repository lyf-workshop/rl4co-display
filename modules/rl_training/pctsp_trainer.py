"""
PCTSP (Prize Collecting TSP) 训练器

奖励收集旅行商问题：
- 每个节点有奖励值（prize）和惩罚值（penalty）
- 必须收集至少 prize_required 的奖励才能返回 depot
- 未访问节点会产生惩罚
- 目标：最大化（已节省的惩罚 - 路径长度）
"""

import os
import torch
import numpy as np
from typing import Optional
from .base_trainer import BaseTrainer
from .visualizations.pctsp_viz import (
    create_pctsp_route_animation,
    create_pctsp_comparison_plot
)


class PCTSPTrainer(BaseTrainer):
    """
    PCTSP（Prize Collecting TSP，奖励收集旅行商问题）训练器

    问题特性：
    - 每个节点有奖励值（prize）和惩罚值（penalty，跳过时产生）
    - 约束：收集的总奖励 >= prize_required（默认 1.0）
    - 无需访问所有节点，选择性跳过（但会承担惩罚）
    - 目标：最大化 saved_penalties - (tour_length + remaining_penalties)

    参数：
        num_loc (int): 客户节点数量（不含 depot），默认 20
        penalty_factor (float): 惩罚缩放系数，控制惩罚大小，默认 3.0
        prize_required (float): 必须收集的最低奖励，默认 1.0
    """

    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)

        self.pctsp_num_loc = int(config.get('num_loc', 20))
        self.penalty_factor = float(config.get('penalty_factor', 3.0))
        self.prize_required = float(config.get('prize_required', 1.0))

        # 覆盖基类的 num_loc 供通用逻辑使用
        self.num_loc = self.pctsp_num_loc

        # 校验参数范围
        if self.penalty_factor < 0.1:
            self.send_message('warning', f'⚠️ penalty_factor={self.penalty_factor} 过小，使用默认值 3.0')
            self.penalty_factor = 3.0
        if not (0.1 <= self.prize_required <= 5.0):
            self.send_message('warning', f'⚠️ prize_required={self.prize_required} 超出合理范围，使用默认值 1.0')
            self.prize_required = 1.0

        self.send_message('info', f'📋 PCTSP 配置: {self.pctsp_num_loc} 个客户节点')
        self.send_message('info', f'📋 惩罚系数: {self.penalty_factor}, 最低奖励要求: {self.prize_required}')

    def initialize_environment(self):
        """
        初始化 PCTSP 环境

        locs 包含 depot（索引 0）+ 客户节点（1..num_loc），共 num_loc+1 个位置
        TensorDict 关键键：
            locs       [B, num_loc+1, 2]  含 depot 的所有节点坐标
            real_prize [B, num_loc+1]     各节点奖励（depot=0）
            penalty    [B, num_loc+1]     各节点惩罚（depot=0）
            prize_required               最低奖励门槛
        """
        try:
            from rl4co.envs.routing import PCTSPEnv

            generator_params = {
                'num_loc': self.pctsp_num_loc,
                'penalty_factor': self.penalty_factor,
                'prize_required': self.prize_required,
            }

            env = PCTSPEnv(generator_params=generator_params)

            self.send_message('info', f'✅ PCTSP 环境初始化成功')
            self.send_message('info', f'  - 节点数 (含depot): {self.pctsp_num_loc + 1}')
            self.send_message('info', f'  - 惩罚系数: {self.penalty_factor}')
            self.send_message('info', f'  - 最低奖励要求: {self.prize_required}')

            return env

        except Exception as e:
            self.send_message('error', f'❌ PCTSP 环境初始化失败: {str(e)}')
            raise

    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """
        生成 PCTSP 可视化结果

        包括：
        1. 路线对比图（PNG）：节点大小=奖励，边框=惩罚；已/未访问区分
        2. 路线动画（GIF）：逐步展示访问过程

        Returns:
            包含可视化文件路径的字典
        """
        self.send_message('info', '🎨 开始生成 PCTSP 可视化...')

        animation_paths = []
        plot_paths = []

        try:
            num_test = min(3, self.batch_size)
            num_vis = min(3, num_test)

            td = env.reset(batch_size=[num_test])

            model.eval()
            with torch.no_grad():
                out = model(td.clone(), phase="test", decode_type="greedy", return_actions=True)

            # 提取坐标：locs [B, num_loc+1, 2]，索引 0 是 depot
            locs_all = td['locs'].cpu().numpy()        # [B, num_loc+1, 2]
            prizes_all = td['real_prize'].cpu().numpy()  # [B, num_loc+1]，含 depot(=0)
            penalties_all = td['penalty'].cpu().numpy()  # [B, num_loc+1]，含 depot(=0)

            depot_all = locs_all[:, 0, :]              # [B, 2]
            customer_locs = locs_all[:, 1:, :]         # [B, num_loc, 2]
            customer_prizes = prizes_all[:, 1:]         # [B, num_loc]
            customer_penalties = penalties_all[:, 1:]   # [B, num_loc]

            actions_all = out['actions'].cpu().numpy()  # [B, seq_len]
            rewards_raw = out.get('reward', out.get('cost', None))
            rewards_all = rewards_raw.cpu().numpy() if rewards_raw is not None else np.zeros(num_test)

            for i in range(num_vis):
                try:
                    depot = depot_all[i].flatten()
                    locs = customer_locs[i]
                    prizes = customer_prizes[i].flatten()
                    penalties = customer_penalties[i].flatten()
                    actions = actions_all[i]

                    reward_val = rewards_all[i]
                    if isinstance(reward_val, np.ndarray):
                        reward_val = float(reward_val.flatten()[0]) if reward_val.size > 0 else 0.0
                    reward_scalar = float(reward_val)

                    # 生成对比图
                    try:
                        plot_path = create_pctsp_comparison_plot(
                            depot=depot,
                            locs=locs,
                            prizes=prizes,
                            penalties=penalties,
                            actions=actions,
                            model_reward=reward_scalar,
                            save_dir=self.user_plots_dir,
                            instance_id=i + 1
                        )
                        if plot_path and os.path.exists(plot_path):
                            plot_filename = os.path.basename(plot_path)
                            url = f'/static/model_plots/user_{self.user_id}/{plot_filename}'
                            plot_paths.append(url)
                            self.send_message('info', f'  ✓ 对比图 {i+1} 生成成功')

                            if self.bg_file_manager:
                                try:
                                    self.bg_file_manager.save_file_record(
                                        user_id=self.user_id,
                                        session_id=self.session_id,
                                        filename=plot_filename,
                                        file_type='comparison',
                                        file_path=plot_path
                                    )
                                except Exception as db_err:
                                    self.send_message('info', f'  ⚠️ 对比图数据库记录失败: {str(db_err)}')
                    except Exception as e:
                        self.send_message('info', f'  ✗ 对比图 {i+1} 失败: {str(e)}')

                    # 生成动画
                    try:
                        anim_path = create_pctsp_route_animation(
                            depot=depot,
                            locs=locs,
                            prizes=prizes,
                            penalties=penalties,
                            actions=actions,
                            model_reward=reward_scalar,
                            save_dir=self.user_plots_dir,
                            instance_id=i + 1
                        )
                        if anim_path and os.path.exists(anim_path):
                            anim_filename = os.path.basename(anim_path)
                            url = f'/static/model_plots/user_{self.user_id}/{anim_filename}'
                            animation_paths.append(url)
                            self.send_message('info', f'  ✓ 动画 {i+1} 生成成功')

                            if self.bg_file_manager:
                                try:
                                    self.bg_file_manager.save_file_record(
                                        user_id=self.user_id,
                                        session_id=self.session_id,
                                        filename=anim_filename,
                                        file_type='animation',
                                        file_path=anim_path
                                    )
                                except Exception as db_err:
                                    self.send_message('info', f'  ⚠️ 动画数据库记录失败: {str(db_err)}')
                    except Exception as e:
                        self.send_message('info', f'  ✗ 动画 {i+1} 失败: {str(e)}')

                except Exception as e:
                    self.send_message('info', f'  ⚠️ 实例 {i+1} 处理失败: {str(e)}')
                    continue

            self.send_message('info', f'🎉 PCTSP 可视化完成: {len(animation_paths)} 个动画, {len(plot_paths)} 个对比图')

            # 保存 checkpoint 记录到数据库
            if self.bg_file_manager and checkpoint_path:
                try:
                    checkpoint_filename = os.path.basename(checkpoint_path)
                    self.bg_file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=checkpoint_filename,
                        file_type='checkpoint',
                        file_path=checkpoint_path
                    )
                    self.send_message('info', '  ✓ checkpoint 记录已保存')
                except Exception as db_err:
                    self.send_message('info', f'  ⚠️ checkpoint 数据库记录失败: {str(db_err)}')

        except Exception as e:
            self.send_message('error', f'❌ 可视化生成失败: {str(e)}')
            import traceback
            traceback.print_exc()

        return {
            'animation_paths': animation_paths,
            'plot_paths': plot_paths,
            'training_curve': self.training_status[self.session_id].get('plot_url', ''),
            'checkpoint_path': checkpoint_path
        }

    def get_training_summary(self):
        """返回 PCTSP 训练摘要"""
        summary = super().get_training_summary()
        summary.update({
            'num_locations': self.pctsp_num_loc,
            'penalty_factor': self.penalty_factor,
            'prize_required': self.prize_required,
        })
        return summary


def train_pctsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    PCTSP 训练入口函数

    参数:
        config: 训练配置字典
        session_id: 训练会话 ID
        user_id: 用户 ID
        queue: 消息队列
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    trainer = PCTSPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()


__all__ = ['PCTSPTrainer', 'train_pctsp']
