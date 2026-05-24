"""
SPCTSP (Stochastic Prize Collecting TSP) 训练器

随机奖励收集旅行商问题 —— PCTSP 的随机版本：
- 每个节点有「期望奖励」(deterministic_prize，预先已知)
  和「真实奖励」(real_prize，访问后才揭晓)
- real_prize = uniform(0, 2 * deterministic_prize)，期望相同但有随机波动
- 这迫使策略依赖期望值决策，形成更鲁棒的路径规划
- 其余规则（惩罚、prize_required 约束、优化目标）与 PCTSP 完全相同
"""

import os
import torch
import numpy as np
from .pctsp_trainer import PCTSPTrainer
from .visualizations.spctsp_viz import (
    create_spctsp_comparison_plot,
    create_spctsp_route_animation,
)


class SPCTSPTrainer(PCTSPTrainer):
    """
    SPCTSP（Stochastic PCTSP，随机奖励收集旅行商问题）训练器

    继承自 PCTSPTrainer，仅在以下方面不同：
    1. 使用 SPCTSPEnv（_stochastic=True）
    2. 可视化额外展示 deterministic_prize vs real_prize 的差异
    3. 日志提示信息区分为 SPCTSP

    参数（与 PCTSPTrainer 完全相同）：
        num_loc (int): 客户节点数量（不含 depot），默认 20
        penalty_factor (float): 惩罚缩放系数，默认 3.0
        prize_required (float): 最低奖励要求，默认 1.0
    """

    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
        # 覆盖 problem_type，确保检查点/文件名正确
        self.problem_type = 'spctsp'
        self.send_message('info', '🎲 SPCTSP 模式：奖励在访问节点后才揭晓（随机性）')
        self.send_message('info', '   期望奖励 (deterministic_prize) 预先可见，真实奖励 (real_prize) 访问后才知')
        self.load_custom_dataset()

    def _inject_custom_data(self, td):
        data = self.custom_dataset_data
        coords = torch.tensor(data['coordinates'], dtype=torch.float32)

        if data.get('depot'):
            depot = torch.tensor(data['depot'], dtype=torch.float32)
        else:
            depot = td['locs'][0, 0].cpu()

        locs = torch.cat([depot.unsqueeze(0), coords], dim=0)
        td['locs'] = locs.unsqueeze(0).to(self.device)

        if data.get('prizes'):
            prizes = data['prizes']
            det_prize = torch.tensor(prizes, dtype=torch.float32)          # [N]
            td['deterministic_prize'] = det_prize.unsqueeze(0).to(self.device)
            real_prize = torch.tensor([0.0] + prizes, dtype=torch.float32) # [N+1]
            td['real_prize'] = real_prize.unsqueeze(0).to(self.device)

        if data.get('penalties'):
            penalty = torch.tensor([0.0] + data['penalties'], dtype=torch.float32)
            td['penalty'] = penalty.unsqueeze(0).to(self.device)

        return td

    def initialize_environment(self):
        """
        初始化 SPCTSP 环境（_stochastic=True）

        与 PCTSPEnv 的唯一区别：
            real_prize ≠ deterministic_prize
            real_prize[i] ~ Uniform(0, 2 * deterministic_prize[i])
            期望值相同，但有随机波动

        TensorDict 关键键：
            locs                [B, num_loc+1, 2]
            deterministic_prize [B, num_loc]      期望奖励（预先已知，不含 depot）
            real_prize          [B, num_loc+1]    真实奖励（含 depot=0，访问时揭晓）
            penalty             [B, num_loc+1]    惩罚值（含 depot=0）
            prize_required      [B]               最低奖励门槛
        """
        try:
            from rl4co.envs.routing import SPCTSPEnv

            generator_params = {
                'num_loc': self.pctsp_num_loc,
                'penalty_factor': self.penalty_factor,
                'prize_required': self.prize_required,
            }

            env = SPCTSPEnv(generator_params=generator_params)

            self.send_message('info', '✅ SPCTSP 环境初始化成功（随机奖励模式）')
            self.send_message('info', f'  - 节点数 (含depot): {self.pctsp_num_loc + 1}')
            self.send_message('info', f'  - 惩罚系数: {self.penalty_factor}')
            self.send_message('info', f'  - 最低奖励要求: {self.prize_required}')
            self.send_message('info', '  - 奖励类型: 随机（real_prize ≠ deterministic_prize）')

            return env

        except Exception as e:
            self.send_message('error', f'❌ SPCTSP 环境初始化失败: {str(e)}')
            raise

    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """
        生成 SPCTSP 可视化结果

        在 PCTSP 可视化基础上额外包括：
        3. 随机奖励对比图：展示每个节点 deterministic_prize vs real_prize 的差异
        """
        self.send_message('info', '🎨 开始生成 SPCTSP 可视化（含随机奖励对比）...')

        animation_paths = []
        plot_paths = []

        try:
            if self.custom_dataset_data:
                td = env.reset(batch_size=[1]).to(self.device)
                td = self._inject_custom_data(td)
                num_test = 1
                num_vis = 1
                self.send_message('info', f'✅ 在上传的SPCTSP数据集上进行测试（{self.num_loc}个客户节点）')
            else:
                num_test = min(3, self.batch_size)
                num_vis = min(3, num_test)
                td = env.reset(batch_size=[num_test])

            model.eval()
            with torch.no_grad():
                out = model(td.clone(), phase="test", decode_type="greedy", return_actions=True)

            # SPCTSP 特有：同时提取 deterministic_prize 和 real_prize
            locs_all = td['locs'].cpu().numpy()                        # [B, num_loc+1, 2]
            det_prizes_all = td['deterministic_prize'].cpu().numpy()   # [B, num_loc]（不含depot）
            real_prizes_all = td['real_prize'].cpu().numpy()           # [B, num_loc+1]（含depot=0）
            penalties_all = td['penalty'].cpu().numpy()                # [B, num_loc+1]（含depot=0）

            depot_all = locs_all[:, 0, :]          # [B, 2]
            customer_locs = locs_all[:, 1:, :]     # [B, num_loc, 2]
            customer_real_prizes = real_prizes_all[:, 1:]   # [B, num_loc]
            customer_penalties = penalties_all[:, 1:]       # [B, num_loc]
            # deterministic_prize 本身就不含 depot
            customer_det_prizes = det_prizes_all             # [B, num_loc]

            actions_all = out['actions'].cpu().numpy()
            rewards_raw = out.get('reward', out.get('cost', None))
            rewards_all = rewards_raw.cpu().numpy() if rewards_raw is not None else np.zeros(num_test)

            for i in range(num_vis):
                try:
                    depot = depot_all[i].flatten()
                    locs = customer_locs[i]
                    real_prizes = customer_real_prizes[i].flatten()
                    det_prizes = customer_det_prizes[i].flatten()
                    penalties = customer_penalties[i].flatten()
                    actions = actions_all[i]

                    reward_val = rewards_all[i]
                    if isinstance(reward_val, np.ndarray):
                        reward_val = float(reward_val.flatten()[0]) if reward_val.size > 0 else 0.0
                    reward_scalar = float(reward_val)

                    # 路线对比图（含随机奖励差异信息）
                    try:
                        plot_path = create_spctsp_comparison_plot(
                            depot=depot,
                            locs=locs,
                            det_prizes=det_prizes,
                            real_prizes=real_prizes,
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

                            # 保存文件记录到数据库
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

                    # 路线动画（帧标题含奖励对比信息）
                    try:
                        anim_path = create_spctsp_route_animation(
                            depot=depot,
                            locs=locs,
                            det_prizes=det_prizes,
                            real_prizes=real_prizes,
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

                            # 保存文件记录到数据库
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

            self.send_message('info', f'🎉 SPCTSP 可视化完成: {len(animation_paths)} 个动画, {len(plot_paths)} 个对比图')

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
                    self.send_message('info', f'  ✓ checkpoint 记录已保存')
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
        """返回 SPCTSP 训练摘要"""
        summary = super().get_training_summary()
        summary['problem_type'] = 'spctsp'
        summary['stochastic'] = True
        return summary


def train_spctsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    SPCTSP 训练入口函数

    参数:
        config: 训练配置字典
        session_id: 训练会话 ID
        user_id: 用户 ID
        queue: 消息队列
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    trainer = SPCTSPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()


__all__ = ['SPCTSPTrainer', 'train_spctsp']
