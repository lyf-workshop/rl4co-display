"""
OP (Orienteering Problem) 训练器

定向问题：在路径长度限制内最大化收集的奖励
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from .base_trainer import BaseTrainer
from .visualizations.op_viz import (
    create_op_route_animation,
    create_op_comparison_plot
)


class OPTrainer(BaseTrainer):
    """
    OP（Orienteering Problem，定向问题）训练器
    
    问题特性：
    - 每个地点有一个奖励值（prize）
    - 总路径长度不能超过 max_length
    - 不需要访问所有地点（选择性访问）
    - 目标：最大化收集的奖励
    
    参数：
        num_loc (int): 地点数量（不含 depot）
        max_length (float): 最大路径长度
        prize_type (str): 奖励类型（'dist'/'unif'/'const'）
    """
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
        """初始化 OP 训练器"""
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
        
        # 提取 OP 特有参数（从 config 直接获取）
        self.op_num_loc = int(config.get('num_loc', 20))
        
        # 最大路径长度（如果未指定，使用预定义值）
        max_length_defaults = {20: 2.0, 50: 3.0, 100: 4.0}
        default_max_length = max_length_defaults.get(self.op_num_loc, 2.0)
        self.max_length = float(config.get('max_length', default_max_length))
        
        # 奖励类型
        self.prize_type = config.get('prize_type', 'dist')
        if self.prize_type not in ['dist', 'unif', 'const']:
            self.send_message('warning', f'⚠️ 未知的奖励类型 {self.prize_type}，使用默认值 dist')
            self.prize_type = 'dist'
        
        # 发送配置信息
        self.send_message('info', f'📋 OP配置: {self.op_num_loc}个地点, 最大路径长度: {self.max_length}')
        self.send_message('info', f'📋 奖励类型: {self.prize_type}')
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
            prize = torch.tensor([0.0] + data['prizes'], dtype=torch.float32)
            td['prize'] = prize.unsqueeze(0).to(self.device)

        return td

    def initialize_environment(self):
        """
        初始化 OP 环境
        
        使用 RL4CO 的 OPEnv
        """
        try:
            from rl4co.envs.routing import OPEnv
            
            # 构建环境参数
            generator_params = {
                'num_loc': self.op_num_loc,
                'prize_type': self.prize_type,
            }
            
            # 如果用户指定了 max_length，传递给环境
            if 'max_length' in self.config:
                generator_params['max_length'] = self.max_length
            
            # 创建环境
            env = OPEnv(
                generator_params=generator_params,
                prize_type=self.prize_type
            )
            
            self.send_message('info', f'✅ OP 环境初始化成功')
            self.send_message('info', f'  - 环境名称: {env.name}')
            self.send_message('info', f'  - 地点数量: {self.op_num_loc}')
            self.send_message('info', f'  - 最大路径长度: {self.max_length}')
            self.send_message('info', f'  - 奖励类型: {self.prize_type}')
            
            return env
            
        except Exception as e:
            self.send_message('error', f'❌ OP 环境初始化失败: {str(e)}')
            raise
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """
        生成 OP 可视化结果
        
        包括：
        1. 路线动画（GIF）：展示访问路径和奖励收集
        2. 对比图（PNG）：对比模型解和贪心解
        
        Args:
            env: 环境实例
            model: 训练好的模型
            trainer: Lightning trainer
            checkpoint_path: 检查点路径
            
        Returns:
            包含可视化文件路径的字典
        """
        self.send_message('info', f'🎨 开始生成OP可视化（定向问题路线）...')
        
        animation_paths = []
        plot_paths = []
        
        try:
            # 定义测试实例数量和可视化数量
            if self.custom_dataset_data:
                td = env.reset(batch_size=[1]).to(self.device)
                td = self._inject_custom_data(td)
                num_test_instances = 1
                num_visualizations = 1
                self.send_message('info', f'✅ 在上传的OP数据集上进行测试（{self.num_loc}个地点）')
            else:
                num_test_instances = min(3, self.batch_size)
                num_visualizations = min(3, num_test_instances)
                # 生成测试数据
                td = env.reset(batch_size=[num_test_instances])
            
            # 使用模型进行推理
            model.eval()
            with torch.no_grad():
                out = model(td.clone(), phase="test", decode_type="greedy", return_actions=True)
            
            # 提取数据
            locs = td['locs'].cpu().numpy()  # [batch, num_loc+1, 2] (包含depot)
            prize = td['prize'].cpu().numpy()  # [batch, num_loc+1]
            max_length_array = td['max_length'].cpu().numpy()  # [batch] or [batch, 1]
            
            # 处理 max_length 的维度
            if max_length_array.ndim == 2:
                max_length_array = max_length_array.squeeze()  # [batch]
            
            # 分离 depot 和其他地点
            depot = locs[:, 0, :]  # [batch, 2]
            customer_locs = locs[:, 1:, :]  # [batch, num_loc, 2]
            customer_prize = prize[:, 1:]  # [batch, num_loc]
            
            actions = out['actions'].cpu().numpy()  # [batch, seq_len]
            rewards = out.get('reward', out.get('cost', None))
            
            if rewards is not None:
                rewards = rewards.cpu().numpy()  # OP 中 reward 是正的（最大化）
            else:
                rewards = np.zeros(locs.shape[0])
            
            # 为每个实例生成可视化
            for i in range(num_visualizations):
                try:
                    # 提取单个实例的数据，确保都是正确的维度和类型
                    instance_depot = depot[i].flatten()  # [2] 确保是一维数组
                    instance_locs = customer_locs[i]  # [num_loc, 2]
                    instance_prize = customer_prize[i].flatten()  # [num_loc] 确保是一维数组
                    instance_actions = actions[i]  # [seq_len]
                    
                    # 提取奖励标量值 - 确保转换为 Python float
                    reward_val = rewards[i]
                    if isinstance(reward_val, np.ndarray):
                        reward_val = reward_val.flatten()[0] if reward_val.size > 0 else 0.0
                    instance_reward = float(reward_val)
                    
                    # 提取 max_length 的标量值 - 确保转换为 Python float
                    ml_val = max_length_array[i] if max_length_array.ndim > 0 else max_length_array
                    if isinstance(ml_val, np.ndarray):
                        ml_val = ml_val.flatten()[0] if ml_val.size > 0 else self.max_length
                    instance_max_length = float(ml_val)
                    
                    # 生成动画
                    try:
                        animation_path = create_op_route_animation(
                            depot=instance_depot,
                            locs=instance_locs,
                            prize=instance_prize,
                            actions=instance_actions,
                            total_prize=instance_reward,
                            max_length=instance_max_length,
                            save_dir=self.user_plots_dir,
                            instance_id=i+1
                        )
                        if animation_path and os.path.exists(animation_path):
                            anim_filename = os.path.basename(animation_path)
                            url_path = f'/static/model_plots/user_{self.user_id}/{anim_filename}'
                            animation_paths.append(url_path)
                            self.send_message('info', f'  ✓ 动画 {i+1} 生成成功')

                            if self.bg_file_manager:
                                try:
                                    self.bg_file_manager.save_file_record(
                                        user_id=self.user_id,
                                        session_id=self.session_id,
                                        filename=anim_filename,
                                        file_type='animation',
                                        file_path=animation_path
                                    )
                                except Exception as db_err:
                                    self.send_message('info', f'  ⚠️ 动画数据库记录失败: {str(db_err)}')
                    except Exception as e:
                        self.send_message('info', f'  ✗ 动画 {i+1} 生成失败: {str(e)}')
                    
                    # 生成对比图
                    try:
                        plot_path = create_op_comparison_plot(
                            depot=instance_depot,
                            locs=instance_locs,
                            prize=instance_prize,
                            actions=instance_actions,
                            model_prize=instance_reward,
                            max_length=instance_max_length,
                            save_dir=self.user_plots_dir,
                            instance_id=i+1
                        )
                        if plot_path and os.path.exists(plot_path):
                            plot_filename = os.path.basename(plot_path)
                            url_path = f'/static/model_plots/user_{self.user_id}/{plot_filename}'
                            plot_paths.append(url_path)
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
                        self.send_message('info', f'  ✗ 对比图 {i+1} 生成失败: {str(e)}')
                        
                except Exception as e:
                    self.send_message('info', f'  ⚠️ 实例 {i+1} 处理失败: {str(e)}')
                    continue
            
            self.send_message('info', f'🎉 OP可视化完成: {len(animation_paths)}个动画, {len(plot_paths)}个对比图')

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
        
        # 返回结果字典
        return {
            'animation_paths': animation_paths,
            'plot_paths': plot_paths,
            'training_curve': self.training_status[self.session_id].get('plot_url', ''),
            'checkpoint_path': checkpoint_path
        }
    
    def get_training_summary(self):
        """
        获取 OP 训练摘要信息
        
        Returns:
            包含训练配置的字典
        """
        summary = super().get_training_summary()
        
        # 添加 OP 特有信息
        summary.update({
            'num_locations': self.op_num_loc,
            'max_length': self.max_length,
            'prize_type': self.prize_type
        })
        
        return summary


def train_op(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    OP训练入口函数
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    trainer = OPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()


__all__ = ['OPTrainer', 'train_op']
