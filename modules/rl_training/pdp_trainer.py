"""
PDP (Pickup and Delivery Problem) 训练器

基于 RL4CO 的 PDPEnv 环境实现
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from .base_trainer import BaseTrainer
from .visualizations.pdp_viz import (
    create_pdp_route_animation,
    create_pdp_comparison_plot
)


class PDPTrainer(BaseTrainer):
    """
    PDP（Pickup and Delivery Problem，取送货问题）训练器
    
    问题特性：
    - 每个需求包含一个取货点（pickup）和一个送货点（delivery）
    - 必须先访问取货点，再访问对应的送货点
    - 目标：最小化总路径长度
    
    参数：
        num_loc (int): 地点数量（不含 depot），必须是偶数
                      num_loc/2 个取货点 + num_loc/2 个送货点
        force_start_at_depot (bool): 是否强制从 depot 开始
    """
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
        """初始化 PDP 训练器"""
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
        
        # 提取 PDP 特有参数（从 config 直接获取，不是从 self.env_params）
        self.pdp_num_loc = int(config.get('num_loc', 20))
        self.force_start_at_depot = config.get('force_start_at_depot', False)
        if isinstance(self.force_start_at_depot, str):
            self.force_start_at_depot = self.force_start_at_depot.lower() == 'true'
        
        # 验证参数
        if self.pdp_num_loc % 2 != 0:
            raise ValueError(f"num_loc 必须是偶数（取货点和送货点成对），当前值: {self.pdp_num_loc}")
        
        self.num_pairs = self.pdp_num_loc // 2
        
        self.send_message('info', f'📋 PDP配置: {self.pdp_num_loc}个地点 ({self.num_pairs}对取送货), 强制从depot开始: {self.force_start_at_depot}')
        self.load_custom_dataset()
        if self.custom_dataset_data:
            self.pdp_num_loc = self.num_loc
            self.num_pairs = self.pdp_num_loc // 2

    def _inject_custom_data(self, td):
        data = self.custom_dataset_data
        coords = torch.tensor(data['coordinates'], dtype=torch.float32)  # [N, 2]

        if data.get('depot'):
            depot = torch.tensor(data['depot'], dtype=torch.float32)
        else:
            depot = td['depot'][0].cpu()  # PDP 有独立 depot key

        td['locs'] = coords.unsqueeze(0).to(self.device)   # [1, N, 2]
        td['depot'] = depot.unsqueeze(0).to(self.device)   # [1, 2]
        return td

    def initialize_environment(self):
        """
        初始化 PDP 环境
        
        使用 RL4CO 的 PDPEnv
        """
        try:
            from rl4co.envs.routing import PDPEnv
            
            # 构建环境参数
            generator_params = {
                'num_loc': self.pdp_num_loc,
            }
            
            # 创建环境
            env = PDPEnv(
                generator_params=generator_params,
                force_start_at_depot=self.force_start_at_depot
            )
            
            self.send_message('info', f'✅ PDP 环境初始化成功')
            self.send_message('info', f'  - 环境名称: {env.name}')
            self.send_message('info', f'  - 地点数量: {self.pdp_num_loc}')
            self.send_message('info', f'  - 取送货对数: {self.num_pairs}')
            
            return env
            
        except Exception as e:
            self.send_message('error', f'❌ PDP 环境初始化失败: {str(e)}')
            raise
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """
        生成 PDP 可视化结果
        
        包括：
        1. 路线动画（GIF）：展示取送货路线
        2. 对比图（PNG）：对比模型解和贪心解
        
        Args:
            env: 环境实例
            model: 训练好的模型
            trainer: Lightning trainer
            checkpoint_path: 检查点路径
            
        Returns:
            包含可视化文件路径的字典
        """
        self.send_message('info', f'🎨 开始生成PDP可视化（取送货路线）...')
        
        animation_paths = []
        plot_paths = []
        
        try:
            # 定义测试实例数量和可视化数量
            if self.custom_dataset_data:
                td = env.reset(batch_size=[1]).to(self.device)
                td = self._inject_custom_data(td)
                num_test_instances = 1
                num_visualizations = 1
                self.send_message('info', f'✅ 在上传的PDP数据集上进行测试（{self.num_loc}个节点，{self.num_pairs}对取送货）')
            else:
                num_test_instances = min(3, self.batch_size)  # 最多3个实例
                num_visualizations = min(3, num_test_instances)  # 最多生成3个可视化
                # 生成测试数据
                td = env.reset(batch_size=[num_test_instances])
            
            # 使用模型进行推理
            model.eval()
            with torch.no_grad():
                out = model(td.clone(), phase="test", decode_type="greedy", return_actions=True)
            
            # 提取数据
            locs = td['locs'].cpu().numpy()  # [batch, num_loc, 2]
            
            # 检查是否有 depot
            if 'depot' in td.keys():
                depot = td['depot'].cpu().numpy()  # [batch, 2]
            else:
                # 如果没有单独的 depot，取第一个位置作为 depot
                depot = locs[:, 0:1, :]  # [batch, 1, 2]
                locs = locs[:, 1:, :]  # 移除第一个位置
            
            actions = out['actions'].cpu().numpy()  # [batch, seq_len]
            costs = out.get('reward', out.get('cost', None))
            
            if costs is not None:
                costs = -costs.cpu().numpy()  # 转换为正的成本
            else:
                costs = np.zeros(locs.shape[0])
            
            # 为每个实例生成可视化
            for i in range(num_visualizations):
                try:
                    # 提取单个实例的数据
                    instance_depot = depot[i]  # [2] or [1, 2]
                    if instance_depot.ndim == 2:
                        instance_depot = instance_depot[0]  # 取第一个
                    
                    instance_locs = locs[i]  # [num_loc, 2]
                    instance_actions = actions[i]  # [seq_len]
                    instance_cost = costs[i]
                    
                    # 生成动画
                    try:
                        animation_path = create_pdp_route_animation(
                            depot=instance_depot,
                            pickups=instance_locs[:self.num_pairs],  # 前一半是取货点
                            deliveries=instance_locs[self.num_pairs:],  # 后一半是送货点
                            actions=instance_actions,
                            cost=instance_cost,
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
                        self.send_message('info', f'  ⚠️ 动画 {i+1} 生成失败: {str(e)}')
                    
                    # 生成对比图
                    try:
                        plot_path = create_pdp_comparison_plot(
                            depot=instance_depot,
                            pickups=instance_locs[:self.num_pairs],
                            deliveries=instance_locs[self.num_pairs:],
                            actions=instance_actions,
                            model_cost=instance_cost,
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
                        self.send_message('info', f'  ⚠️ 对比图 {i+1} 生成失败: {str(e)}')
                        
                except Exception as e:
                    self.send_message('info', f'  ⚠️ 实例 {i+1} 处理失败: {str(e)}')
                    continue
            
            self.send_message('info', f'🎉 PDP可视化完成: {len(animation_paths)}个动画, {len(plot_paths)}个对比图')

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
        获取 PDP 训练摘要信息
        
        Returns:
            包含训练配置的字典
        """
        summary = super().get_training_summary()
        
        # 添加 PDP 特有信息
        summary.update({
            'num_locations': self.pdp_num_loc,
            'num_pickup_delivery_pairs': self.num_pairs,
            'force_start_at_depot': self.force_start_at_depot
        })
        
        return summary


def train_pdp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    PDP训练入口函数
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    trainer = PDPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()
