"""
mTSP问题专用训练器
包含mTSP特有的训练逻辑和可视化生成
"""

import os
import json
import logging
import torch
import numpy as np
from datetime import datetime

logger = logging.getLogger('rl4co_display')

try:
    from rl4co.envs import MTSPEnv
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    logger.warning("RL4CO 库未安装或不支持 MTSPEnv")

from .base_trainer import BaseTrainer
from .visualizations.mtsp_viz import create_mtsp_route_animation, create_mtsp_comparison_plot


class MTSPTrainer(BaseTrainer):
    """mTSP问题训练器"""
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
        
        # mTSP特有参数
        self.num_agents = int(config.get('num_agents', 5))
        self.cost_type = config.get('cost_type', 'minmax')
        
        self.send_message('info', f'📋 mTSP配置: {self.num_agents}个代理, 优化目标={self.cost_type}')
    
    def get_problem_type(self):
        return 'mtsp'
    
    def initialize_environment(self):
        """初始化mTSP环境（必须实现的抽象方法）"""
        if not RL4CO_AVAILABLE:
            raise ImportError("RL4CO库未安装或不支持MTSPEnv，无法创建mTSP环境")
        
        env = MTSPEnv(
            generator_params={
                'num_loc': self.num_loc,
                'min_num_agents': self.num_agents,
                'max_num_agents': self.num_agents,  # 固定代理数量
            },
            cost_type=self.cost_type
        )
        
        self.send_message('info', f'✅ mTSP环境创建成功: {self.num_loc}个城市, {self.num_agents}个代理')
        return env
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """
        生成mTSP可视化（多代理路径）
        
        参数:
            env: mTSP环境
            model: 训练好的模型
            trainer: PyTorch Lightning训练器
            checkpoint_path: 检查点保存路径
        
        返回:
            dict: 包含可视化路径和训练信息的字典
        """
        try:
            self.send_message('info', '🎨 开始生成mTSP可视化（多代理路径）...')
            
            device = next(model.parameters()).device
            model.eval()
            
            # 生成测试数据
            num_test_instances = min(3, self.batch_size)  # 最多3个实例
            td = env.reset(batch_size=[num_test_instances])
            td = td.to(device)
            
            # 使用模型生成路径
            with torch.no_grad():
                out = model(td.clone(), phase='test', decode_type='greedy')
            
            # 提取动作和奖励
            actions = out['actions'].cpu().numpy()
            rewards = out['reward'].cpu().numpy()
            locs = td['locs'].cpu().numpy()
            
            # 为每个测试实例创建可视化
            animation_paths = []
            comparison_paths = []
            
            for i in range(num_test_instances):
                # 生成文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                anim_filename = f'mtsp_animation_{i+1}_{timestamp}.gif'
                comp_filename = f'mtsp_comparison_{i+1}_{timestamp}.png'
                
                anim_path = os.path.join(self.user_plots_dir, anim_filename)
                comp_path = os.path.join(self.user_plots_dir, comp_filename)
                
                # 创建路线动画
                try:
                    # ========== 修复维度问题 ==========
                    # locs 维度: (batch, num_cities, 2)
                    # 取单个实例时需要去掉 batch 维度
                    instance_locs = locs[i]  # shape: (num_cities, 2)
                    instance_actions = actions[i]  # shape: (action_length,)
                    
                    create_mtsp_route_animation(
                        td={'locs': torch.from_numpy(instance_locs)},  # ✅ 不保留batch维度
                        actions=instance_actions,
                        save_path=anim_path,
                        title=f'mTSP路线生成过程 - 问题{i+1} ({self.num_agents}个代理)',
                        fps=2
                    )
                    animation_paths.append(f"/static/model_plots/user_{self.user_id}/{anim_filename}")
                    self.send_message('info', f'✅ 动画 {i+1} 生成成功')
                    
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
                        except Exception as e:
                            logger.warning(f"保存动画记录失败: {str(e)}")
                except Exception as e:
                    self.send_message('info', f'⚠️ 动画 {i+1} 生成失败: {str(e)}')
                
                # 创建对比图
                try:
                    create_mtsp_comparison_plot(
                        td={'locs': torch.from_numpy(instance_locs)},  # ✅ 使用已处理的数据
                        actions=instance_actions,
                        save_path=comp_path,
                        cost=-rewards[i],  # 负奖励即为成本
                        title=f'mTSP路线对比图 - 问题{i+1}'
                    )
                    comparison_paths.append(f"/static/model_plots/user_{self.user_id}/{comp_filename}")
                    self.send_message('info', f'✅ 对比图 {i+1} 生成成功')
                    
                    # 保存文件记录到数据库
                    if self.bg_file_manager:
                        try:
                            self.bg_file_manager.save_file_record(
                                user_id=self.user_id,
                                session_id=self.session_id,
                                filename=comp_filename,
                                file_type='plot',
                                file_path=comp_path
                            )
                        except Exception as e:
                            logger.warning(f"保存对比图记录失败: {str(e)}")
                except Exception as e:
                    self.send_message('info', f'⚠️ 对比图 {i+1} 生成失败: {str(e)}')
            
            self.send_message('info', f'🎉 mTSP可视化完成: {len(animation_paths)}个动画, {len(comparison_paths)}个对比图')
            logger.debug("mTSP可视化完成: animations=%s, comparisons=%s", animation_paths, comparison_paths)

            # 保存检查点
            trainer.save_checkpoint(checkpoint_path)
            
            # 保存checkpoint文件记录到数据库
            if self.bg_file_manager:
                try:
                    checkpoint_filename = os.path.basename(checkpoint_path)
                    self.bg_file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=checkpoint_filename,
                        file_type='checkpoint',
                        file_path=checkpoint_path
                    )
                except Exception as e:
                    logger.warning(f"保存checkpoint记录失败: {str(e)}")
            
            self.send_message('info', f'检查点已保存: {checkpoint_path}')
            
            # 返回字典格式（注意：前端期望 plot_paths，不是 comparison_paths）
            return {
                'plot_paths': comparison_paths,  # ✅ 前端期望的字段名
                'animation_paths': animation_paths,
                'training_curve': self.training_status[self.session_id].get('plot_url', ''),
                'checkpoint_path': checkpoint_path
            }
            
        except Exception as e:
            self.send_message('info', f'❌ mTSP可视化失败: {str(e)}')
            import traceback
            traceback.print_exc()
            # 返回空字典（注意：前端期望 plot_paths，不是 comparison_paths）
            return {
                'plot_paths': [],  # ✅ 前端期望的字段名
                'animation_paths': [],
                'training_curve': '',
                'checkpoint_path': checkpoint_path
            }
    
    def get_training_summary(self):
        """获取mTSP训练总结"""
        summary = super().get_training_summary()
        summary.update({
            'num_agents': self.num_agents,
            'cost_type': self.cost_type,
        })
        return summary


def train_mtsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    mTSP训练入口函数
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    trainer = MTSPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()


__all__ = ['MTSPTrainer', 'train_mtsp']
