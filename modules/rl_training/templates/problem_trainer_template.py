"""
[PROBLEM_NAME]问题专用训练器模板
请将 [PROBLEM_NAME] 替换为具体问题名称，如 OP, PCTSP 等

使用步骤：
1. 复制此模板文件
2. 重命名为 {problem}_trainer.py，如 op_trainer.py
3. 替换所有 [PROBLEM_NAME] 标记
4. 实现必要的方法
5. 在 __init__.py 中注册新训练器
"""

import os
import torch
from datetime import datetime

try:
    from rl4co.envs import [PROBLEM_NAME]Env  # 替换为实际环境类
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    print("警告: RL4CO 库未安装")

from .base_trainer import BaseTrainer
from .visualizations.[problem]_viz import (  # 创建对应的可视化文件
    create_[problem]_route_animation,
    create_[problem]_comparison_plot
)


class [PROBLEM_NAME]Trainer(BaseTrainer):
    """[PROBLEM_NAME]问题训练器"""
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func)
        
        # TODO: 添加问题特有的参数
        # 例如：
        # self.max_prize = float(config.get('max_prize', 1.0))
        # self.time_limit = float(config.get('time_limit', 2.0))
    
    def initialize_environment(self):
        """初始化[PROBLEM_NAME]环境"""
        # TODO: 根据实际环境调整参数
        env = [PROBLEM_NAME]Env(generator_params={
            'num_loc': self.num_loc,
            # 添加其他环境参数
        })
        return env
    
    def create_model(self, env, policy):
        """创建适用于[PROBLEM_NAME]的RL模型"""
        from rl4co.models import REINFORCE
        
        model = REINFORCE(
            env,
            policy,
            baseline="rollout",
            batch_size=self.batch_size,
            train_data_size=10_000,
            val_data_size=1_000,
            optimizer_kwargs={"lr": self.learning_rate},
        )
        
        self.send_message('info', f'使用[PROBLEM_NAME]训练模式（{self.num_loc}个节点）')
        
        return model
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """生成[PROBLEM_NAME]可视化结果"""
        # 训练后测试并生成可视化
        policy = model.policy.to(self.device)
        
        # 生成测试数据
        td_init = env.reset(batch_size=[3]).to(self.device)
        
        # 未训练模型测试
        out_untrained = policy(td_init.clone(), phase="test", decode_type="sampling", return_actions=True)
        actions_untrained = out_untrained['actions'].cpu().detach()
        rewards_untrained = out_untrained['reward'].cpu().detach()
        
        # 训练后模型测试
        out_trained = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
        actions_trained = out_trained['actions'].cpu().detach()
        rewards_trained = out_trained['reward'].cpu().detach()
        
        # 生成对比图和动画
        plot_paths = []
        animation_paths = []
        
        for i, td in enumerate(td_init):
            # 生成静态对比图
            plot_filename = f"[problem]_comparison_{self.session_id[:8]}_{i+1}.png"
            plot_path = os.path.join(self.user_plots_dir, plot_filename)
            
            create_[problem]_comparison_plot(
                env, td, 
                actions_untrained[i], rewards_untrained[i],
                actions_trained[i], rewards_trained[i],
                plot_path, index=i+1
            )
            
            # 保存文件记录到数据库
            if self.bg_file_manager:
                try:
                    self.bg_file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=plot_filename,
                        file_type='plot',
                        file_path=plot_path
                    )
                except Exception as e:
                    print(f"保存文件记录失败: {str(e)}")
            
            plot_paths.append(f"/static/model_plots/user_{self.user_id}/{plot_filename}")
            
            # 生成动态路线构建过程GIF
            self.send_message('info', f'正在生成[PROBLEM_NAME]动态路线图 {i+1}/{len(td_init)}...')
            
            animation_filename = f"[problem]_animation_{self.session_id[:8]}_{i+1}.gif"
            animation_path = os.path.join(self.user_plots_dir, animation_filename)
            
            create_[problem]_route_animation(
                td, 
                actions_trained[i].cpu().numpy(), 
                animation_path,
                title="[PROBLEM_NAME]训练后路线生成过程"
            )
            
            # 保存文件记录到数据库
            if self.bg_file_manager:
                try:
                    self.bg_file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=animation_filename,
                        file_type='animation',
                        file_path=animation_path
                    )
                except Exception as e:
                    print(f"保存文件记录失败: {str(e)}")
            
            animation_paths.append(f"/static/model_plots/user_{self.user_id}/{animation_filename}")
        
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
                print(f"保存checkpoint记录失败: {str(e)}")
        
        self.send_message('info', f'检查点已保存: {checkpoint_path}')
        
        return {
            'plot_paths': plot_paths,
            'animation_paths': animation_paths,
            'training_curve': self.training_status[self.session_id].get('plot_url', ''),
            'checkpoint_path': checkpoint_path
        }


def train_[problem](config, session_id, user_id, queue, training_status, get_background_db_func):
    """
    [PROBLEM_NAME]训练的入口函数
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    trainer = [PROBLEM_NAME]Trainer(config, session_id, user_id, queue, training_status, get_background_db_func)
    trainer.train()



