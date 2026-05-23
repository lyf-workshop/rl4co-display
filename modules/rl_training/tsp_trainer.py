"""
TSP问题专用训练器
包含TSP特有的训练逻辑和可视化生成
"""

import os
import json
import logging
import torch
from datetime import datetime

logger = logging.getLogger('rl4co_display')

try:
    from rl4co.envs import TSPEnv
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    logger.warning("RL4CO 库未安装")

from .base_trainer import BaseTrainer
from .visualizations.tsp_viz import create_tsp_route_animation, create_tsp_comparison_plot


class TSPTrainer(BaseTrainer):
    """TSP问题训练器"""
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
        
        # 处理自定义数据集
        self.custom_dataset = None
        self.load_custom_dataset()
    
    def load_custom_dataset(self):
        """加载用户上传的TSP数据集"""
        dataset_mode = self.config.get('dataset_mode', 'random')
        dataset_id = self.config.get('dataset_id', None)
        
        if dataset_mode == 'upload' and dataset_id:
            dataset_path = os.path.join('datasets', f'user_{self.user_id}', f'{dataset_id}.json')
            if os.path.exists(dataset_path):
                try:
                    with open(dataset_path, 'r') as f:
                        dataset_data = json.load(f)
                        self.custom_dataset = dataset_data['coordinates']
                        self.num_loc = len(self.custom_dataset)
                        self.send_message('info', f'✅ 已加载自定义TSP数据集: {dataset_data["filename"]} ({self.num_loc}个城市)')
                except Exception as e:
                    self.send_message('info', f'⚠️ 加载数据集失败: {str(e)}，将使用随机生成')
                    self.custom_dataset = None
            else:
                self.send_message('info', '⚠️ 数据集文件不存在，将使用随机生成')
        elif dataset_mode == 'upload':
            self.send_message('info', '⚠️ 未找到数据集ID，将使用随机生成')
    
    def initialize_environment(self):
        """初始化TSP环境"""
        env = TSPEnv(generator_params={'num_loc': self.num_loc})
        return env
    
    def create_model(self, env, policy):
        """创建适用于TSP的RL模型（委托给BaseTrainer使用算法注册表）"""
        if self.custom_dataset is not None:
            n = len(self.custom_dataset)
            # 根据数据集规模调整训练数据量，覆盖 config 供算法注册表读取
            self.config['batch_size'] = min(self.batch_size, n)
            self.config['train_data_size'] = min(10_000, n * 20)
            self.config['val_data_size'] = min(1_000, n * 2)
            self.send_message('info', f'使用自定义TSP数据集训练模式（{n}个城市）')
        else:
            self.send_message('info', f'使用随机生成TSP数据集训练模式（{self.num_loc}个城市）')
        return super().create_model(env, policy)
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """生成TSP可视化结果"""
        # 训练后测试并生成可视化
        policy = model.policy.to(self.device)
        
        # 如果使用了自定义数据集，在该数据集上测试
        if self.custom_dataset is not None:
            td_init = env.reset(batch_size=[1]).to(self.device)
            coords_tensor = torch.tensor([self.custom_dataset], dtype=torch.float32).to(self.device)
            td_init['locs'] = coords_tensor
            self.send_message('info', f'✅ 在上传的TSP数据集上进行测试（{len(self.custom_dataset)}个城市）')
        else:
            td_init = env.reset(batch_size=[3]).to(self.device)
        
        # 未训练模型测试（使用随机策略）
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
            plot_filename = f"tsp_comparison_{self.session_id[:8]}_{i+1}.png"
            plot_path = os.path.join(self.user_plots_dir, plot_filename)
            
            create_tsp_comparison_plot(
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
                    logger.warning(f"保存文件记录失败: {str(e)}")
            
            plot_paths.append(f"/static/model_plots/user_{self.user_id}/{plot_filename}")
            
            # 生成动态路线构建过程GIF
            self.send_message('info', f'正在生成TSP动态路线图 {i+1}/{len(td_init)}...')
            
            animation_filename = f"tsp_animation_{self.session_id[:8]}_{i+1}.gif"
            animation_path = os.path.join(self.user_plots_dir, animation_filename)
            
            create_tsp_route_animation(
                td, 
                actions_trained[i].cpu().numpy(), 
                animation_path,
                title="TSP训练后路线生成过程"
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
                    logger.warning(f"保存文件记录失败: {str(e)}")
            
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
                logger.warning(f"保存checkpoint记录失败: {str(e)}")
        
        self.send_message('info', f'检查点已保存: {checkpoint_path}')
        
        return {
            'plot_paths': plot_paths,
            'animation_paths': animation_paths,
            'training_curve': self.training_status[self.session_id].get('plot_url', ''),
            'checkpoint_path': checkpoint_path
        }


def train_tsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    TSP训练的入口函数
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    trainer = TSPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()



