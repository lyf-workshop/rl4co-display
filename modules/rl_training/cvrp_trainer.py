"""
CVRP问题专用训练器
包含CVRP特有的训练逻辑和可视化生成
"""

import os
import logging
import torch
from datetime import datetime

logger = logging.getLogger('rl4co_display')

try:
    from rl4co.envs import CVRPEnv
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    logger.warning("RL4CO 库未安装")

from .base_trainer import BaseTrainer
from .visualizations.cvrp_viz import create_cvrp_route_animation, create_cvrp_comparison_plot


class CVRPTrainer(BaseTrainer):
    """CVRP问题训练器"""
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
        
        # CVRP特有的参数
        self.vehicle_capacity = float(config.get('vehicle_capacity', 1.0))
        self.num_vehicles = int(config.get('num_vehicles', 1))
        self.load_custom_dataset()

    def _inject_custom_data(self, td):
        """将自定义数据集注入 TensorDict。"""
        data = self.custom_dataset_data
        coords = torch.tensor(data['coordinates'], dtype=torch.float32)  # [N, 2]

        if data.get('depot'):
            depot = torch.tensor(data['depot'], dtype=torch.float32)
        else:
            depot = td['locs'][0, 0].cpu()

        locs = torch.cat([depot.unsqueeze(0), coords], dim=0)  # [N+1, 2]
        td['locs'] = locs.unsqueeze(0).to(self.device)

        if data.get('demands'):
            demand = torch.tensor(data['demands'], dtype=torch.float32)  # [N]
            td['demand'] = demand.unsqueeze(0).to(self.device)

        return td

    def initialize_environment(self):
        """初始化CVRP环境"""
        env = CVRPEnv(generator_params={
            'num_loc': self.num_loc,
            'vehicle_capacity': self.vehicle_capacity
        })
        self.send_message('info', f'CVRP训练模式（{self.num_loc}个客户，容量={self.vehicle_capacity}）')
        return env
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """生成CVRP可视化结果"""
        # 训练后测试并生成可视化
        policy = model.policy.to(self.device)
        
        # 生成测试数据
        if self.custom_dataset_data:
            td_init = env.reset(batch_size=[1]).to(self.device)
            td_init = self._inject_custom_data(td_init)
            self.send_message('info', f'✅ 在上传的CVRP数据集上进行测试（{self.num_loc}个客户）')
        else:
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
            plot_filename = f"cvrp_comparison_{self.session_id[:8]}_{i+1}.png"
            plot_path = os.path.join(self.user_plots_dir, plot_filename)
            
            create_cvrp_comparison_plot(
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
            self.send_message('info', f'正在生成CVRP动态路线图 {i+1}/{len(td_init)}...')
            
            animation_filename = f"cvrp_animation_{self.session_id[:8]}_{i+1}.gif"
            animation_path = os.path.join(self.user_plots_dir, animation_filename)
            
            create_cvrp_route_animation(
                td, 
                actions_trained[i].cpu().numpy(), 
                animation_path,
                title="CVRP训练后路线生成过程"
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


def train_cvrp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    CVRP训练的入口函数
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    trainer = CVRPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()



