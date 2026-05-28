"""
SDVRP问题专用训练器
包含SDVRP特有的训练逻辑和可视化生成
"""

import os
import json
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
from .visualizations.sdvrp_viz import (
    create_sdvrp_route_animation,
    create_sdvrp_comparison_plot,
    create_sdvrp_split_analysis
)


class SDVRPTrainer(BaseTrainer):
    """SDVRP问题训练器（允许分割配送）"""
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
        
        # SDVRP特有的参数
        self.vehicle_capacity = float(config.get('vehicle_capacity', 1.0))
        self.num_vehicles = int(config.get('num_vehicles', 1))
        self.max_split_per_customer = int(config.get('max_split_per_customer', 3))
        self.allow_split = True
        self.load_custom_dataset()

    def _inject_custom_data(self, td, device):
        data = self.custom_dataset_data
        coords = torch.tensor(data['coordinates'], dtype=torch.float32)

        if data.get('depot'):
            depot = torch.tensor(data['depot'], dtype=torch.float32)
        else:
            depot = td['locs'][0, 0].cpu()

        locs = torch.cat([depot.unsqueeze(0), coords], dim=0)
        td['locs'] = locs.unsqueeze(0).to(device)

        if data.get('demands'):
            demand = torch.tensor(data['demands'], dtype=torch.float32)
            td['demand'] = demand.unsqueeze(0).to(device)

        return td

    def initialize_environment(self):
        """初始化SDVRP环境（使用CVRP环境并允许分割）"""
        try:
            # 尝试使用专用SDVRP环境
            try:
                from rl4co.envs import SDVRPEnv
                env = SDVRPEnv(generator_params={
                    'num_loc': self.num_loc,
                    'vehicle_capacity': self.vehicle_capacity,
                })
                self.send_message('info', f'使用SDVRP专用环境')
            except (ImportError, AttributeError):
                # 使用CVRP环境模拟
                env = CVRPEnv(generator_params={
                    'num_loc': self.num_loc,
                    'vehicle_capacity': self.vehicle_capacity,
                })
                # 标记允许分割配送
                env.allow_split_delivery = True
                self.send_message('info', f'使用CVRP环境模拟SDVRP（允许分割配送）')
            
            self.send_message('info',
                f'SDVRP训练模式（{self.num_loc}个客户，容量={self.vehicle_capacity}，'
                f'最大分割={self.max_split_per_customer}次）')
            return env
        except Exception as e:
            self.send_message('error', f'环境初始化失败: {str(e)}')
            raise
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """生成SDVRP可视化（包含分割配送分析）"""
        try:
            policy = model.policy.to(self.device)

            # 生成测试数据
            if self.custom_dataset_data:
                td_init = env.reset(batch_size=[1]).to(self.device)
                td_init = self._inject_custom_data(td_init, self.device)
                self.send_message('info', f'✅ 在上传的SDVRP数据集上进行测试（{self.num_loc}个客户）')
            else:
                td_init = env.reset(batch_size=[3]).to(self.device)

            # 训练前基线（未训练权重 + 贪心解码）vs 训练后模型（训练权重 + 贪心解码）
            untrained_policy = self.create_untrained_policy_copy(model)
            with torch.no_grad():
                out_untrained = self._run_policy(untrained_policy, td_init.clone(), env,
                                                 phase="test", decode_type="greedy",
                                                 return_actions=True)
                out_trained = self._run_policy(policy, td_init.clone(), env,
                                               phase="test", decode_type="greedy",
                                               return_actions=True)
            actions_untrained = out_untrained['actions'].cpu().detach()
            rewards_untrained = out_untrained['reward'].cpu().detach()
            actions_trained = out_trained['actions'].cpu().detach()
            rewards_trained = out_trained['reward'].cpu().detach()
            
            plot_paths = []
            animation_paths = []
            analysis_paths = []
            
            # 为每个实例生成可视化
            for i, td in enumerate(td_init):
                # 1. 生成静态对比图
                plot_filename = f"comparison_{self.session_id[:8]}_{i+1}.png"
                plot_path = os.path.join(self.user_plots_dir, plot_filename)
                
                create_sdvrp_comparison_plot(
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
                
                # 2. 生成动态路线动画
                self.send_message('info', f'正在生成SDVRP动态路线图 {i+1}/3...')
                
                animation_filename = f"animation_{self.session_id[:8]}_{i+1}.gif"
                animation_path = os.path.join(self.user_plots_dir, animation_filename)
                
                create_sdvrp_route_animation(
                    td,
                    actions_trained[i].cpu().numpy(),
                    animation_path,
                    title="SDVRP分割配送路线生成"
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
                
                # 3. 生成分割配送分析图
                analysis_filename = f"split_analysis_{self.session_id[:8]}_{i+1}.png"
                analysis_path = os.path.join(self.user_plots_dir, analysis_filename)
                
                demands = td.get('demand', td.get('demands', None))
                if demands is not None:
                    demands = demands.cpu().numpy()
                
                create_sdvrp_split_analysis(
                    actions_trained[i].cpu().numpy(),
                    demands,
                    analysis_path,
                    title=f"SDVRP分割配送分析 - 实例{i+1}"
                )
                
                # 保存文件记录到数据库
                if self.bg_file_manager:
                    try:
                        self.bg_file_manager.save_file_record(
                            user_id=self.user_id,
                            session_id=self.session_id,
                            filename=analysis_filename,
                            file_type='analysis',
                            file_path=analysis_path
                        )
                    except Exception as e:
                        logger.warning(f"保存文件记录失败: {str(e)}")
                analysis_paths.append(f"/static/model_plots/user_{self.user_id}/{analysis_filename}")
            
            return {
                'plot_paths': plot_paths,
                'animation_paths': animation_paths,
                'analysis_paths': analysis_paths,
            }
            
        except Exception as e:
            self.send_message('warning', f'生成可视化失败: {str(e)}')
            import traceback
            traceback.print_exc()
            return {}


def train_sdvrp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    SDVRP训练入口函数
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列
        training_status: 训练状态字典
        get_background_db_func: 获取数据库连接的函数
    """
    trainer = SDVRPTrainer(
        config=config,
        session_id=session_id,
        user_id=user_id,
        queue=queue,
        training_status=training_status,
        get_background_db_func=get_background_db_func,
        pause_event=pause_event
    )
    
    trainer.train()

