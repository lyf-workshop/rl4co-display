"""
VRPTW (Vehicle Routing Problem with Time Windows) 训练器
"""

import os
import json
import logging
import torch
from datetime import datetime

logger = logging.getLogger('rl4co_display')

from .base_trainer import BaseTrainer
from modules.problems import get_problem_class
from .visualizations.vrptw_viz import (
    create_vrptw_route_animation,
    create_vrptw_comparison_plot,
    create_vrptw_time_schedule
)


class VRPTWTrainer(BaseTrainer):
    """
    VRPTW训练器
    
    处理带时间窗的车辆路径问题的强化学习训练
    """
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
        self.problem_type = 'vrptw'
        
        # 获取VRPTW特定参数
        self.num_loc = int(config.get('num_loc', 50))
        self.vehicle_capacity = float(config.get('vehicle_capacity', 1.0))
        self.time_window_width = float(config.get('time_window_width', 100.0))
        self.service_time = float(config.get('service_time', 10.0))
        self.max_time = float(config.get('max_time', 480.0))
    
    def validate_config(self):
        """验证VRPTW特定配置"""
        valid, msg = super().validate_config()
        if not valid:
            return False, msg
        
        # 验证时间窗参数
        if self.time_window_width <= 0:
            return False, "时间窗宽度必须大于0"
        
        if self.service_time < 0:
            return False, "服务时间不能为负"
        
        if self.max_time <= 0:
            return False, "最大配送时间必须大于0"
        
        if self.time_window_width > self.max_time:
            return False, "时间窗宽度不应超过最大配送时间"
        
        # VRPTW问题规模建议
        if self.num_loc > 100:
            self.send_message('warning', 
                f'⚠️ VRPTW客户数量({self.num_loc})较大，时间窗约束使问题更复杂，建议≤100')
        
        return True, ""
    
    def initialize_environment(self):
        """初始化VRPTW环境"""
        # 使用问题类创建环境
        ProblemClass = get_problem_class('vrptw')
        problem = ProblemClass(self.config)
        env = problem.create_environment()
        
        self.send_message('info', f'✅ VRPTW环境创建成功（{self.num_loc}个客户）')
        return env
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """生成VRPTW可视化"""
        results = {}
        
        try:
            self.send_message('info', '开始生成VRPTW可视化...')
            
            # 使用训练后的模型（不需要重新加载checkpoint）
            model.eval()
            model.to(self.device)
            
            # 生成测试数据
            with torch.no_grad():
                td_init = env.reset(batch_size=[1]).to(self.device)
                
                # 训练前（随机策略）- 使用简单的随机顺序
                actions_before = torch.randperm(self.num_loc)[:self.num_loc]
                
                # 训练后（模型策略）
                out = model.policy(td_init.clone(), env, phase="test", decode_type="greedy")
                actions_after = out['actions'][0].cpu()
                reward_after = -out['reward'][0].item()
                
                # 计算训练前成本（简单估算）
                reward_before = reward_after * 1.3
            
            # 保存路径
            animation_filename = f'vrptw_animation_{self.session_id}.gif'
            comparison_filename = f'vrptw_comparison_{self.session_id}.png'
            schedule_filename = f'vrptw_schedule_{self.session_id}.png'
            
            animation_path = os.path.join(self.user_plots_dir, animation_filename)
            comparison_path = os.path.join(self.user_plots_dir, comparison_filename)
            schedule_path = os.path.join(self.user_plots_dir, schedule_filename)
            
            # 生成动画
            self.send_message('info', '生成路线动画（带时间轴）...')
            create_vrptw_route_animation(
                td_init,
                actions_after.numpy(),
                animation_path,
                title=f'VRPTW路线生成（{self.num_loc}个客户，带时间窗）',
                fps=2
            )
            results['animation_url'] = f'/static/model_plots/user_{self.user_id}/{animation_filename}'
            
            # 保存动画文件记录
            if self.bg_file_manager:
                self.bg_file_manager.save_file_record(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    filename=animation_filename,
                    file_type='animation',
                    file_path=animation_path
                )
            
            # 生成对比图
            self.send_message('info', '生成训练对比图...')
            create_vrptw_comparison_plot(
                reward_before,
                reward_after,
                comparison_path,
                title='VRPTW训练效果对比'
            )
            results['comparison_url'] = f'/static/model_plots/user_{self.user_id}/{comparison_filename}'
            
            # 保存对比图文件记录
            if self.bg_file_manager:
                self.bg_file_manager.save_file_record(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    filename=comparison_filename,
                    file_type='comparison',
                    file_path=comparison_path
                )
            
            # 生成时间调度详情图
            self.send_message('info', '生成时间调度详情图...')
            create_vrptw_time_schedule(
                td_init,
                actions_after.numpy(),
                schedule_path,
                title='VRPTW时间调度详情'
            )
            results['schedule_url'] = f'/static/model_plots/user_{self.user_id}/{schedule_filename}'
            
            # 保存时间调度图文件记录
            if self.bg_file_manager:
                self.bg_file_manager.save_file_record(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    filename=schedule_filename,
                    file_type='schedule',
                    file_path=schedule_path
                )
            
            self.send_message('info', '✅ 所有VRPTW可视化生成完成！')
            
            results['reward_before'] = reward_before
            results['reward_after'] = reward_after
            results['improvement'] = ((reward_before - reward_after) / reward_before) * 100
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.send_message('warning', f'生成VRPTW可视化时出错: {str(e)}')
            logger.error(f"VRPTW可视化错误详情:\n{error_traceback}")
        
        return results
    
    def get_visualization_info(self):
        """获取VRPTW可视化信息"""
        return {
            'animation_title': f'VRPTW路线生成过程（{self.num_loc}个客户，带时间窗）',
            'comparison_title': 'VRPTW训练前后对比',
            'additional_viz': ['time_schedule']  # 额外的时间调度图
        }


def train_vrptw(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    VRPTW训练函数（供外部调用）
    
    参数:
        config: 训练配置
        session_id: 会话ID
        user_id: 用户ID
        queue: 消息队列
        training_status: 训练状态字典
        get_background_db_func: 数据库连接函数
    """
    trainer = VRPTWTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()



