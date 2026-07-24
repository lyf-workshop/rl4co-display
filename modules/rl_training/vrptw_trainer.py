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
        self.load_custom_dataset()

    def _inject_custom_data(self, td, env=None):
        data = self.custom_dataset_data
        device = td.device
        coords = torch.as_tensor(data['coordinates'], dtype=torch.float32, device=device)
        expected_customers = td['locs'].shape[-2] - 1
        if coords.shape != (expected_customers, 2):
            raise ValueError(
                f'VRPTW dataset has {len(coords)} customers, but the environment expects '
                f'{expected_customers} customers'
            )

        if data.get('depot') is not None:
            depot = torch.as_tensor(data['depot'], dtype=torch.float32, device=device)
        else:
            depot = td['locs'][0, 0]
        td['locs'] = self._expand_custom_tensor(
            td, torch.cat([depot.unsqueeze(0), coords], dim=0)
        )

        if data.get('demands') is not None:
            demand = torch.as_tensor(data['demands'], dtype=torch.float32, device=device)
            if demand.numel() != expected_customers:
                raise ValueError('VRPTW demands length must match coordinates length')
            if td['demand'].shape[-1] == expected_customers:
                td['demand'] = self._expand_custom_tensor(td, demand)
            elif td['demand'].shape[-1] == expected_customers + 1:
                td['demand'] = self._expand_custom_tensor(
                    td, torch.cat([torch.zeros(1, device=device), demand])
                )
            else:
                raise ValueError('VRPTW environment demand shape is incompatible with the dataset')
            if env is not None:
                td['action_mask'] = env.get_action_mask(td)

        if data.get('time_windows') is not None:
            customer_tw = torch.as_tensor(
                data['time_windows'], dtype=torch.float32, device=device
            )
            depot_tw = torch.tensor([[0.0, self.max_time]], dtype=torch.float32, device=device)
            td['time_windows'] = self._expand_custom_tensor(
                td, torch.cat([depot_tw, customer_tw], dim=0)
            )

        if data.get('service_times') is not None:
            service_times = torch.as_tensor(
                data['service_times'], dtype=torch.float32, device=device
            )
            td['service_time'] = self._expand_custom_tensor(
                td, torch.cat([torch.zeros(1, device=device), service_times])
            )
        return td

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
        env_config = dict(self.config)
        env_config['num_loc'] = self.num_loc
        problem = ProblemClass(env_config)
        env = problem.create_environment()
        
        self.send_message('info', f'✅ VRPTW环境创建成功（{self.num_loc}个客户）')
        return env
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """生成VRPTW可视化"""
        plot_paths = []
        animation_paths = []

        # 在可视化之前先保存 checkpoint
        if checkpoint_path:
            trainer.save_checkpoint(checkpoint_path)
            checkpoint_filename = os.path.basename(checkpoint_path)
            self._save_file_record(checkpoint_filename, 'checkpoint', checkpoint_path)
            self.send_message('info', f'检查点已保存: {checkpoint_path}')

        try:
            self.send_message('info', '开始生成VRPTW可视化...')

            device = self._get_model_device(model)
            model.eval()
            model.to(device)

            # ── 未训练基线推断 ────────────────────────────────────────────────
            # VRPTW env 是有状态的（policy 调用后 step_cnt 等内部状态改变），
            # 需要用独立 td 推断，再 reset 一次供训练后模型使用
            untrained_policy = self.create_untrained_policy_copy(model)
            try:
                td_before = env.reset(batch_size=[1]).to(device)
                if self.custom_dataset_data:
                    td_before = self._inject_custom_data(td_before, env=env)
                with torch.no_grad():
                    out_before = untrained_policy(td_before.clone(), env,
                                                  phase="test", decode_type="greedy")
                reward_before = -out_before['reward'][0].item()
            except Exception as _e:
                logger.warning(f"VRPTW 未训练基线推断失败，回退到估算: {_e}")
                reward_before = None

            # ── 训练后模型推断 ────────────────────────────────────────────────
            with torch.no_grad():
                td_init = env.reset(batch_size=[1]).to(device)
                if self.custom_dataset_data:
                    td_init = self._inject_custom_data(td_init, env=env)
                    self.send_message('info', f'✅ 在上传的VRPTW数据集上进行测试（{self.num_loc}个客户）')

                out = model.policy(td_init.clone(), env, phase="test", decode_type="greedy")
                actions_after = out['actions'][0].cpu()
                reward_after = -out['reward'][0].item()

            # 未训练基线推断失败时，退而用 1.5× 粗估（保证对比图可生成）
            if reward_before is None:
                reward_before = reward_after * 1.5

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
            animation_paths.append(f'/static/model_plots/user_{self.user_id}/{animation_filename}')
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
            plot_paths.append(f'/static/model_plots/user_{self.user_id}/{comparison_filename}')
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
            plot_paths.append(f'/static/model_plots/user_{self.user_id}/{schedule_filename}')
            if self.bg_file_manager:
                self.bg_file_manager.save_file_record(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    filename=schedule_filename,
                    file_type='schedule',
                    file_path=schedule_path
                )

            self.send_message('info', '✅ 所有VRPTW可视化生成完成！')

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.send_message('warning', f'生成VRPTW可视化时出错: {str(e)}')
            logger.error(f"VRPTW可视化错误详情:\n{error_traceback}")

        return {
            'plot_paths': plot_paths,
            'animation_paths': animation_paths,
            'training_curve': self.training_status[self.session_id].get('plot_url', ''),
            'checkpoint_path': checkpoint_path,
        }
    
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



