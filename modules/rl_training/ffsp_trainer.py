"""
FFSP问题专用训练器
包含FFSP特有的训练逻辑和可视化生成
"""

import os
import json
import logging
import torch
from datetime import datetime

logger = logging.getLogger('rl4co_display')

try:
    from rl4co.envs.scheduling import FFSPEnv
    from rl4co.models.zoo.matnet import MatNet
    from tensordict import TensorDict
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    TensorDict = None
    logger.warning("RL4CO 库未安装或版本过旧")

from .base_trainer import BaseTrainer
from .visualizations.ffsp_viz import (
    create_ffsp_gantt_chart,
    create_ffsp_schedule_comparison
)


# ============================================================================
# 全局修补：为 RL4CO 的 IndexTables 类添加缺失的 augment_machine_tables 方法
# 这个方法在代码中被调用但未实现，是 RL4CO 库的 bug
# ============================================================================
def _patch_index_tables():
    """全局修补 IndexTables 类，添加缺失的 augment_machine_tables 方法"""
    try:
        from rl4co.envs.scheduling.ffsp.env import IndexTables
        
        # 检查方法是否已存在
        if hasattr(IndexTables, 'augment_machine_tables'):
            return  # 已经有了，无需修补
        
        # 添加缺失的方法
        def augment_machine_tables(self, num_starts):
            """为 POMO 多起点扩展 batch size
            
            Args:
                num_starts: POMO 起点数量（通常是 num_machine 的阶乘）
            """
            if hasattr(self, 'bs') and self.bs is not None:
                self.bs = self.bs * num_starts
            else:
                self.set_bs(num_starts)
        
        IndexTables.augment_machine_tables = augment_machine_tables
        logger.info("已修补 IndexTables.augment_machine_tables 方法")

    except Exception as e:
        logger.warning(f"无法修补 IndexTables: {e}")

# 在模块导入时立即执行修补
_patch_index_tables()
# ============================================================================


class FFSPEnvWithCostMatrix:
    """
    FFSP环境适配器，为MatNet添加cost_matrix支持
    
    MatNet的初始化嵌入需要cost_matrix键，但FFSPEnv只提供run_time。
    此适配器在reset时自动添加cost_matrix = run_time。
    
    注意：仅显式代理RL4CO/MatNet需要的属性和方法，避免 __getattr__ 导致
    Lightning 序列化或深度复制时出现 maximum recursion depth exceeded。
    """
    
    def __init__(self, base_env):
        """
        参数:
            base_env: 原始的FFSPEnv实例
        """
        object.__setattr__(self, '_base_env', base_env)
        # 仅显式代理 RL4CO/MatNet 会用到的属性，避免递归
        object.__setattr__(self, 'name', base_env.name)
        object.__setattr__(self, 'num_stage', base_env.num_stage)
        object.__setattr__(self, 'num_machine', base_env.num_machine)
        object.__setattr__(self, 'num_job', base_env.num_job)
        object.__setattr__(self, 'num_machine_total', base_env.num_machine_total)
        object.__setattr__(self, 'flatten_stages', base_env.flatten_stages)
        object.__setattr__(self, 'generator', base_env.generator)
        object.__setattr__(self, 'device', getattr(base_env, 'device', 'cpu'))
        object.__setattr__(self, 'observation_spec', getattr(base_env, 'observation_spec', None))
        object.__setattr__(self, 'action_spec', getattr(base_env, 'action_spec', None))
        object.__setattr__(self, 'reward_spec', getattr(base_env, 'reward_spec', None))
        object.__setattr__(self, 'done_spec', getattr(base_env, 'done_spec', None))
        
        # 添加 Lightning/RL4CO 训练需要的属性
        object.__setattr__(self, 'dataset_cls', getattr(base_env, 'dataset_cls', None))
        object.__setattr__(self, 'data_dir', getattr(base_env, 'data_dir', 'data/'))
        object.__setattr__(self, 'train_file', getattr(base_env, 'train_file', None))
        object.__setattr__(self, 'val_file', getattr(base_env, 'val_file', None))
        object.__setattr__(self, 'test_file', getattr(base_env, 'test_file', None))
    
    @property
    def base_env(self):
        """获取底层环境"""
        return object.__getattribute__(self, '_base_env')
    
    @property
    def tables(self):
        """动态获取 tables 属性（在 reset() 后才会初始化）
        注意：IndexTables 已在模块级被全局修补，添加了 augment_machine_tables 方法
        """
        return getattr(object.__getattribute__(self, '_base_env'), 'tables', None)
    
    @property
    def step_cnt(self):
        """动态获取 step_cnt 属性"""
        return getattr(object.__getattribute__(self, '_base_env'), 'step_cnt', None)
    
    def reset(self, *args, **kwargs):
        """重写reset方法，添加cost_matrix"""
        base = object.__getattribute__(self, '_base_env')
        td = base.reset(*args, **kwargs)
        
        if 'run_time' in td.keys() and 'cost_matrix' not in td.keys():
            td['cost_matrix'] = td['run_time']
        
        return td
    
    def step(self, *args, **kwargs):
        """代理step方法"""
        return object.__getattribute__(self, '_base_env').step(*args, **kwargs)
    
    def get_reward(self, *args, **kwargs):
        """代理get_reward方法"""
        return object.__getattribute__(self, '_base_env').get_reward(*args, **kwargs)
    
    def get_num_starts(self, td):
        """POMO/MatNet 多起点需要"""
        return object.__getattribute__(self, '_base_env').get_num_starts(td)
    
    def select_start_nodes(self, td, num_starts):
        """POMO/MatNet 多起点需要"""
        return object.__getattribute__(self, '_base_env').select_start_nodes(td, num_starts)
    
    def pre_step(self, td):
        """MultiStageFFSPPolicy 可能需要"""
        base = object.__getattribute__(self, '_base_env')
        if hasattr(base, 'pre_step'):
            return base.pre_step(td)
        return td
    
    def dataset(self, batch_size=[], phase="train", filename=None):
        """代理 dataset 方法 - Lightning/RL4CO 训练时需要"""
        return object.__getattribute__(self, '_base_env').dataset(batch_size, phase, filename)
    
    def load_data(self, *args, **kwargs):
        """代理 load_data 方法"""
        base = object.__getattribute__(self, '_base_env')
        if hasattr(base, 'load_data'):
            return base.load_data(*args, **kwargs)
        raise NotImplementedError("Base environment does not have load_data method")
    
    def to(self, device):
        """代理 to 方法 - 设备转换"""
        base = object.__getattribute__(self, '_base_env')
        base.to(device)
        object.__setattr__(self, 'device', device)
        return self


class FFSPTrainer(BaseTrainer):
    """FFSP问题训练器"""
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
        
        # FFSP特有参数
        self.num_stage = int(config.get('num_stage', 3))
        self.num_machine = int(config.get('num_machine', 4))
        self.num_job = int(config.get('num_job', 20))
        self.min_time = int(config.get('min_time', 2))
        self.max_time = int(config.get('max_time', 10))
        self.flatten_stages = config.get('flatten_stages', True)
        
        # 覆盖num_loc（FFSP使用num_job）
        self.num_loc = self.num_job
        
        # 初始化策略参数（从config中读取，用于create_policy）
        self.embed_dim = int(config.get('embed_dim', 256))  # MatNet推荐256
        self.num_encoder_layers = int(config.get('num_encoder_layers', 5))  # MatNet推荐5
        self.num_heads = int(config.get('num_heads', 16))  # MatNet推荐16
        
        self.send_message('info', 
            f'FFSP配置: {self.num_stage}阶段 × {self.num_machine}机器/阶段 | '
            f'{self.num_job}个作业 | flatten_stages={self.flatten_stages}'
        )
    
    def initialize_environment(self):
        """初始化FFSP环境"""
        base_env = FFSPEnv(generator_params={
            'num_stage': self.num_stage,
            'num_machine': self.num_machine,
            'num_job': self.num_job,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'flatten_stages': self.flatten_stages,
        })
        
        # 包装环境，添加 cost_matrix 适配器
        # MatNet 的 init_embedding 需要 cost_matrix，而 FFSP 提供 run_time
        env = FFSPEnvWithCostMatrix(base_env)
        
        self.send_message('info', 
            f'✅ FFSP环境初始化完成: '
            f'总机器数={self.num_stage * self.num_machine}, '
            f'总操作数={self.num_job * self.num_stage}'
        )
        
        return env
    
    def create_policy(self, env):
        """创建适用于FFSP的策略（MatNet）"""
        # FFSP必须使用MatNet策略
        policy_type = self.config.get('model', 'matnet').lower()
        
        if policy_type not in ['matnet']:
            self.send_message('info', 
                f'⚠️ FFSP问题不支持{policy_type}策略，自动切换到MatNet'
            )
            policy_type = 'matnet'
        
        # 兼容性补丁：部分 RL4CO 版本中 MatNetPolicy 传 out_bias，而 MatNetFFSPDecoder 接受 out_bias_pointer_attn
        try:
            from rl4co.models.zoo.matnet.decoder import MatNetFFSPDecoder
            _orig_ffsp_decoder_init = MatNetFFSPDecoder.__init__
            def _patched_ffsp_decoder_init(self, *args, **kwargs):
                if 'out_bias' in kwargs and 'out_bias_pointer_attn' not in kwargs:
                    kwargs['out_bias_pointer_attn'] = kwargs.pop('out_bias')
                _orig_ffsp_decoder_init(self, *args, **kwargs)
            MatNetFFSPDecoder.__init__ = _patched_ffsp_decoder_init
        except Exception:
            pass  # 若导入/补丁失败则继续，后续可能由本地 rl4co-main 修复
        
        # 导入MatNetPolicy
        try:
            from rl4co.models.zoo.matnet import MatNetPolicy
            from rl4co.models.zoo.matnet.policy import MultiStageFFSPPolicy
        except ImportError:
            raise ImportError(
                "RL4CO库未安装MatNet模块，无法训练FFSP。\n"
                "请更新: pip install rl4co --upgrade"
            )
        
        # 根据flatten_stages选择策略
        if self.flatten_stages:
            # 使用标准MatNet策略
            policy = MatNetPolicy(
                env_name='ffsp',
                embed_dim=self.embed_dim,
                num_encoder_layers=self.num_encoder_layers,
                num_heads=self.num_heads,
                normalization='instance',
                use_graph_context=False,
            )
            self.send_message('info', '✅ 使用MatNet策略（flatten模式）')
        else:
            # 使用多阶段策略（每个阶段独立的encoder/decoder）
            policy = MultiStageFFSPPolicy(
                stage_cnt=self.num_stage,
                embed_dim=self.embed_dim,
                num_encoder_layers=self.num_encoder_layers,
                num_heads=self.num_heads,
                normalization='instance',
                use_graph_context=False,
            )
            self.send_message('info', '✅ 使用多阶段FFSP策略（独立encoder/decoder）')
        
        return policy
    
    def create_model(self, env, policy):
        """创建适用于FFSP的RL模型（MatNet）"""
        # 获取算法类型
        algorithm_type = self.config.get('algorithm', 'reinforce').lower()
        
        # FFSP推荐使用PPO或A2C
        if algorithm_type == 'reinforce':
            self.send_message('info', 
                '💡 提示: FFSP是复杂调度问题，建议使用PPO或A2C算法以获得更好的收敛性'
            )
        
        # MatNet继承自POMO，支持多起点优化
        # 计算合理的起点数（不超过机器排列数）
        import math
        max_starts = min(math.factorial(self.num_machine), 100)  # 最多100个起点
        num_starts = min(int(self.config.get('num_starts', 20)), max_starts)
        
        try:
            model = MatNet(
                env,
                policy=policy,
                num_starts=num_starts,
                batch_size=self.batch_size,
                train_data_size=5_000,  # FFSP训练数据量适当减少（计算复杂）
                val_data_size=500,
                optimizer_kwargs={"lr": self.learning_rate},
            )
            
            self.send_message('info', 
                f'✅ MatNet模型创建完成: {num_starts}个起点, '
                f'batch_size={self.batch_size}, lr={self.learning_rate}'
            )
            
            return model
        except Exception as e:
            self.send_message('error', f'创建MatNet模型失败: {str(e)}')
            import traceback
            self.send_message('error', f'详细错误: {traceback.format_exc()}')
            raise
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """生成FFSP可视化结果"""
        # 训练后测试并生成可视化
        policy = model.policy.to(self.device)
        
        # 生成测试数据
        # 注意：FFSP环境有实例级可变状态（step_cnt, tables），
        # 必须将env传给policy，否则policy会创建新的env（step_cnt=None导致TypeError）
        td_init = env.reset(batch_size=[3]).to(self.device)
        
        # 未训练模型测试（使用随机策略）
        self.send_message('info', '正在测试训练前的调度质量...')
        out_untrained = policy(td_init.clone(), env, phase="test", decode_type="sampling", return_actions=True)
        
        # 重新reset环境（上一次policy调用会修改env的step_cnt等状态）
        td_init = env.reset(batch_size=[3]).to(self.device)
        
        # 训练后模型测试（使用greedy策略）
        self.send_message('info', '正在测试训练后的调度质量...')
        out_trained = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)
        
        # 提取调度结果和actions
        try:
            rewards_untrained = out_untrained.get('reward', torch.zeros(3))
            rewards_trained = out_trained.get('reward', torch.zeros(3))
            actions_trained = out_trained.get('actions', None)
            
            makespan_untrained = -rewards_untrained.cpu().detach().numpy()
            makespan_trained = -rewards_trained.cpu().detach().numpy()
            
            # 检查是否成功获取actions
            if actions_trained is None:
                self.send_message('info', '⚠️ 未能从policy输出中获取actions，无法生成甘特图')
                return {
                    'plot_paths': [],
                    'animation_paths': [],
                    'training_curve': self.training_status[self.session_id].get('plot_url', ''),
                    'checkpoint_path': checkpoint_path
                }
            
        except Exception as e:
            self.send_message('info', f'⚠️ 提取调度结果时出错: {str(e)}')
            import traceback
            self.send_message('info', f'详细错误: {traceback.format_exc()}')
            return {
                'plot_paths': [],
                'animation_paths': [],
                'training_curve': self.training_status[self.session_id].get('plot_url', ''),
                'checkpoint_path': checkpoint_path
            }
        
        # 使用actions重新执行环境以获取完整的schedule
        self.send_message('info', '正在重放调度以生成可视化...')
        try:
            # 重置环境并逐步执行以获取最终状态
            td_replay = env.reset(batch_size=[3]).to(self.device)
            
            # 执行每个action
            for step_idx in range(actions_trained.shape[-1]):
                action = actions_trained[:, step_idx]
                td_replay.set('action', action)
                td_replay = env.step(td_replay)['next']
                
                # 如果所有实例都完成了，停止
                if td_replay['done'].all():
                    break
            
            # 现在td_replay包含了完整的调度信息
            self.send_message('info', f'✅ 调度重放完成，可用键: {list(td_replay.keys())}')
            
        except Exception as e:
            self.send_message('info', f'⚠️ 调度重放失败: {str(e)}')
            import traceback
            self.send_message('info', f'详细错误: {traceback.format_exc()}')
            return {
                'plot_paths': [],
                'animation_paths': [],
                'training_curve': self.training_status[self.session_id].get('plot_url', ''),
                'checkpoint_path': checkpoint_path
            }
        
        # 生成甘特图
        plot_paths = []
        
        # 遍历每个样本生成甘特图
        for i in range(min(3, td_replay.batch_size[0])):
            try:
                # 生成训练后的甘特图
                gantt_filename = f"ffsp_gantt_{self.session_id[:8]}_{i+1}.png"
                gantt_path = os.path.join(self.user_plots_dir, gantt_filename)
                
                self.send_message('info', f'正在生成FFSP甘特图 {i+1}/3...')
                
                # 检查是否有schedule键
                if 'schedule' not in td_replay.keys():
                    self.send_message('info', f'⚠️ TensorDict中没有schedule键，可用键: {list(td_replay.keys())}')
                    break
                
                # 提取单个样本的schedule和job_duration
                # schedule: [batch, num_machines, num_jobs+1]
                schedule_single = td_replay['schedule'][i].cpu()  # [num_machines, num_jobs+1]
                
                # 创建单个样本的TensorDict，包含必要的键
                from tensordict import TensorDict
                td_single_keys = {}
                for key in ['job_duration', 'run_time', 'job_location', 'machine_wait_step']:
                    if key in td_replay.keys():
                        value = td_replay[key][i:i+1] if td_replay[key].dim() > 0 else td_replay[key]
                        td_single_keys[key] = value
                
                td_single = TensorDict(td_single_keys, batch_size=[1])
                
                # 调用可视化函数生成甘特图
                from modules.rl_training.visualizations.ffsp_viz import create_ffsp_gantt_chart
                
                makespan = create_ffsp_gantt_chart(
                    td_single,
                    schedule_single,
                    gantt_path,
                    title=f"FFSP训练后调度 (实例 {i+1})"
                )
                
                self.send_message('info', f'✅ 甘特图 {i+1} 已生成: makespan={makespan:.2f}')
                
                # 保存文件记录到数据库
                if self.bg_file_manager:
                    try:
                        self.bg_file_manager.save_file_record(
                            user_id=self.user_id,
                            session_id=self.session_id,
                            filename=gantt_filename,
                            file_type='plot',
                            file_path=gantt_path
                        )
                    except Exception as e:
                        logger.warning(f"保存文件记录失败: {str(e)}")
                
                plot_paths.append(f"/static/model_plots/user_{self.user_id}/{gantt_filename}")
                
            except Exception as e:
                self.send_message('info', f'⚠️ 生成第{i+1}个甘特图时出错: {str(e)}')
                import traceback
                self.send_message('info', f'详细错误: {traceback.format_exc()}')
        
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
        
        # 计算改进
        avg_makespan_before = makespan_untrained.mean()
        avg_makespan_after = makespan_trained.mean()
        improvement = ((avg_makespan_before - avg_makespan_after) / avg_makespan_before) * 100
        
        self.send_message('info', 
            f'📊 调度结果: 训练前makespan={avg_makespan_before:.2f}, '
            f'训练后={avg_makespan_after:.2f}, 改进={improvement:.2f}%'
        )
        
        return {
            'plot_paths': plot_paths,
            'animation_paths': [],  # FFSP不生成动画
            'training_curve': self.training_status[self.session_id].get('plot_url', ''),
            'checkpoint_path': checkpoint_path,
            'metrics': {
                'makespan_before': float(avg_makespan_before),
                'makespan_after': float(avg_makespan_after),
                'improvement_percent': float(improvement),
            }
        }
    
    def get_problem_specific_info(self):
        """获取FFSP特有的问题信息"""
        return {
            'problem_type': 'ffsp',
            'num_stage': self.num_stage,
            'num_machine': self.num_machine,
            'num_job': self.num_job,
            'num_machine_total': self.num_stage * self.num_machine,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'flatten_stages': self.flatten_stages,
            'total_operations': self.num_job * self.num_stage,
        }


def train_ffsp(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
    """
    FFSP训练入口函数
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    trainer = FFSPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func, pause_event)
    trainer.train()
