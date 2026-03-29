"""
RL4CO 训练基类和通用组件
包含所有问题类型共享的训练逻辑、回调函数等
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# 导入 RL4CO 相关组件
try:
    from rl4co.utils.trainer import RL4COTrainer
    from lightning.pytorch.callbacks import Callback
    from tensordict import TensorDict
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    Callback = object  # 降级为普通对象基类
    TensorDict = None
    print("警告: RL4CO 库未安装，训练功能将不可用")

# ========== 导入算法和策略模块 ==========
try:
    from modules.algorithms import get_algorithm_class
    from modules.policies import get_policy_class
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("警告: 算法/策略模块未找到，使用传统模式")

# 导入认证模块的路径辅助函数
from auth_module import (
    get_user_plot_dir,
    get_user_checkpoint_dir,
    FileManager,
    TrainingSessionManager
)

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class ProgressCallback(Callback):
    """Lightning回调类，用于捕获训练进度并推送到消息队列"""
    
    def __init__(self, queue, session_id, total_epochs, user_id, training_status=None):
        super().__init__()
        self.queue = queue  # 与前端通信的消息队列
        self.session_id = session_id  # 当前训练会话ID
        self.total_epochs = total_epochs  # 总训练轮数
        self.user_id = user_id  # 用户ID
        self.training_status = training_status  # 全局训练状态字典引用
        self.best_reward = float('-inf')  # 记录历史最优奖励
        self.epoch_losses = []  # 当前epoch内每个batch的loss
        self.epoch_rewards = []  # 当前epoch内每个batch的reward
        # 用于存储所有epoch的历史数据，用于绘制折线图
        self.history_losses = []  # 所有epoch的平均loss历史
        self.history_rewards = []  # 所有epoch的平均reward历史
        self.history_epochs = []  # epoch编号列表
        # 为后台线程创建独立的数据库连接和管理器
        self.db = None
        self.file_manager = None
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """每个 batch 结束时收集指标"""
        # 尝试从多个源头获取 loss 和 reward
        loss_collected = False
        reward_collected = False
        
        # 方法1: 从 outputs 获取
        if outputs is not None and isinstance(outputs, dict):
            if 'loss' in outputs:
                loss_val = outputs['loss']
                if isinstance(loss_val, torch.Tensor):
                    self.epoch_losses.append(loss_val.item())
                    loss_collected = True
            
            if 'reward' in outputs:
                reward_val = outputs['reward']
                if isinstance(reward_val, torch.Tensor):
                    self.epoch_rewards.append(reward_val.mean().item())
                    reward_collected = True
        
        # 方法2: 从 pl_module 的 logged_metrics 获取
        if not loss_collected and hasattr(pl_module, 'log_dict') and hasattr(trainer, 'logged_metrics'):
            logged = trainer.logged_metrics
            if 'loss' in logged:
                loss_val = logged['loss']
                if isinstance(loss_val, torch.Tensor):
                    self.epoch_losses.append(loss_val.item())
        
        if not reward_collected and hasattr(pl_module, 'log_dict') and hasattr(trainer, 'logged_metrics'):
            logged = trainer.logged_metrics
            if 'reward' in logged:
                reward_val = logged['reward']
                if isinstance(reward_val, torch.Tensor):
                    self.epoch_rewards.append(reward_val.item())
    
    def on_train_epoch_end(self, trainer, pl_module):
        """每个训练 epoch 结束时调用"""
        epoch = trainer.current_epoch + 1
        
        # 首先尝试从累积的 batch 指标中计算平均值
        loss = 0.0
        reward = 0.0
        
        if self.epoch_losses:
            loss = sum(self.epoch_losses) / len(self.epoch_losses)
        
        if self.epoch_rewards:
            reward = sum(self.epoch_rewards) / len(self.epoch_rewards)
        
        # 如果没有从 batch 中获取到，尝试从 metrics 获取
        if loss == 0.0 or reward == 0.0:
            metrics = trainer.callback_metrics
            
            # 调试：打印所有可用的指标键名（仅第一个epoch）
            if epoch == 1:
                self.queue.put(json.dumps({
                    'type': 'info',
                    'message': f'🔍 可用的 callback_metrics 键: {list(metrics.keys())}'
                }))
                if hasattr(trainer, 'logged_metrics'):
                    self.queue.put(json.dumps({
                        'type': 'info',
                        'message': f'🔍 可用的 logged_metrics 键: {list(trainer.logged_metrics.keys())}'
                    }))
            
            # RL4CO 的 REINFORCE 模型可能使用的键名（扩展搜索）
            if loss == 0.0:
                loss = (metrics.get('loss') or 
                       metrics.get('train_loss') or 
                       metrics.get('train/loss') or
                       metrics.get('train_loss_epoch') or
                       0.0)
            if reward == 0.0:
                reward = (metrics.get('reward') or 
                         metrics.get('train_reward') or 
                         metrics.get('train/reward') or
                         metrics.get('train_reward_epoch') or
                         metrics.get('val_reward') or
                         0.0)
            
            # 如果还是没有找到，尝试从 logged_metrics 获取
            if loss == 0.0 and hasattr(trainer, 'logged_metrics'):
                logged = trainer.logged_metrics
                loss = (logged.get('loss') or 
                       logged.get('train_loss') or 
                       logged.get('train/loss') or
                       0.0)
            
            if reward == 0.0 and hasattr(trainer, 'logged_metrics'):
                logged = trainer.logged_metrics
                reward = (logged.get('reward') or 
                         logged.get('train_reward') or 
                         logged.get('train/reward') or
                         0.0)
            
            # 转换tensor为标量
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            
            # 如果仍然都是0，打印警告（仅第一次）
            if epoch == 1 and loss == 0.0 and reward == 0.0:
                self.queue.put(json.dumps({
                    'type': 'warning',
                    'message': '⚠️ 无法从trainer获取loss和reward，将使用累积的batch指标'
                }))
        
        # 清空本 epoch 的累积指标
        self.epoch_losses = []
        self.epoch_rewards = []
        
        self.best_reward = max(self.best_reward, reward)
        progress = (epoch / self.total_epochs) * 100
        
        # 记录历史数据用于绘制折线图
        self.history_epochs.append(epoch)
        self.history_losses.append(loss)
        self.history_rewards.append(reward)
        
        # 生成实时训练曲线图
        try:
            USER_PLOTS_DIR = get_user_plot_dir(self.user_id)
            plot_filename = f"training_curves_{self.session_id[:8]}.png"
            plot_path = os.path.join(USER_PLOTS_DIR, plot_filename)
            
            # 创建包含loss和reward的双子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 绘制Loss曲线
            ax1.plot(self.history_epochs, self.history_losses, 'b-o', linewidth=2, markersize=6, label='Loss')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('训练Loss变化曲线', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper right', fontsize=10)
            
            # 绘制Reward曲线
            ax2.plot(self.history_epochs, self.history_rewards, 'g-o', linewidth=2, markersize=6, label='Reward')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Reward', fontsize=12)
            ax2.set_title('训练Reward变化曲线', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='lower right', fontsize=10)
            
            # 在reward图上标注最佳reward
            best_epoch_idx = self.history_rewards.index(max(self.history_rewards))
            best_epoch_num = self.history_epochs[best_epoch_idx]
            ax2.axhline(y=self.best_reward, color='r', linestyle='--', alpha=0.5, label=f'Best: {self.best_reward:.4f}')
            ax2.scatter([best_epoch_num], [self.best_reward], color='red', s=100, zorder=5, marker='*')
            ax2.legend(loc='lower right', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            # 保存文件记录到数据库
            if self.file_manager is None:
                # 为后台线程创建独立的数据库连接
                from config.config import Config
                import mysql.connector as mysql_connector
                try:
                    self.db = mysql_connector.connect(
                        host=Config.MYSQL_HOST,
                        user=Config.MYSQL_USER,
                        password=Config.MYSQL_PASSWORD,
                        database=Config.MYSQL_DB,
                        autocommit=True
                    )
                    if self.db:
                        self.file_manager = FileManager(self.db)
                except Exception as e:
                    print(f"创建后台数据库连接失败: {str(e)}")
            
            if self.file_manager:
                try:
                    self.file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=plot_filename,
                        file_type='curve',
                        file_path=plot_path
                    )
                except Exception as e:
                    print(f"保存文件记录失败: {str(e)}")
            
            # 通过队列发送图表路径
            self.queue.put(json.dumps({
                'type': 'plot',
                'plot_url': f"/static/model_plots/user_{self.user_id}/{plot_filename}",
                'message': f'Epoch {epoch} 训练曲线已更新'
            }))
        except Exception as e:
            self.queue.put(json.dumps({
                'type': 'warning',
                'message': f'生成训练曲线失败: {str(e)}'
            }))
        
        # 同步更新全局 training_status，供训练结束时 final_results 读取
        if self.training_status is not None and self.session_id in self.training_status:
            self.training_status[self.session_id]['loss'] = round(loss, 4)
            self.training_status[self.session_id]['reward'] = round(reward, 4)
            self.training_status[self.session_id]['best_reward'] = round(self.best_reward, 4)
            self.training_status[self.session_id]['epoch'] = epoch
            self.training_status[self.session_id]['progress'] = round(progress, 2)

        # 发送进度更新
        self.queue.put(json.dumps({
            'type': 'progress',
            'epoch': epoch,
            'total_epochs': self.total_epochs,
            'progress': round(progress, 2),
            'loss': round(loss, 4),
            'reward': round(reward, 4),
            'best_reward': round(self.best_reward, 4),
            'plot_url': f"/static/model_plots/user_{self.user_id}/training_curves_{self.session_id[:8]}.png"
        }))
        
        # 发送详细信息
        self.queue.put(json.dumps({
            'type': 'info',
            'message': f'Epoch {epoch}/{self.total_epochs} - Loss: {loss:.4f}, Reward: {reward:.4f}, Best: {self.best_reward:.4f}'
        }))
    
    def __del__(self):
        """析构时关闭数据库连接"""
        if self.db:
            try:
                self.db.close()
            except:
                pass


class BaseTrainer:
    """强化学习训练器基类，提供通用训练逻辑"""
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func):
        """
        初始化训练器
        
        参数:
            config: 训练配置字典
            session_id: 训练会话ID
            user_id: 用户ID
            queue: 消息队列（用于推送进度）
            training_status: 全局训练状态字典
            get_background_db_func: 获取后台数据库连接的函数
        """
        self.config = config
        self.session_id = session_id
        self.user_id = user_id
        self.queue = queue
        self.training_status = training_status
        self.get_background_db_func = get_background_db_func
        
        # 创建数据库连接
        self.bg_db = get_background_db_func()
        self.bg_session_manager = TrainingSessionManager(self.bg_db) if self.bg_db else None
        self.bg_file_manager = FileManager(self.bg_db) if self.bg_db else None
        
        # 创建用户专属目录
        self.user_plots_dir = get_user_plot_dir(user_id)
        self.user_checkpoints_dir = get_user_checkpoint_dir(user_id)
        
        # 获取配置参数
        self.epochs = int(config.get('epochs', 3))
        self.model_type = config.get('model', 'attention')
        self.problem_type = config.get('problem', 'tsp')
        self.batch_size = int(config.get('batch_size', 512))
        self.learning_rate = float(config.get('learning_rate', 1e-4))
        self.num_loc = 50  # 默认问题规模
        
        # ========== 新增：从配置中获取算法和策略名称 ==========
        self.algorithm_name = config.get('algorithm', 'reinforce').lower()
        self.policy_name = config.get('model', 'attention').lower()
        
        # 兼容性：映射旧的model值到新的策略名称
        model_mapping = {
            'attention': 'attention',
            'am': 'attention',
            'pomo': 'pomo',
            'symnco': 'symnco',
        }
        if self.policy_name in model_mapping:
            self.policy_name = model_mapping[self.policy_name]
        
        # 检测设备，支持从 config 指定 GPU 编号
        # gpu_id 为整数 → 使用该 GPU；为 None → 强制 CPU（与用户在 UI 选择一致）
        gpu_id = config.get('gpu_id', None)
        if gpu_id is not None and torch.cuda.is_available():
            try:
                gpu_id = int(gpu_id)
                if 0 <= gpu_id < torch.cuda.device_count():
                    self.device = torch.device(f"cuda:{gpu_id}")
                    self.accelerator = "gpu"
                    self.devices = [gpu_id]
                else:
                    print(f"警告: gpu_id={gpu_id} 超出范围，回退到 cuda:0")
                    self.device = torch.device("cuda:0")
                    self.accelerator = "gpu"
                    self.devices = [0]
            except (ValueError, TypeError):
                self.device = torch.device("cpu")
                self.accelerator = "cpu"
                self.devices = "auto"
        else:
            # gpu_id 为 None（用户选 CPU）或 CUDA 不可用，均使用 CPU
            self.device = torch.device("cpu")
            self.accelerator = "cpu"
            self.devices = "auto"
            if gpu_id is not None and not torch.cuda.is_available():
                print("警告: 当前环境不支持 CUDA，已自动切换到 CPU 训练")
    
    def send_message(self, msg_type, message, **kwargs):
        """发送消息到队列"""
        msg = {
            'type': msg_type,
            'message': message,
            **kwargs
        }
        self.queue.put(json.dumps(msg))
    
    def initialize_environment(self):
        """初始化环境（子类实现）"""
        raise NotImplementedError("子类必须实现 initialize_environment 方法")
    
    def create_policy(self, env):
        """创建策略网络（支持多种策略模型）"""
        # ========== 使用新的策略注册表（如果可用） ==========
        if MODULES_AVAILABLE:
            try:
                PolicyClass = get_policy_class(self.policy_name)
                policy_wrapper = PolicyClass(self.config)
                
                # 验证配置
                valid, error_msg = policy_wrapper.validate_config()
                if not valid:
                    self.send_message('error', f'策略配置无效: {error_msg}')
                    raise ValueError(error_msg)
                
                policy = policy_wrapper.create_policy(env)
                self.send_message('info', f'✅ 策略网络: {policy_wrapper.policy_name.upper()}')
                return policy
            except Exception as e:
                self.send_message('warning', f'使用新策略模块失败，降级到传统模式: {str(e)}')
                # 降级到传统模式
        
        # ========== 传统模式（向后兼容） ==========
        from rl4co.models import AttentionModelPolicy
        
        # ========== 关键调试信息：环境名称 ==========
        self.send_message('info', f'🔍 [DEBUG] 准备创建 AttentionModelPolicy')
        self.send_message('info', f'🔍 [DEBUG] 环境名称 (env.name): "{env.name}"')
        self.send_message('info', f'🔍 [DEBUG] 问题类型 (problem_type): "{self.problem_type}"')
        
        # ========== 环境名称处理 ==========
        # 获取实际的环境名称（统一转小写）
        actual_env_name = env.name.lower()
        mapped_env_name = actual_env_name
        
        self.send_message('info', f'✅ [环境名称] 使用: "{mapped_env_name}"')
        self.send_message('info', f'💡 根据问题类型选择合适的策略和embedding')
        
        # ========== 尝试创建策略 ==========
        try:
            self.send_message('info', f'🔧 正在创建 AttentionModelPolicy (env_name="{mapped_env_name}")...')
            
            policy = AttentionModelPolicy(
                env_name=mapped_env_name,  # 使用映射后的名称
                embed_dim=self.config.get('embed_dim', 128),
                num_encoder_layers=self.config.get('num_encoder_layers', 3),
                num_heads=self.config.get('num_heads', 8),
            )
            
            self.send_message('info', '✅ AttentionModelPolicy 创建成功 (传统模式)')
            return policy
            
        except KeyError as e:
            # 环境名称不支持
            self.send_message('info', f'❌ [ERROR] AttentionModelPolicy 不支持环境: "{mapped_env_name}"')
            self.send_message('info', f'💡 [SOLUTION] 可能的解决方案:')
            self.send_message('info', f'   1. 手动指定 init_embedding')
            self.send_message('info', f'   2. 使用其他策略模型')
            raise
        except Exception as e:
            self.send_message('info', f'❌ [ERROR] 创建策略失败: {str(e)}')
            raise
        
        self.send_message('info', '策略网络初始化完成 (传统模式)')
        return policy
    
    def _create_symnco_model(self, env, policy):
        """为 SymNCO 策略创建专用模型（内置自定义多损失训练算法）"""
        try:
            from rl4co.models.zoo.symnco import SymNCO
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含SymNCO。\n"
                "请安装最新版: pip install rl4co"
            )

        num_augment = int(self.config.get('num_augment', 8))
        num_starts = int(self.config.get('num_starts', 0))
        alpha = float(self.config.get('symnco_alpha', 0.2))
        beta = float(self.config.get('symnco_beta', 1.0))

        model = SymNCO(
            env,
            policy,
            baseline='symnco',
            num_augment=num_augment,
            num_starts=num_starts,
            alpha=alpha,
            beta=beta,
            batch_size=self.batch_size,
            train_data_size=10_000,
            val_data_size=1_000,
            optimizer_kwargs={'lr': self.learning_rate},
        )

        self.send_message(
            'info',
            f'✅ SymNCO模型创建成功 '
            f'(num_augment={num_augment}, num_starts={num_starts}, '
            f'alpha={alpha}, beta={beta})'
        )
        return model

    def create_model(self, env, policy):
        """创建RL模型（支持多种算法）"""
        # SymNCO 使用内置的自定义多损失训练算法，跳过算法注册表
        if self.policy_name == 'symnco':
            return self._create_symnco_model(env, policy)

        # ========== 使用新的算法注册表（如果可用） ==========
        if MODULES_AVAILABLE:
            try:
                AlgorithmClass = get_algorithm_class(self.algorithm_name)
                algorithm_wrapper = AlgorithmClass(self.config)
                
                # 验证配置
                valid, error_msg = algorithm_wrapper.validate_config()
                if not valid:
                    self.send_message('error', f'算法配置无效: {error_msg}')
                    raise ValueError(error_msg)
                
                model = algorithm_wrapper.create_model(env, policy)
                self.send_message('info', f'✅ RL算法: {algorithm_wrapper.algorithm_name.upper()}')
                return model
            except Exception as e:
                self.send_message('warning', f'使用新算法模块失败，降级到传统模式: {str(e)}')
                # 降级到传统模式
        
        # ========== 传统模式（向后兼容） ==========
        from rl4co.models import REINFORCE
        model = REINFORCE(
            env,
            policy,
            baseline=self.config.get('baseline', 'rollout'),
            batch_size=self.batch_size,
            train_data_size=10_000,
            val_data_size=1_000,
            optimizer_kwargs={"lr": self.learning_rate},
        )
        self.send_message('info', f'使用REINFORCE算法 (传统模式)')
        return model
    
    def train(self):
        """执行训练流程"""
        try:
            # 初始化训练状态
            self.training_status[self.session_id] = {
                'status': 'running',
                'progress': 0,
                'epoch': 0,
                'loss': 0,
                'reward': 0,
                'best_reward': 0
            }
            
            self.send_message('info', f'开始训练 {self.model_type.upper()} 模型，问题类型: {self.problem_type.upper()}')
            self.send_message('info', f'配置: Epochs={self.epochs}, Batch={self.batch_size}, LR={self.learning_rate}, 问题规模={self.num_loc}')
            self.send_message('info', f'使用设备: {self.device}')
            
            # 初始化环境
            env = self.initialize_environment()
            self.send_message('info', f'环境初始化完成: {env.name}')
            
            # 创建策略和模型
            policy = self.create_policy(env)
            model = self.create_model(env, policy)
            
            # 检查checkpoint
            checkpoint_path = os.path.join(self.user_checkpoints_dir, f"{self.problem_type}-{self.model_type}.ckpt")
            ckpt_path = checkpoint_path if os.path.exists(checkpoint_path) else None
            
            if ckpt_path:
                self.send_message('info', f'加载检查点: {checkpoint_path}')
            
            # 创建进度回调（传入 training_status 以便每个 epoch 同步更新指标）
            progress_callback = ProgressCallback(self.queue, self.session_id, self.epochs, self.user_id, self.training_status)
            
            # 初始化训练器
            trainer = RL4COTrainer(
                max_epochs=self.epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                callbacks=[progress_callback],
                logger=None,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            
            self.send_message('info', '开始训练...')
            
            # 开始训练
            if ckpt_path:
                trainer.fit(model, ckpt_path=ckpt_path)
            else:
                trainer.fit(model)
            
            self.send_message('info', '训练完成，开始生成可视化结果...')
            
            # 生成可视化
            results = self.generate_visualizations(env, model, trainer, checkpoint_path)
            
            # ========== 调试信息 ==========
            print("=" * 80)
            print("BaseTrainer.train() - 生成可视化完成")
            print("=" * 80)
            print(f"generate_visualizations 返回类型: {type(results)}")
            print(f"返回内容: {results}")
            if isinstance(results, dict):
                print(f"字典键: {list(results.keys())}")
                if 'plot_paths' in results:
                    print(f"plot_paths 长度: {len(results['plot_paths'])}")
                    print(f"plot_paths 内容: {results['plot_paths']}")
                if 'animation_paths' in results:
                    print(f"animation_paths 长度: {len(results['animation_paths'])}")
                    print(f"animation_paths 内容: {results['animation_paths']}")
            print("=" * 80)
            # ========== 调试信息结束 ==========
            
            # 训练完成
            self.training_status[self.session_id]['status'] = 'completed'
            
            # 更新训练会话状态到数据库
            if self.bg_session_manager:
                try:
                    self.bg_session_manager.update_session(
                        session_id=self.session_id,
                        status='completed',
                        end_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        final_loss=self.training_status[self.session_id]['loss'],
                        final_reward=self.training_status[self.session_id]['reward'],
                        best_reward=self.training_status[self.session_id]['best_reward'],
                        checkpoint_path=checkpoint_path
                    )
                except Exception as e:
                    print(f"更新训练会话状态失败: {str(e)}")
            
            # 优先从 progress_callback 读取最终指标（最准确），
            # 再兜底到 training_status（已由 callback 同步写入）
            final_loss = (progress_callback.history_losses[-1]
                          if progress_callback.history_losses
                          else self.training_status[self.session_id]['loss'])
            final_reward = (progress_callback.history_rewards[-1]
                            if progress_callback.history_rewards
                            else self.training_status[self.session_id]['reward'])
            best_reward = (progress_callback.best_reward
                           if progress_callback.best_reward != float('-inf')
                           else self.training_status[self.session_id]['best_reward'])

            final_results = {
                'model': self.model_type,
                'problem': self.problem_type,
                'strategy': self.algorithm_name.upper(),
                'total_epochs': self.epochs,
                'final_loss': round(float(final_loss), 4),
                'final_reward': round(float(final_reward), 4),
                'best_reward': round(float(best_reward), 4),
                **results
            }
            
            # ========== 调试信息 ==========
            print("=" * 80)
            print("BaseTrainer.train() - 准备发送 complete 消息")
            print("=" * 80)
            print(f"final_results 类型: {type(final_results)}")
            print(f"final_results 键: {list(final_results.keys())}")
            print(f"final_results 完整内容:")
            import json as json_module
            print(json_module.dumps(final_results, indent=2, ensure_ascii=False))
            print("=" * 80)
            # ========== 调试信息结束 ==========
            
            self.send_message('complete', '训练完成！', results=final_results)
            
        except Exception as e:
            import traceback
            error_msg = f'{str(e)}\n{traceback.format_exc()}'
            self.training_status[self.session_id]['status'] = 'error'
            
            # 更新训练会话状态为失败
            if self.bg_session_manager:
                try:
                    self.bg_session_manager.update_session(
                        session_id=self.session_id,
                        status='failed',
                        end_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    )
                except Exception as update_error:
                    print(f"更新失败状态失败: {str(update_error)}")
            
            self.send_message('error', f'训练出错: {str(e)}')
        
        finally:
            # 关闭后台数据库连接
            if self.bg_db:
                try:
                    self.bg_db.close()
                except:
                    pass
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """生成可视化结果（子类实现）"""
        raise NotImplementedError("子类必须实现 generate_visualizations 方法")



