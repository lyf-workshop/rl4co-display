"""
RL4CO 训练基类和通用组件
包含所有问题类型共享的训练逻辑、回调函数等
"""

import os
import json
import time
import logging
import threading
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger('rl4co_display')

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
    logger.warning("RL4CO 库未安装，训练功能将不可用")

# ========== 导入算法和策略模块 ==========
try:
    from modules.algorithms import get_algorithm_class
    from modules.policies import get_policy_class
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logger.warning("算法/策略模块未找到，使用传统模式")

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


def _prepare_checkpoint_safe_globals():
    """
    为 PyTorch 2.6+ 将 RL4CO 所有环境类加入 torch.load 安全白名单。
    PyTorch 2.6 起 weights_only 默认为 True，自定义类需显式声明可信才能反序列化。
    """
    try:
        import importlib
        import torch.serialization as _ser

        _candidates = [
            ('rl4co.envs',                          ['TSPEnv', 'CVRPEnv', 'MTSPEnv', 'SDVRPEnv']),
            ('rl4co.envs.routing',                  ['ATSPEnv', 'PCTSPEnv', 'SPCTSPEnv', 'PDPEnv', 'OPEnv']),
            ('rl4co.envs.routing.atsp.env',         ['ATSPEnv']),
            ('rl4co.envs.scheduling',               ['FFSPEnv']),
            ('modules.envs.vrptw_env_wrapper',      ['CVRPEnvWithTimeWindows']),
        ]

        safe_classes = []
        for module_path, class_names in _candidates:
            try:
                mod = importlib.import_module(module_path)
                for name in class_names:
                    cls = getattr(mod, name, None)
                    if cls is not None and cls not in safe_classes:
                        safe_classes.append(cls)
            except (ImportError, ModuleNotFoundError):
                pass

        if safe_classes and hasattr(_ser, 'add_safe_globals'):
            _ser.add_safe_globals(safe_classes)
    except Exception:
        pass  # 加载失败时由 trainer.fit 自行报错，不影响正常流程


class ProgressCallback(Callback):
    """Lightning回调类，用于捕获训练进度并推送到消息队列"""
    
    def __init__(self, queue, session_id, total_epochs, user_id,
                 training_status=None, file_manager=None, pause_event=None):
        super().__init__()
        self.queue = queue
        self.session_id = session_id
        self.total_epochs = total_epochs
        self.user_id = user_id
        self.training_status = training_status
        # 由 BaseTrainer 传入已建立的后台 FileManager，避免 Callback 内部自建连接
        self.file_manager = file_manager
        # pause_event: threading.Event，set=运行，clear=暂停
        self.pause_event = pause_event
        self.best_reward = float('-inf')
        self.epoch_losses = []
        self.epoch_rewards = []
        self.history_losses = []
        self.history_rewards = []
        self.history_epochs = []
    
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
    
    def _save_plot_async(self, epochs, losses, rewards, best_reward,
                         plot_path, plot_url, epoch):
        """在后台线程中渲染并保存训练曲线，避免阻塞 epoch 边界"""
        def _save():
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=6, label='Loss')
                ax1.set_xlabel('Epoch', fontsize=12)
                ax1.set_ylabel('Loss', fontsize=12)
                ax1.set_title('训练Loss变化曲线', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, linestyle='--')
                ax1.legend(loc='upper right', fontsize=10)

                ax2.plot(epochs, rewards, 'g-o', linewidth=2, markersize=6, label='Reward')
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('Reward', fontsize=12)
                ax2.set_title('训练Reward变化曲线', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.legend(loc='lower right', fontsize=10)

                best_epoch_idx = rewards.index(max(rewards))
                best_epoch_num = epochs[best_epoch_idx]
                ax2.axhline(y=best_reward, color='r', linestyle='--', alpha=0.5,
                            label=f'Best: {best_reward:.4f}')
                ax2.scatter([best_epoch_num], [best_reward], color='red',
                            s=100, zorder=5, marker='*')
                ax2.legend(loc='lower right', fontsize=10)

                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                if self.file_manager:
                    try:
                        self.file_manager.save_file_record(
                            user_id=self.user_id,
                            session_id=self.session_id,
                            filename=os.path.basename(plot_path),
                            file_type='curve',
                            file_path=plot_path,
                        )
                    except Exception as e:
                        logger.warning(f"保存文件记录失败: {e}")
            except Exception as e:
                logger.warning(f"后台绘图线程异常: {e}")

        t = threading.Thread(target=_save, daemon=True)
        t.start()

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
        
        # 生成实时训练曲线图（后台线程，不阻塞 epoch 边界）
        USER_PLOTS_DIR = get_user_plot_dir(self.user_id)
        plot_filename = f"training_curves_{self.session_id[:8]}.png"
        plot_path = os.path.join(USER_PLOTS_DIR, plot_filename)
        plot_url = f"/static/model_plots/user_{self.user_id}/{plot_filename}"

        # 复制历史列表（后台线程读取，避免与训练线程共享引用）
        self._save_plot_async(
            epochs=list(self.history_epochs),
            losses=list(self.history_losses),
            rewards=list(self.history_rewards),
            best_reward=self.best_reward,
            plot_path=plot_path,
            plot_url=plot_url,
            epoch=epoch,
        )

        # 立刻通知前端（图片由后台线程写入，极短时间内可用）
        self.queue.put(json.dumps({
            'type': 'plot',
            'plot_url': plot_url,
            'message': f'Epoch {epoch} 训练曲线已更新'
        }))
        
        # 同步更新全局 training_status，供训练结束时 final_results 读取
        # 使用单次 dict.update() 减少并发读取时看到部分状态的窗口
        if self.training_status is not None and self.session_id in self.training_status:
            self.training_status[self.session_id].update({
                'loss': round(loss, 4),
                'reward': round(reward, 4),
                'best_reward': round(self.best_reward, 4),
                'epoch': epoch,
                'progress': round(progress, 2),
            })

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

        # 中止检测：若已设置 stop_requested，通知 Lightning 终止并发送 stopped 事件
        if (self.training_status is not None
                and self.session_id in self.training_status
                and self.training_status[self.session_id].get('stop_requested')):
            trainer.should_stop = True
            self.training_status[self.session_id]['status'] = 'stopped'
            self.queue.put(json.dumps({
                'type': 'stopped',
                'message': f'训练已中止（已完成 Epoch {epoch}/{self.total_epochs}）',
                'epoch': epoch,
                'progress': round(progress, 2)
            }))
            return  # 跳过后续的暂停检测

        # 暂停检测：如果 pause_event 已被清除（clear），则阻塞直到恢复（set）
        if self.pause_event is not None and not self.pause_event.is_set():
            if self.training_status is not None and self.session_id in self.training_status:
                self.training_status[self.session_id]['status'] = 'paused'
            self.queue.put(json.dumps({
                'type': 'paused',
                'message': f'训练已暂停（已完成 Epoch {epoch}/{self.total_epochs}）',
                'epoch': epoch,
                'progress': round(progress, 2)
            }))
            self.pause_event.wait()  # 阻塞，直到 resume 将 event 重新 set
            if self.training_status is not None and self.session_id in self.training_status:
                self.training_status[self.session_id]['status'] = 'running'
            self.queue.put(json.dumps({
                'type': 'resumed',
                'message': f'训练已恢复，继续 Epoch {epoch + 1}'
            }))



class BaseTrainer:
    """强化学习训练器基类，提供通用训练逻辑"""
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func, pause_event=None):
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
        self.pause_event = pause_event

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
                    logger.warning(f"gpu_id={gpu_id} 超出范围，回退到 cuda:0")
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
                logger.warning("当前环境不支持 CUDA，已自动切换到 CPU 训练")

        # 自定义数据集（由子类调用 load_custom_dataset() 填充）
        self.custom_dataset_data = None

    def send_message(self, msg_type, message, **kwargs):
        """发送消息到队列"""
        msg = {
            'type': msg_type,
            'message': message,
            **kwargs
        }
        self.queue.put(json.dumps(msg))
    
    def load_custom_dataset(self):
        """加载用户上传的自定义数据集（各 trainer 在 __init__ 末尾调用）。"""
        dataset_mode = self.config.get('dataset_mode', 'random')
        dataset_id = self.config.get('dataset_id', None)
        problem_type = self.config.get('problem', 'tsp')

        self.custom_dataset_data = None

        if dataset_mode != 'upload' or not dataset_id:
            return

        path = os.path.join('datasets', f'user_{self.user_id}', f'{dataset_id}.json')
        if not os.path.exists(path):
            self.send_message('info', '⚠️ 数据集文件不存在，将使用随机生成')
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.send_message('info', f'⚠️ 读取数据集失败: {e}，将使用随机生成')
            return

        stored_type = data.get('problem_type', 'tsp')
        if stored_type != problem_type:
            self.send_message('info',
                f'⚠️ 数据集类型({stored_type})与当前问题({problem_type})不匹配，将使用随机生成')
            return

        self.custom_dataset_data = data
        self.num_loc = len(data['coordinates'])
        self.send_message('info',
            f'✅ 已加载自定义数据集: {data["filename"]} ({self.num_loc} 个节点)')

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

    def _create_mdam_model(self, env, policy):
        """为 MDAM 策略创建专用训练模型（内置多解码器 REINFORCE）"""
        try:
            from rl4co.models.zoo.mdam import MDAM
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含MDAM。\n"
                "请安装最新版: pip install rl4co"
            )

        model = MDAM(
            env,
            policy,
            baseline=self.config.get('baseline', 'rollout'),
            batch_size=self.batch_size,
            train_data_size=10_000,
            val_data_size=1_000,
            optimizer_kwargs={'lr': self.learning_rate},
        )
        self.send_message('info', '✅ MDAM模型创建成功（多解码器注意力）')
        return model

    def _create_deepaco_model(self, env, policy):
        """为 DeepACO 策略创建专用训练模型（深度蚁群 REINFORCE 子类）"""
        try:
            from rl4co.models.zoo.deepaco import DeepACO
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含DeepACO。\n"
                "请安装最新版: pip install rl4co"
            )

        model = DeepACO(
            env,
            policy,
            baseline='no',  # DeepACO 内部实现共享基线，跳过外部 baseline
            batch_size=self.batch_size,
            train_data_size=10_000,
            val_data_size=1_000,
            optimizer_kwargs={'lr': self.learning_rate},
        )
        self.send_message('info', '✅ DeepACO模型创建成功（深度蚁群优化）')
        return model

    def create_model(self, env, policy):
        """创建RL模型（支持多种算法）"""
        # SymNCO 使用内置的自定义多损失训练算法，跳过算法注册表
        if self.policy_name == 'symnco':
            return self._create_symnco_model(env, policy)

        # MDAM 使用内置的多解码器 REINFORCE 子类
        if self.policy_name == 'mdam':
            return self._create_mdam_model(env, policy)

        # DeepACO 使用内置的蚁群 REINFORCE 子类
        if self.policy_name == 'deepaco':
            return self._create_deepaco_model(env, policy)

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
            
            # 检查点管理
            # 每次训练后会将模型保存到此路径（problem_type + model_type 唯一标识）
            checkpoint_path = os.path.join(self.user_checkpoints_dir, f"{self.problem_type}-{self.model_type}.ckpt")
            resume_checkpoint = self.config.get('resume_checkpoint', False)
            ckpt_path = None

            if resume_checkpoint:
                if os.path.exists(checkpoint_path):
                    # PyTorch 2.6+ 默认 weights_only=True，需要先将 RL4CO 环境类加入安全白名单
                    _prepare_checkpoint_safe_globals()
                    ckpt_path = checkpoint_path
                    self.send_message('info', f'🔄 从检查点恢复训练: {os.path.basename(checkpoint_path)}')
                else:
                    self.send_message('warning', '⚠️ 未找到历史检查点，将从头开始全新训练')
            else:
                if os.path.exists(checkpoint_path):
                    self.send_message('info', '🆕 全新训练（已跳过历史检查点，如需继续上次训练请勾选"继续上次训练"）')
                else:
                    self.send_message('info', '🆕 开始全新训练')
            
            # 创建进度回调
            # training_status: 每个 epoch 结束后同步写入指标供 final_results 读取
            # file_manager: 复用 BaseTrainer 已建立的后台连接，避免 Callback 内部自建新连接
            progress_callback = ProgressCallback(
                self.queue, self.session_id, self.epochs, self.user_id,
                training_status=self.training_status,
                file_manager=self.bg_file_manager,
                pause_event=self.pause_event,
            )
            
            # 初始化训练器
            # logger=False: 完全禁用 Lightning 日志（logger=None 仍会创建默认 TensorBoardLogger/CSVLogger）
            # enable_checkpointing=False: 禁用 Lightning 自动存档（项目使用自己的检查点管理）
            trainer = RL4COTrainer(
                max_epochs=self.epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                callbacks=[progress_callback],
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                # 禁用 Lightning 在第一个 epoch 前强制跑的 sanity validation
                # 该 check 会跑 2 个完整 val batch（FFSP+POMO 时极慢），
                # 且对生产训练无实质价值
                num_sanity_val_steps=0,
            )
            
            self.send_message('info', '开始训练...')
            
            # 开始训练
            if ckpt_path:
                trainer.fit(model, ckpt_path=ckpt_path)
            else:
                trainer.fit(model)

            # 若训练被中止，ProgressCallback 已发送 stopped 事件，直接返回
            if self.training_status.get(self.session_id, {}).get('status') == 'stopped':
                return

            self.send_message('info', '训练完成，开始生成可视化结果...')
            
            # 生成可视化
            results = self.generate_visualizations(env, model, trainer, checkpoint_path)
            
            logger.debug("generate_visualizations 完成: keys=%s", list(results.keys()) if isinstance(results, dict) else type(results))
            
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
                    logger.error(f"更新训练会话状态失败: {e}", exc_info=True)
            
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
            
            logger.debug("final_results: loss=%.4f reward=%.4f best_reward=%.4f",
                         final_results.get('final_loss', 0),
                         final_results.get('final_reward', 0),
                         final_results.get('best_reward', 0))
            
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
                    logger.error(f"更新失败状态失败: {update_error}")
            
            # 把完整堆栈也推送到前端，方便诊断
            import traceback as _tb
            full_tb = _tb.format_exc()
            logger.error(f"训练出错 (session={self.session_id}):\n{full_tb}")
            self.send_message('error', f'训练出错: {str(e)}\n\n{full_tb}')
        
        finally:
            # 关闭后台数据库连接
            if self.bg_db:
                try:
                    self.bg_db.close()
                except Exception as e:
                    logger.warning(f"关闭后台数据库连接时出错: {e}")
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """生成可视化结果（子类实现）"""
        raise NotImplementedError("子类必须实现 generate_visualizations 方法")



