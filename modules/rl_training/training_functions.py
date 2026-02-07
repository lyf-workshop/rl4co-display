"""
RL4CO 训练核心函数模块
包含强化学习训练的主要逻辑、进度回调和可视化功能
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# 导入 RL4CO 相关组件
try:
    from rl4co.envs import TSPEnv, CVRPEnv
    from rl4co.models import AttentionModelPolicy, REINFORCE
    from rl4co.utils.trainer import RL4COTrainer
    from lightning.pytorch.callbacks import Callback
    from tensordict import TensorDict
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    Callback = object  # 降级为普通对象基类
    TensorDict = None
    print("警告: RL4CO 库未安装，训练功能将不可用")

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


def create_route_animation(td, actions, save_path, title="路线生成过程", fps=2):
    """
    创建TSP路线逐步生成的动态GIF
    
    参数:
        td: TensorDict，包含城市坐标等信息
        actions: numpy数组，访问城市的顺序
        save_path: GIF保存路径
        title: 图表标题
        fps: 帧率（每秒帧数）
    """
    # 提取城市坐标
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs']).cpu().numpy()
    else:
        locs = td['locs'].cpu().numpy()
    
    num_cities = len(locs)
    frames = []
    
    # 计算每一步的累计距离
    def calculate_partial_distance(locs, actions, step):
        """计算到第step步为止的累计距离"""
        if step < 1:
            return 0.0
        total_dist = 0.0
        for i in range(step):
            city_a = locs[actions[i]]
            # 如果是最后一步，返回起点；否则继续下一个城市
            if i + 1 < len(actions):
                city_b = locs[actions[i + 1]]
            else:
                city_b = locs[actions[0]]  # 返回起点
            dist = np.sqrt(np.sum((city_a - city_b) ** 2))
            total_dist += dist
        return total_dist
    
    # 为每一步生成一帧图像
    for step in range(num_cities + 1):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制所有城市点
        ax.scatter(locs[:, 0], locs[:, 1], c='lightblue', s=200, 
                  zorder=3, alpha=0.6, edgecolors='black', linewidths=2)
        
        # 标注城市编号
        for i, (x, y) in enumerate(locs):
            ax.text(x, y, str(i), fontsize=10, ha='center', va='center',
                   fontweight='bold', color='darkblue')
        
        # 绘制已经构建的路径
        if step > 0:
            for i in range(step):
                start = locs[actions[i]]
                if i + 1 < len(actions):
                    end = locs[actions[i + 1]]
                else:
                    end = locs[actions[0]]  # 最后返回起点
                
                # 绘制路径线
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       'b-', linewidth=3, alpha=0.7, zorder=1)
                
                # 添加箭头表示方向
                mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
                dx, dy = end[0] - start[0], end[1] - start[1]
                ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                          xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                          arrowprops=dict(arrowstyle='->', color='blue', 
                                        lw=2, alpha=0.7))
        
        # 高亮当前访问的城市
        if step > 0 and step <= num_cities:
            current_city = actions[step - 1]
            ax.scatter(locs[current_city, 0], locs[current_city, 1], 
                      c='red', s=400, zorder=5, marker='*', 
                      edgecolors='darkred', linewidths=2,
                      label=f'当前: 城市 {current_city}')
        
        # 高亮起点
        start_city = actions[0]
        ax.scatter(locs[start_city, 0], locs[start_city, 1], 
                  c='green', s=300, zorder=4, marker='s',
                  edgecolors='darkgreen', linewidths=2,
                  label=f'起点: 城市 {start_city}')
        
        # 计算当前累计成本
        current_cost = calculate_partial_distance(locs, actions, step)
        
        # 设置标题和信息
        if step == 0:
            info_text = "开始构建路线..."
        elif step < num_cities:
            info_text = f"第 {step} 步 | 已访问 {step} 个城市 | 累计成本: {current_cost:.3f}"
        else:
            # 最后一步，返回起点
            final_dist = np.sqrt(np.sum((locs[actions[-1]] - locs[actions[0]]) ** 2))
            total_cost = current_cost + final_dist
            info_text = f"完成！总共 {num_cities} 个城市 | 总成本: {total_cost:.3f}"
        
        ax.set_title(f"{title}\n{info_text}", fontsize=14, fontweight='bold', pad=20)
        
        # 设置坐标轴
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        
        # 添加图例
        if step > 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # 添加进度条
        progress = step / num_cities
        ax.text(0.5, -0.12, f"进度: {int(progress * 100)}%", 
               ha='center', va='top', transform=ax.transAxes,
               fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 保存当前帧为图像
        fig.tight_layout()
        
        # 将图形转换为PIL Image（兼容新旧版本matplotlib）
        fig.canvas.draw()
        try:
            # 新版本 matplotlib (>= 3.8)
            buf = fig.canvas.buffer_rgba()
            image = np.frombuffer(buf, dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # 转换 RGBA 到 RGB
            image = image[:, :, :3]
        except AttributeError:
            # 旧版本 matplotlib
            try:
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # 更老的版本，使用 tostring_argb
                buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # ARGB 转 RGB
                image = buf[:, :, 1:]
        
        frames.append(Image.fromarray(image))
        
        plt.close(fig)
    
    # 在最后一帧停留更长时间
    for _ in range(3):
        frames.append(frames[-1])
    
    # 保存为GIF
    duration = int(1000 / fps)  # 每帧持续时间（毫秒）
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False
    )


class ProgressCallback(Callback):
    """Lightning回调类，用于捕获训练进度并推送到消息队列"""
    
    def __init__(self, queue, session_id, total_epochs, user_id):
        super().__init__()
        self.queue = queue  # 与前端通信的消息队列
        self.session_id = session_id  # 当前训练会话ID
        self.total_epochs = total_epochs  # 总训练轮数
        self.user_id = user_id  # 用户ID
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
                    'message': f'可用的 callback_metrics 键: {list(metrics.keys())}'
                }))
                if hasattr(trainer, 'logged_metrics'):
                    self.queue.put(json.dumps({
                        'type': 'info',
                        'message': f'可用的 logged_metrics 键: {list(trainer.logged_metrics.keys())}'
                    }))
            
            # RL4CO 的 REINFORCE 模型使用的键名
            if loss == 0.0:
                loss = metrics.get('loss', metrics.get('train_loss', metrics.get('train/loss', 0.0)))
            if reward == 0.0:
                reward = metrics.get('reward', metrics.get('train_reward', metrics.get('train/reward', 0.0)))
            
            # 如果还是没有找到，尝试从 logged_metrics 获取
            if loss == 0.0 and hasattr(trainer, 'logged_metrics'):
                logged = trainer.logged_metrics
                loss = logged.get('loss', logged.get('train_loss', logged.get('train/loss', 0.0)))
            
            if reward == 0.0 and hasattr(trainer, 'logged_metrics'):
                logged = trainer.logged_metrics
                reward = logged.get('reward', logged.get('train_reward', logged.get('train/reward', 0.0)))
            
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
        
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
        
        # 发送进度更新（需要更新全局 training_status）
        # 注意：这里需要从 app.py 传入 training_status 的引用
        # 为了解耦，我们通过队列发送所有信息
        
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


def real_rl4co_training(config, session_id, user_id, queue, training_status, get_background_db_func):
    """
    使用 RL4CO 进行真实的强化学习训练
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列（用于推送进度）
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    # 为后台线程创建独立的数据库连接
    bg_db = get_background_db_func()
    bg_session_manager = TrainingSessionManager(bg_db) if bg_db else None
    bg_file_manager = FileManager(bg_db) if bg_db else None
    
    try:
        # 创建用户专属目录
        USER_PLOTS_DIR = get_user_plot_dir(user_id)
        USER_CHECKPOINTS_DIR = get_user_checkpoint_dir(user_id)
        
        # 初始化训练状态
        training_status[session_id] = {
            'status': 'running',
            'progress': 0,
            'epoch': 0,
            'loss': 0,
            'reward': 0,
            'best_reward': 0
        }
        
        # 获取配置参数
        epochs = int(config.get('epochs', 3))
        model_type = config.get('model', 'attention')
        problem_type = config.get('problem', 'tsp')
        batch_size = int(config.get('batch_size', 512))
        learning_rate = float(config.get('learning_rate', 1e-4))
        num_loc = 50  # 问题规模（TSP点数）
        
        # 检查是否使用上传的数据集
        dataset_mode = config.get('dataset_mode', 'random')
        dataset_id = config.get('dataset_id', None)
        custom_dataset = None
        
        if dataset_mode == 'upload' and dataset_id:
            # 加载上传的数据集
            dataset_path = os.path.join('datasets', f'user_{user_id}', f'{dataset_id}.json')
            if os.path.exists(dataset_path):
                try:
                    with open(dataset_path, 'r') as f:
                        dataset_data = json.load(f)
                        custom_dataset = dataset_data['coordinates']
                        num_loc = len(custom_dataset)
                        queue.put(json.dumps({
                            'type': 'info',
                            'message': f'✅ 已加载自定义数据集: {dataset_data["filename"]} ({num_loc}个城市)'
                        }))
                except Exception as e:
                    queue.put(json.dumps({
                        'type': 'info',
                        'message': f'⚠️ 加载数据集失败: {str(e)}，将使用随机生成'
                    }))
                    custom_dataset = None
            else:
                queue.put(json.dumps({
                    'type': 'info',
                    'message': '⚠️ 数据集文件不存在，将使用随机生成'
                }))
        elif dataset_mode == 'upload':
            queue.put(json.dumps({
                'type': 'info',
                'message': '⚠️ 未找到数据集ID，将使用随机生成'
            }))
        
        # 发送训练开始消息
        queue.put(json.dumps({
            'type': 'info',
            'message': f'开始训练 {model_type.upper()} 模型，问题类型: {problem_type.upper()}'
        }))
        
        queue.put(json.dumps({
            'type': 'info',
            'message': f'配置: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, 问题规模={num_loc}'
        }))
        
        # 检测设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1 if torch.cuda.is_available() else "auto"
        
        queue.put(json.dumps({
            'type': 'info',
            'message': f'使用设备: {device}'
        }))
        
        # 初始化环境
        if problem_type.lower() == 'tsp':
            env = TSPEnv(generator_params={'num_loc': num_loc})
        elif problem_type.lower() == 'cvrp':
            env = CVRPEnv(generator_params={'num_loc': num_loc})
        else:
            env = TSPEnv(generator_params={'num_loc': num_loc})
        
        queue.put(json.dumps({
            'type': 'info',
            'message': f'环境初始化完成: {env.name}'
        }))
        
        # 定义策略网络
        policy = AttentionModelPolicy(
            env_name=env.name,
            embed_dim=128,
            num_encoder_layers=3,
            num_heads=8,
        )
        
        queue.put(json.dumps({
            'type': 'info',
            'message': '策略网络初始化完成'
        }))
        
        # 定义 RL 模型
        if custom_dataset is not None:
            # 使用自定义数据集
            coords_tensor = torch.tensor(custom_dataset, dtype=torch.float32)
            
            model = REINFORCE(
                env,
                policy,
                baseline="rollout",
                batch_size=min(batch_size, len(custom_dataset)),
                train_data_size=min(10_000, len(custom_dataset) * 20),
                val_data_size=min(1_000, len(custom_dataset) * 2),
                optimizer_kwargs={"lr": learning_rate},
            )
            
            queue.put(json.dumps({
                'type': 'info',
                'message': f'使用自定义数据集训练模式（{len(custom_dataset)}个城市）'
            }))
        else:
            # 使用随机生成数据集
            model = REINFORCE(
                env,
                policy,
                baseline="rollout",
                batch_size=batch_size,
                train_data_size=10_000,
                val_data_size=1_000,
                optimizer_kwargs={"lr": learning_rate},
            )
            
            queue.put(json.dumps({
                'type': 'info',
                'message': f'使用随机生成数据集训练模式（{num_loc}个城市）'
            }))
        
        # 检查是否有已保存的 checkpoint
        checkpoint_path = os.path.join(USER_CHECKPOINTS_DIR, f"{problem_type}-{model_type}.ckpt")
        ckpt_path = checkpoint_path if os.path.exists(checkpoint_path) else None
        
        if ckpt_path:
            queue.put(json.dumps({
                'type': 'info',
                'message': f'加载检查点: {checkpoint_path}'
            }))
        
        # 创建进度回调
        progress_callback = ProgressCallback(queue, session_id, epochs, user_id)
        
        # 初始化训练器
        trainer = RL4COTrainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=[progress_callback],
            logger=None,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        
        queue.put(json.dumps({
            'type': 'info',
            'message': '开始训练...'
        }))
        
        # 开始训练
        if ckpt_path:
            trainer.fit(model, ckpt_path=ckpt_path)
        else:
            trainer.fit(model)
        
        queue.put(json.dumps({
            'type': 'info',
            'message': '训练完成，开始生成可视化结果...'
        }))
        
        # 训练后测试并生成可视化
        policy = model.policy.to(device)
        
        # 如果使用了自定义数据集，在该数据集上测试
        if custom_dataset is not None:
            # 使用上传的数据集进行测试
            td_init = env.reset(batch_size=[1]).to(device)
            
            # 替换坐标为自定义数据集
            coords_tensor = torch.tensor([custom_dataset], dtype=torch.float32).to(device)
            td_init['locs'] = coords_tensor
            
            queue.put(json.dumps({
                'type': 'info',
                'message': f'✅ 在上传的数据集上进行测试（{len(custom_dataset)}个城市）'
            }))
        else:
            # 使用随机生成的测试数据
            td_init = env.reset(batch_size=[3]).to(device)
        
        # 未训练模型测试（使用随机策略）
        out_untrained = policy(td_init.clone(), phase="test", decode_type="sampling", return_actions=True)
        actions_untrained = out_untrained['actions'].cpu().detach()
        rewards_untrained = out_untrained['reward'].cpu().detach()
        
        # 训练后模型测试
        out_trained = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
        actions_trained = out_trained['actions'].cpu().detach()
        rewards_trained = out_trained['reward'].cpu().detach()
        
        # 生成对比图
        plot_paths = []
        animation_paths = []
        
        for i, td in enumerate(td_init):
            # 生成静态对比图
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            env.render(td, actions_untrained[i], ax=axs[0])
            env.render(td, actions_trained[i], ax=axs[1])
            axs[0].set_title(f"Random | Cost = {-rewards_untrained[i].item():.3f}")
            axs[1].set_title(f"Trained | Cost = {-rewards_trained[i].item():.3f}")
            
            plot_filename = f"comparison_{session_id[:8]}_{i+1}.png"
            plot_path = os.path.join(USER_PLOTS_DIR, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            # 保存文件记录到数据库
            if bg_file_manager:
                try:
                    bg_file_manager.save_file_record(
                        user_id=user_id,
                        session_id=session_id,
                        filename=plot_filename,
                        file_type='plot',
                        file_path=plot_path
                    )
                except Exception as e:
                    print(f"保存文件记录失败: {str(e)}")
            
            plot_paths.append(f"/static/model_plots/user_{user_id}/{plot_filename}")
            
            # 生成动态路线构建过程GIF
            queue.put(json.dumps({
                'type': 'info',
                'message': f'正在生成动态路线图 {i+1}/3...'
            }))
            
            animation_filename = f"animation_{session_id[:8]}_{i+1}.gif"
            animation_path = os.path.join(USER_PLOTS_DIR, animation_filename)
            
            # 生成训练后路线的逐步构建动画
            create_route_animation(
                td, 
                actions_trained[i].cpu().numpy(), 
                animation_path,
                title="训练后路线生成过程"
            )
            
            # 保存文件记录到数据库
            if bg_file_manager:
                try:
                    bg_file_manager.save_file_record(
                        user_id=user_id,
                        session_id=session_id,
                        filename=animation_filename,
                        file_type='animation',
                        file_path=animation_path
                    )
                except Exception as e:
                    print(f"保存文件记录失败: {str(e)}")
            
            animation_paths.append(f"/static/model_plots/user_{user_id}/{animation_filename}")
        
        # 保存检查点
        trainer.save_checkpoint(checkpoint_path)
        
        # 保存checkpoint文件记录到数据库
        if bg_file_manager:
            try:
                checkpoint_filename = os.path.basename(checkpoint_path)
                bg_file_manager.save_file_record(
                    user_id=user_id,
                    session_id=session_id,
                    filename=checkpoint_filename,
                    file_type='checkpoint',
                    file_path=checkpoint_path
                )
            except Exception as e:
                print(f"保存checkpoint记录失败: {str(e)}")
        
        queue.put(json.dumps({
            'type': 'info',
            'message': f'检查点已保存: {checkpoint_path}'
        }))
        
        # 训练完成
        training_status[session_id]['status'] = 'completed'
        
        # 更新训练会话状态到数据库
        if bg_session_manager:
            try:
                bg_session_manager.update_session(
                    session_id=session_id,
                    status='completed',
                    end_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    final_loss=training_status[session_id]['loss'],
                    final_reward=training_status[session_id]['reward'],
                    best_reward=training_status[session_id]['best_reward'],
                    checkpoint_path=checkpoint_path
                )
            except Exception as e:
                print(f"更新训练会话状态失败: {str(e)}")
        
        final_results = {
            'model': model_type,
            'problem': problem_type,
            'strategy': 'REINFORCE',
            'total_epochs': epochs,
            'final_loss': training_status[session_id]['loss'],
            'final_reward': training_status[session_id]['reward'],
            'best_reward': training_status[session_id]['best_reward'],
            'plot_paths': plot_paths,
            'animation_paths': animation_paths,
            'training_curve': training_status[session_id].get('plot_url', ''),
            'checkpoint_path': checkpoint_path
        }
        
        queue.put(json.dumps({
            'type': 'complete',
            'message': '训练完成！',
            'results': final_results
        }))
        
    except Exception as e:
        import traceback
        error_msg = f'{str(e)}\n{traceback.format_exc()}'
        training_status[session_id]['status'] = 'error'
        
        # 更新训练会话状态为失败
        if bg_session_manager:
            try:
                bg_session_manager.update_session(
                    session_id=session_id,
                    status='failed',
                    end_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
            except Exception as update_error:
                print(f"更新失败状态失败: {str(update_error)}")
        
        queue.put(json.dumps({
            'type': 'error',
            'message': f'训练出错: {str(e)}'
        }))
    
    finally:
        # 关闭后台数据库连接
        if bg_db:
            try:
                bg_db.close()
            except:
                pass

