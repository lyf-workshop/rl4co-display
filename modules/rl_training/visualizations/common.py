"""
通用可视化函数
提供所有问题类型共享的可视化工具
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def create_training_curve_plot(history_epochs, history_losses, history_rewards, save_path, best_reward):
    """
    创建训练曲线图（Loss和Reward）
    
    参数:
        history_epochs: epoch编号列表
        history_losses: loss历史数据
        history_rewards: reward历史数据
        save_path: 保存路径
        best_reward: 最佳reward值
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制Loss曲线
    ax1.plot(history_epochs, history_losses, 'b-o', linewidth=2, markersize=6, label='Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练Loss变化曲线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    
    # 绘制Reward曲线
    ax2.plot(history_epochs, history_rewards, 'g-o', linewidth=2, markersize=6, label='Reward')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('训练Reward变化曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=10)
    
    # 在reward图上标注最佳reward
    if history_rewards:
        best_epoch_idx = history_rewards.index(max(history_rewards))
        best_epoch_num = history_epochs[best_epoch_idx]
        ax2.axhline(y=best_reward, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_reward:.4f}')
        ax2.scatter([best_epoch_num], [best_reward], color='red', s=100, zorder=5, marker='*')
        ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



