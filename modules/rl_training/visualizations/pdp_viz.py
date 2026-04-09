"""
PDP (Pickup and Delivery Problem) 可视化函数

生成取送货问题的路线动画和对比图
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, Tuple

logger = logging.getLogger('rl4co_display')


def create_pdp_route_animation(
    depot: np.ndarray,
    pickups: np.ndarray,
    deliveries: np.ndarray,
    actions: np.ndarray,
    cost: float,
    save_dir: str,
    instance_id: int = 1,
    fps: int = 2
) -> Optional[str]:
    """
    创建 PDP 路线动画（GIF）
    
    动画展示车辆如何依次访问取货点和送货点
    
    Args:
        depot: depot 坐标 [2]
        pickups: 取货点坐标 [num_pairs, 2]
        deliveries: 送货点坐标 [num_pairs, 2]
        actions: 访问序列 [seq_len]，索引为地点编号
        cost: 路线总成本
        save_dir: 保存目录
        instance_id: 实例编号
        fps: 动画帧率
        
    Returns:
        保存的文件路径，如果失败返回 None
    """
    try:
        # 合并所有地点：depot + pickups + deliveries
        # 索引映射：0 = depot, 1..n = pickups, n+1..2n = deliveries
        num_pairs = len(pickups)
        
        # 构建完整的地点列表
        all_locs = np.vstack([depot.reshape(1, 2), pickups, deliveries])  # [1 + 2*num_pairs, 2]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 配色方案：每对取送货使用相同颜色
        colors = plt.cm.tab10(np.linspace(0, 1, num_pairs))
        
        # 绘制 depot（黑色五角星）
        ax.scatter(depot[0], depot[1], c='black', marker='*', s=500, zorder=5, label='Depot')
        
        # 绘制取货点（圆形）和送货点（方形）
        for i in range(num_pairs):
            # 取货点
            ax.scatter(
                pickups[i, 0], pickups[i, 1],
                c=[colors[i]], marker='o', s=300, edgecolors='black',
                linewidths=2, zorder=4, label=f'Pickup {i+1}' if i == 0 else ""
            )
            # 送货点
            ax.scatter(
                deliveries[i, 0], deliveries[i, 1],
                c=[colors[i]], marker='s', s=300, edgecolors='black',
                linewidths=2, zorder=4, label=f'Delivery {i+1}' if i == 0 else ""
            )
            
            # 绘制取送货配对连线（虚线）
            ax.plot(
                [pickups[i, 0], deliveries[i, 0]],
                [pickups[i, 1], deliveries[i, 1]],
                'k--', alpha=0.3, linewidth=1, zorder=1
            )
            
            # 标注编号
            ax.text(pickups[i, 0], pickups[i, 1], f'P{i+1}',
                   ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(deliveries[i, 0], deliveries[i, 1], f'D{i+1}',
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.text(depot[0], depot[1] - 0.05, 'Depot',
               ha='center', va='top', fontsize=12, fontweight='bold')
        
        # 初始化路线（空列表）
        route_line, = ax.plot([], [], 'r-', linewidth=2, zorder=3, label='Route')
        
        # 当前位置标记
        current_marker, = ax.plot([], [], 'ro', markersize=15, zorder=6)
        
        # 设置坐标轴
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'PDP Route Animation (Instance {instance_id})\nTotal Cost: {cost:.3f}',
                    fontsize=14, fontweight='bold')
        
        # 图例
        handles = [
            mpatches.Patch(color='black', label='Depot'),
            mpatches.Patch(color='gray', label='Pickup (○)'),
            mpatches.Patch(color='gray', label='Delivery (□)'),
            mpatches.Patch(color='red', label='Route')
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=10)
        
        # 动画更新函数
        route_x, route_y = [depot[0]], [depot[1]]
        
        def update(frame):
            if frame > 0 and frame <= len(actions):
                action_idx = actions[frame - 1]
                
                # 获取当前访问的地点坐标
                if action_idx < len(all_locs):
                    loc = all_locs[action_idx]
                    route_x.append(loc[0])
                    route_y.append(loc[1])
            
            # 更新路线
            route_line.set_data(route_x, route_y)
            
            # 更新当前位置标记
            if len(route_x) > 0:
                current_marker.set_data([route_x[-1]], [route_y[-1]])
            
            return route_line, current_marker
        
        # 创建动画
        num_frames = len(actions) + 2  # +2 for initial and final frames
        anim = FuncAnimation(fig, update, frames=num_frames, interval=1000//fps, blit=True, repeat=True)
        
        # 保存为 GIF
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'pdp_route_animation_{instance_id}.gif')
        
        writer = PillowWriter(fps=fps)
        anim.save(filepath, writer=writer)
        
        plt.close(fig)
        
        logger.info(f"PDP动画已保存: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"PDP动画生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_pdp_comparison_plot(
    depot: np.ndarray,
    pickups: np.ndarray,
    deliveries: np.ndarray,
    actions: np.ndarray,
    model_cost: float,
    save_dir: str,
    instance_id: int = 1,
    greedy_cost: Optional[float] = None
) -> Optional[str]:
    """
    创建 PDP 路线对比图
    
    对比模型生成的路线和贪心算法路线
    
    Args:
        depot: depot 坐标 [2]
        pickups: 取货点坐标 [num_pairs, 2]
        deliveries: 送货点坐标 [num_pairs, 2]
        actions: 访问序列 [seq_len]
        model_cost: 模型路线成本
        save_dir: 保存目录
        instance_id: 实例编号
        greedy_cost: 贪心算法成本（可选）
        
    Returns:
        保存的文件路径，如果失败返回 None
    """
    try:
        num_pairs = len(pickups)
        
        # 构建完整的地点列表
        all_locs = np.vstack([depot.reshape(1, 2), pickups, deliveries])
        
        # 创建图形（1行1列或1行2列）
        if greedy_cost is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            axes = [ax]
        
        # 配色方案
        colors = plt.cm.tab10(np.linspace(0, 1, num_pairs))
        
        def plot_route(ax, title, route_actions, route_cost):
            """绘制单个路线"""
            # 绘制 depot
            ax.scatter(depot[0], depot[1], c='black', marker='*', s=500, zorder=5)
            
            # 绘制取货点和送货点
            for i in range(num_pairs):
                ax.scatter(
                    pickups[i, 0], pickups[i, 1],
                    c=[colors[i]], marker='o', s=300, edgecolors='black',
                    linewidths=2, zorder=4
                )
                ax.scatter(
                    deliveries[i, 0], deliveries[i, 1],
                    c=[colors[i]], marker='s', s=300, edgecolors='black',
                    linewidths=2, zorder=4
                )
                
                # 配对虚线
                ax.plot(
                    [pickups[i, 0], deliveries[i, 0]],
                    [pickups[i, 1], deliveries[i, 1]],
                    'k--', alpha=0.3, linewidth=1, zorder=1
                )
                
                # 标注
                ax.text(pickups[i, 0], pickups[i, 1], f'P{i+1}',
                       ha='center', va='center', fontsize=9, fontweight='bold')
                ax.text(deliveries[i, 0], deliveries[i, 1], f'D{i+1}',
                       ha='center', va='center', fontsize=9, fontweight='bold')
            
            ax.text(depot[0], depot[1] - 0.05, 'Depot',
                   ha='center', va='top', fontsize=11, fontweight='bold')
            
            # 绘制路线
            route_x, route_y = [depot[0]], [depot[1]]
            for action_idx in route_actions:
                if action_idx < len(all_locs):
                    loc = all_locs[action_idx]
                    route_x.append(loc[0])
                    route_y.append(loc[1])
            
            ax.plot(route_x, route_y, 'r-', linewidth=2.5, alpha=0.7, zorder=3)
            
            # 标注访问顺序
            for step, (x, y) in enumerate(zip(route_x[1:], route_y[1:]), 1):
                ax.text(x + 0.02, y + 0.02, str(step),
                       fontsize=8, color='blue', fontweight='bold',
                       bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7))
            
            # 设置
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{title}\nCost: {route_cost:.3f}',
                        fontsize=12, fontweight='bold')
            
            # 图例
            handles = [
                mpatches.Patch(color='black', label='Depot'),
                mpatches.Patch(color='gray', label='Pickup (○)'),
                mpatches.Patch(color='gray', label='Delivery (□)'),
                mpatches.Patch(color='red', label='Route')
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=9)
        
        # 绘制模型路线
        plot_route(axes[0], f'Model Solution (Instance {instance_id})', actions, model_cost)
        
        # 如果有贪心解，绘制对比
        if greedy_cost is not None and len(axes) > 1:
            # 简单贪心：按距离顺序访问
            greedy_actions = np.arange(1, len(all_locs))  # 简化版
            plot_route(axes[1], f'Greedy Baseline (Instance {instance_id})', greedy_actions, greedy_cost)
            
            # 添加总标题
            improvement = ((greedy_cost - model_cost) / greedy_cost) * 100
            fig.suptitle(
                f'PDP Route Comparison (Instance {instance_id})\n'
                f'Improvement: {improvement:.2f}%',
                fontsize=14, fontweight='bold', y=0.98
            )
        
        plt.tight_layout()
        
        # 保存
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'pdp_comparison_{instance_id}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"PDP对比图已保存: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"PDP对比图生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def calculate_route_length(locs: np.ndarray, actions: np.ndarray) -> float:
    """
    计算路线总长度
    
    Args:
        locs: 所有地点坐标 [num_locs, 2]
        actions: 访问序列 [seq_len]
        
    Returns:
        总长度
    """
    total_length = 0.0
    current_loc = locs[0]  # 从 depot 开始
    
    for action_idx in actions:
        if action_idx < len(locs):
            next_loc = locs[action_idx]
            total_length += np.linalg.norm(next_loc - current_loc)
            current_loc = next_loc
    
    # 返回 depot
    total_length += np.linalg.norm(locs[0] - current_loc)
    
    return total_length
