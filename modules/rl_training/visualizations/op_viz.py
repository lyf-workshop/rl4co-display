"""
OP (Orienteering Problem) 可视化函数

生成定向问题的路线动画和对比图
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, Tuple

logger = logging.getLogger('rl4co_display')


def create_op_route_animation(
    depot: np.ndarray,
    locs: np.ndarray,
    prize: np.ndarray,
    actions: np.ndarray,
    total_prize: float,
    max_length: float,
    save_dir: str,
    instance_id: int = 1,
    fps: int = 2
) -> Optional[str]:
    """
    创建 OP 路线动画（GIF）
    
    动画展示车辆如何在路径长度限制内收集奖励
    
    Args:
        depot: depot 坐标 [2]
        locs: 客户地点坐标 [num_loc, 2]
        prize: 每个地点的奖励值 [num_loc]
        actions: 访问序列 [seq_len]，索引为地点编号（0=depot, 1..n=customers）
        total_prize: 收集的总奖励（标量）
        max_length: 最大路径长度（标量）
        save_dir: 保存目录
        instance_id: 实例编号
        fps: 动画帧率
        
    Returns:
        保存的文件路径，如果失败返回 None
    """
    try:
        # 确保所有输入都是正确的类型 - 转换为 Python 原生类型
        if isinstance(depot, np.ndarray):
            depot = depot.flatten()
        
        # 处理 total_prize - 确保是 Python float
        if isinstance(total_prize, np.ndarray):
            total_prize = float(total_prize.flatten()[0] if total_prize.size > 0 else 0.0)
        elif hasattr(total_prize, 'item'):  # PyTorch tensor
            total_prize = float(total_prize.item())
        else:
            total_prize = float(total_prize)
        
        # 处理 max_length - 确保是 Python float
        if isinstance(max_length, np.ndarray):
            max_length = float(max_length.flatten()[0] if max_length.size > 0 else 2.0)
        elif hasattr(max_length, 'item'):  # PyTorch tensor
            max_length = float(max_length.item())
        else:
            max_length = float(max_length)
        
        num_locs = len(locs)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 归一化奖励值用于大小缩放（最小100，最大500）
        if prize.max() > prize.min():
            normalized_size = 100 + 400 * (prize - prize.min()) / (prize.max() - prize.min())
        else:
            normalized_size = np.full(len(prize), 250)
        
        # 颜色映射：根据奖励值
        colors = plt.cm.YlOrRd(0.3 + 0.7 * (prize - prize.min()) / (prize.max() - prize.min() + 1e-6))
        
        # 绘制 depot（黑色五角星）
        ax.scatter(depot[0], depot[1], c='black', marker='*', s=600, zorder=10, 
                  edgecolors='gold', linewidths=3, label='Depot')
        ax.text(depot[0], depot[1] - 0.05, 'Depot', ha='center', va='top', 
               fontsize=12, fontweight='bold')
        
        # 绘制所有地点（初始为空心）
        scatter = ax.scatter(locs[:, 0], locs[:, 1], 
                            c=colors, s=normalized_size, 
                            alpha=0.3, edgecolors='black', linewidths=2, 
                            zorder=5, label='Unvisited')
        
        # 标注奖励值
        for i in range(num_locs):
            ax.text(locs[i, 0] + 0.02, locs[i, 1] + 0.02, 
                   f'{prize[i]:.1f}', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # 初始化路线
        route_line, = ax.plot([], [], 'b-', linewidth=2.5, zorder=8, label='Route')
        
        # 当前位置标记
        current_marker, = ax.plot([], [], 'ro', markersize=15, zorder=9)
        
        # 访问过的点
        visited_scatter = ax.scatter([], [], c='green', s=300, marker='o', 
                                    edgecolors='darkgreen', linewidths=3, zorder=6)
        
        # 绘制路径长度限制圆圈（从depot出发）
        max_length_circle = plt.Circle((float(depot[0]), float(depot[1])), float(max_length), 
                                      color='red', fill=False, 
                                      linestyle='--', linewidth=2, 
                                      alpha=0.3, label=f'Max Length: {max_length:.2f}')
        ax.add_patch(max_length_circle)
        
        # 设置坐标轴
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 动态标题和统计信息
        title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, 
                            ha='center', fontsize=14, fontweight='bold')
        stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                            va='top', fontsize=10, 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 图例
        ax.legend(loc='upper right', fontsize=10)
        
        # 动画更新函数
        route_x, route_y = [depot[0]], [depot[1]]
        visited_indices = []
        current_length = 0.0
        current_prize = 0.0
        
        def update(frame):
            nonlocal current_length, current_prize
            
            if frame > 0 and frame <= len(actions):
                action_idx = actions[frame - 1]
                
                # 计算新位置
                if action_idx == 0:
                    # 返回 depot
                    new_loc = depot
                else:
                    # 访问客户（actions中的索引需要映射）
                    customer_idx = action_idx - 1
                    if customer_idx < len(locs):
                        new_loc = locs[customer_idx]
                        # 记录访问
                        if customer_idx not in visited_indices:
                            visited_indices.append(customer_idx)
                            current_prize += prize[customer_idx]
                
                # 更新路径
                if len(route_x) > 0:
                    dist = np.linalg.norm(new_loc - np.array([route_x[-1], route_y[-1]]))
                    current_length += dist
                
                route_x.append(new_loc[0])
                route_y.append(new_loc[1])
            
            # 更新路线
            route_line.set_data(route_x, route_y)
            
            # 更新当前位置标记
            if len(route_x) > 0:
                current_marker.set_data([route_x[-1]], [route_y[-1]])
            
            # 更新访问过的点
            if visited_indices:
                visited_locs = locs[visited_indices]
                visited_scatter.set_offsets(visited_locs)
            
            # 更新标题
            title_text.set_text(f'OP Route Animation (Instance {instance_id})\n'
                              f'Total Prize Collected: {current_prize:.2f} / {total_prize:.2f}')
            
            # 更新统计信息
            visit_rate = len(visited_indices) / num_locs * 100
            length_usage = current_length / max_length * 100 if max_length > 0 else 0
            stats_text.set_text(
                f'📊 Statistics:\n'
                f'  • Visited: {len(visited_indices)}/{num_locs} ({visit_rate:.1f}%)\n'
                f'  • Length: {current_length:.3f}/{max_length:.2f} ({length_usage:.1f}%)\n'
                f'  • Prize: {current_prize:.2f}'
            )
            
            return route_line, current_marker, visited_scatter, title_text, stats_text
        
        # 创建动画
        num_frames = len(actions) + 2
        anim = FuncAnimation(fig, update, frames=num_frames, 
                           interval=1000//fps, blit=True, repeat=True)
        
        # 保存为 GIF
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'op_route_animation_{instance_id}.gif')
        
        writer = PillowWriter(fps=fps)
        anim.save(filepath, writer=writer)
        
        plt.close(fig)
        
        logger.info(f"OP动画已保存: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"OP动画生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_op_comparison_plot(
    depot: np.ndarray,
    locs: np.ndarray,
    prize: np.ndarray,
    actions: np.ndarray,
    model_prize: float,
    max_length: float,
    save_dir: str,
    instance_id: int = 1,
    greedy_prize: Optional[float] = None
) -> Optional[str]:
    """
    创建 OP 对比图
    
    对比模型生成的路线和贪心算法路线
    
    Args:
        depot: depot 坐标 [2]
        locs: 客户地点坐标 [num_loc, 2]
        prize: 每个地点的奖励值 [num_loc]
        actions: 访问序列 [seq_len]
        model_prize: 模型收集的总奖励（标量）
        max_length: 最大路径长度（标量）
        save_dir: 保存目录
        instance_id: 实例编号
        greedy_prize: 贪心算法奖励（可选）
        
    Returns:
        保存的文件路径，如果失败返回 None
    """
    try:
        # 确保所有输入都是正确的类型 - 转换为 Python 原生类型
        if isinstance(depot, np.ndarray):
            depot = depot.flatten()
        
        # 处理 model_prize - 确保是 Python float
        if isinstance(model_prize, np.ndarray):
            model_prize = float(model_prize.flatten()[0] if model_prize.size > 0 else 0.0)
        elif hasattr(model_prize, 'item'):  # PyTorch tensor
            model_prize = float(model_prize.item())
        else:
            model_prize = float(model_prize)
        
        # 处理 max_length - 确保是 Python float
        if isinstance(max_length, np.ndarray):
            max_length = float(max_length.flatten()[0] if max_length.size > 0 else 2.0)
        elif hasattr(max_length, 'item'):  # PyTorch tensor
            max_length = float(max_length.item())
        else:
            max_length = float(max_length)
        
        # 处理 greedy_prize (如果提供)
        if greedy_prize is not None:
            if isinstance(greedy_prize, np.ndarray):
                greedy_prize = float(greedy_prize.flatten()[0] if greedy_prize.size > 0 else 0.0)
            elif hasattr(greedy_prize, 'item'):
                greedy_prize = float(greedy_prize.item())
            else:
                greedy_prize = float(greedy_prize)
        
        num_locs = len(locs)
        
        # 创建图形
        if greedy_prize is not None:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            axes = [ax]
        
        # 归一化奖励值用于大小缩放
        if prize.max() > prize.min():
            normalized_size = 100 + 400 * (prize - prize.min()) / (prize.max() - prize.min())
        else:
            normalized_size = np.full(len(prize), 250)
        
        colors = plt.cm.YlOrRd(0.3 + 0.7 * (prize - prize.min()) / (prize.max() - prize.min() + 1e-6))
        
        def plot_route(ax, title, route_actions, total_prize_collected):
            """绘制单个路线"""
            # 绘制 depot
            ax.scatter(depot[0], depot[1], c='black', marker='*', s=600, 
                      edgecolors='gold', linewidths=3, zorder=10)
            ax.text(depot[0], depot[1] - 0.05, 'Depot', 
                   ha='center', va='top', fontsize=11, fontweight='bold')
            
            # 找出访问过的地点
            visited_indices = []
            for action_idx in route_actions:
                if action_idx > 0:  # 不是depot
                    customer_idx = action_idx - 1
                    if customer_idx < len(locs) and customer_idx not in visited_indices:
                        visited_indices.append(customer_idx)
            
            # 绘制未访问的点（浅色空心）
            unvisited_mask = np.ones(num_locs, dtype=bool)
            unvisited_mask[visited_indices] = False
            if unvisited_mask.any():
                ax.scatter(locs[unvisited_mask, 0], locs[unvisited_mask, 1],
                          c=colors[unvisited_mask], s=normalized_size[unvisited_mask],
                          alpha=0.3, edgecolors='gray', linewidths=1, zorder=4)
            
            # 绘制访问过的点（深色实心）
            if visited_indices:
                visited_locs = locs[visited_indices]
                visited_colors = colors[visited_indices]
                visited_sizes = normalized_size[visited_indices]
                ax.scatter(visited_locs[:, 0], visited_locs[:, 1],
                          c=visited_colors, s=visited_sizes,
                          alpha=0.9, edgecolors='darkgreen', linewidths=3, zorder=6)
            
            # 标注奖励值
            for i in range(num_locs):
                color = 'green' if i in visited_indices else 'gray'
                weight = 'bold' if i in visited_indices else 'normal'
                ax.text(locs[i, 0] + 0.02, locs[i, 1] + 0.02, 
                       f'{prize[i]:.1f}',
                       fontsize=8, color=color, weight=weight,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # 绘制路线
            route_x, route_y = [depot[0]], [depot[1]]
            path_length = 0.0
            
            for action_idx in route_actions:
                if action_idx == 0:
                    new_loc = depot
                else:
                    customer_idx = action_idx - 1
                    if customer_idx < len(locs):
                        new_loc = locs[customer_idx]
                    else:
                        continue
                
                # 计算距离
                if len(route_x) > 0:
                    dist = np.linalg.norm(new_loc - np.array([route_x[-1], route_y[-1]]))
                    path_length += dist
                
                route_x.append(new_loc[0])
                route_y.append(new_loc[1])
            
            ax.plot(route_x, route_y, 'b-', linewidth=2.5, alpha=0.7, zorder=8)
            
            # 标注访问顺序
            for step, (x, y) in enumerate(zip(route_x[1:], route_y[1:]), 1):
                ax.text(x - 0.03, y - 0.03, str(step),
                       fontsize=7, color='blue', fontweight='bold',
                       bbox=dict(boxstyle='circle', facecolor='lightblue', alpha=0.7))
            
            # 绘制路径长度限制圆圈
            max_length_circle = plt.Circle((float(depot[0]), float(depot[1])), float(max_length),
                                          color='red', fill=False,
                                          linestyle='--', linewidth=2, alpha=0.3)
            ax.add_patch(max_length_circle)
            
            # 设置
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # 统计信息
            visit_rate = len(visited_indices) / num_locs * 100
            length_usage = path_length / max_length * 100 if max_length > 0 else 0
            
            title_str = f'{title}\n' \
                       f'Prize: {total_prize_collected:.2f} | ' \
                       f'Visited: {len(visited_indices)}/{num_locs} ({visit_rate:.1f}%) | ' \
                       f'Length: {path_length:.3f}/{max_length:.2f} ({length_usage:.1f}%)'
            ax.set_title(title_str, fontsize=12, fontweight='bold')
            
            # 图例
            legend_elements = [
                mpatches.Patch(color='green', label='Visited'),
                mpatches.Patch(color='gray', label='Unvisited'),
                mpatches.Patch(color='blue', label='Route'),
                mpatches.Patch(color='red', label='Max Length Limit')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # 绘制模型解
        plot_route(axes[0], f'Model Solution (Instance {instance_id})', actions, model_prize)
        
        # 如果有贪心解，绘制对比
        if greedy_prize is not None and len(axes) > 1:
            # 简单贪心：按奖励/距离比例访问
            greedy_actions = np.arange(1, min(num_locs + 1, len(actions)))
            plot_route(axes[1], f'Greedy Baseline (Instance {instance_id})', greedy_actions, greedy_prize)
            
            # 添加总标题
            improvement = ((model_prize - greedy_prize) / greedy_prize) * 100 if greedy_prize > 0 else 0
            fig.suptitle(
                f'OP Route Comparison (Instance {instance_id})\n'
                f'Model Prize: {model_prize:.2f} | Greedy Prize: {greedy_prize:.2f} | '
                f'Improvement: {improvement:+.2f}%',
                fontsize=14, fontweight='bold', y=0.98
            )
        
        plt.tight_layout()
        
        # 保存
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'op_comparison_{instance_id}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"OP对比图已保存: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"OP对比图生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def calculate_path_length(depot: np.ndarray, locs: np.ndarray, actions: np.ndarray) -> float:
    """
    计算路径总长度
    
    Args:
        depot: depot 坐标 [2]
        locs: 地点坐标 [num_loc, 2]
        actions: 访问序列 [seq_len]
        
    Returns:
        总长度
    """
    total_length = 0.0
    current_loc = depot
    
    for action_idx in actions:
        if action_idx == 0:
            next_loc = depot
        else:
            customer_idx = action_idx - 1
            if customer_idx < len(locs):
                next_loc = locs[customer_idx]
            else:
                continue
        
        total_length += np.linalg.norm(next_loc - current_loc)
        current_loc = next_loc
    
    return total_length
