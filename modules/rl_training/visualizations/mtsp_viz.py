"""
mTSP问题专用可视化函数
提供多旅行商问题的路线动画、对比图等可视化
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

logger = logging.getLogger('rl4co_display')

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 为不同代理定义颜色
AGENT_COLORS = [
    '#FF6B6B',  # 红色
    '#4ECDC4',  # 青色
    '#45B7D1',  # 蓝色
    '#FFA07A',  # 橙色
    '#98D8C8',  # 绿色
    '#C7CEEA',  # 紫色
    '#FFD93D',  # 黄色
    '#6BCB77',  # 浅绿
]


def extract_agent_routes(actions):
    """
    从动作序列中提取每个代理的路径
    
    参数:
        actions: numpy数组，包含所有城市访问顺序（0代表depot）
    
    返回:
        list: 每个代理的路径列表 [[agent1_route], [agent2_route], ...]
    """
    routes = []
    current_route = []
    
    for city in actions:
        if city == 0:  # 遇到depot
            if current_route:  # 如果当前路径非空，保存并开始新路径
                routes.append(current_route)
                current_route = []
        else:
            current_route.append(city)
    
    # 添加最后一条路径
    if current_route:
        routes.append(current_route)
    
    return routes


def calculate_route_distance(locs, route, include_depot=True):
    """
    计算单条路径的总距离
    
    参数:
        locs: 城市坐标数组
        route: 城市访问顺序（不含depot）
        include_depot: 是否包含从depot出发和返回depot的距离
    
    返回:
        float: 路径总距离
    """
    if len(route) == 0:
        return 0.0
    
    total_dist = 0.0
    depot = locs[0]
    
    # 从depot到第一个城市
    if include_depot:
        total_dist += np.sqrt(np.sum((depot - locs[route[0]]) ** 2))
    
    # 城市之间的距离
    for i in range(len(route) - 1):
        city_a = locs[route[i]]
        city_b = locs[route[i + 1]]
        total_dist += np.sqrt(np.sum((city_a - city_b) ** 2))
    
    # 从最后一个城市返回depot
    if include_depot:
        total_dist += np.sqrt(np.sum((locs[route[-1]] - depot) ** 2))
    
    return total_dist


def create_mtsp_route_animation(td, actions, save_path, title="mTSP路线生成过程", fps=2):
    """
    创建mTSP路线逐步生成的动态GIF
    
    参数:
        td: TensorDict，包含城市坐标等信息
        actions: numpy数组，访问城市的顺序（包含depot=0的分隔）
        save_path: GIF保存路径
        title: 图表标题
        fps: 帧率（每秒帧数）
    """
    # 提取城市坐标
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs'])
    else:
        locs = td['locs']
    
    # 如果是 Tensor，转换为 numpy
    if hasattr(locs, 'cpu'):
        locs = locs.cpu().numpy()
    
    # 确保 locs 是 2D 数组 (num_cities, 2)
    if locs.ndim == 3:
        locs = locs[0]  # 去掉 batch 维度
    
    # 提取每个代理的路径
    agent_routes = extract_agent_routes(actions)
    num_agents = len(agent_routes)
    
    frames = []
    
    # 为每个代理的每一步生成帧
    step_count = 0
    for agent_idx, route in enumerate(agent_routes):
        # 为每条路径的每一步生成帧
        for step in range(len(route) + 1):
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 绘制depot（仓库）
            depot = locs[0]
            ax.scatter(depot[0], depot[1], c='gold', s=500, 
                      marker='D', zorder=10, edgecolors='orange', linewidths=3,
                      label='Depot')
            ax.text(depot[0], depot[1], 'D', fontsize=14, ha='center', va='center',
                   fontweight='bold', color='black')
            
            # 绘制所有城市点
            customer_locs = locs[1:]
            ax.scatter(customer_locs[:, 0], customer_locs[:, 1], 
                      c='lightgray', s=150, zorder=3, alpha=0.6,
                      edgecolors='gray', linewidths=1.5)
            
            # 标注城市编号
            for i, (x, y) in enumerate(customer_locs):
                ax.text(x, y, str(i+1), fontsize=9, ha='center', va='center',
                       color='darkgray')
            
            # 绘制已完成的代理路径（之前的代理）
            total_distances = []
            for prev_agent_idx in range(agent_idx):
                prev_route = agent_routes[prev_agent_idx]
                color = AGENT_COLORS[prev_agent_idx % len(AGENT_COLORS)]
                
                # 绘制完整路径
                path_locs = [depot] + [locs[city] for city in prev_route] + [depot]
                path_x = [loc[0] for loc in path_locs]
                path_y = [loc[1] for loc in path_locs]
                ax.plot(path_x, path_y, color=color, linewidth=2.5, 
                       alpha=0.6, linestyle='--', zorder=2,
                       label=f'代理 {prev_agent_idx + 1}')
                
                # 标记已访问城市
                for city in prev_route:
                    ax.scatter(locs[city, 0], locs[city, 1], 
                             c=color, s=150, zorder=4, alpha=0.8,
                             edgecolors='black', linewidths=1.5)
                
                # 计算距离
                dist = calculate_route_distance(locs, prev_route)
                total_distances.append(dist)
            
            # 绘制当前代理的路径（到当前步）
            if step > 0:
                current_route = route[:step]
                color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
                
                # 绘制路径
                path_locs = [depot] + [locs[city] for city in current_route]
                if step == len(route):  # 如果完成，返回depot
                    path_locs.append(depot)
                
                path_x = [loc[0] for loc in path_locs]
                path_y = [loc[1] for loc in path_locs]
                ax.plot(path_x, path_y, color=color, linewidth=3, 
                       alpha=0.9, zorder=5,
                       label=f'代理 {agent_idx + 1} (当前)')
                
                # 添加箭头
                for i in range(len(path_locs) - 1):
                    mid_x = (path_locs[i][0] + path_locs[i+1][0]) / 2
                    mid_y = (path_locs[i][1] + path_locs[i+1][1]) / 2
                    dx = path_locs[i+1][0] - path_locs[i][0]
                    dy = path_locs[i+1][1] - path_locs[i][1]
                    ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1),
                              xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                              arrowprops=dict(arrowstyle='->', color=color, 
                                            lw=2.5, alpha=0.9))
                
                # 标记已访问城市
                for city in current_route:
                    ax.scatter(locs[city, 0], locs[city, 1], 
                             c=color, s=180, zorder=6, alpha=0.9,
                             edgecolors='black', linewidths=2)
                
                # 高亮当前城市
                if step < len(route):
                    current_city = route[step - 1]
                    ax.scatter(locs[current_city, 0], locs[current_city, 1],
                             c='red', s=400, marker='*', zorder=8,
                             edgecolors='darkred', linewidths=2,
                             label=f'当前位置: 城市 {current_city}')
                
                # 计算当前距离
                current_dist = calculate_route_distance(locs, current_route, 
                                                       include_depot=(step == len(route)))
                total_distances.append(current_dist)
            
            # 设置标题和信息
            info_lines = []
            info_lines.append(f"代理 {agent_idx + 1}/{num_agents}")
            
            if step == 0:
                info_lines.append("准备出发...")
            elif step < len(route):
                info_lines.append(f"已访问 {step} 个城市")
            else:
                info_lines.append("返回depot")
            
            # 显示距离信息
            if total_distances:
                max_dist = max(total_distances)
                sum_dist = sum(total_distances)
                info_lines.append(f"最长路径: {max_dist:.3f} | 总距离: {sum_dist:.3f}")
            
            ax.set_title(f"{title}\n{' | '.join(info_lines)}", 
                        fontsize=14, fontweight='bold', pad=20)
            
            # 设置坐标轴
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('X 坐标', fontsize=12)
            ax.set_ylabel('Y 坐标', fontsize=12)
            
            # 添加图例
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
            
            # 添加进度信息
            total_steps = sum(len(r) + 1 for r in agent_routes)
            step_count += 1
            progress = step_count / total_steps
            ax.text(0.5, -0.10, f"总进度: {int(progress * 100)}%", 
                   ha='center', va='top', transform=ax.transAxes,
                   fontsize=11, bbox=dict(boxstyle='round', 
                   facecolor='lightblue', alpha=0.5))
            
            # 保存当前帧
            fig.tight_layout()
            fig.canvas.draw()
            
            try:
                buf = fig.canvas.buffer_rgba()
                image = np.frombuffer(buf, dtype=np.uint8)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                image = image[:, :, :3]
            except AttributeError:
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            frames.append(Image.fromarray(image))
            plt.close(fig)
    
    # 保存为GIF
    if frames:
        duration = int(1000 / fps)  # 每帧持续时间（毫秒）
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                      duration=duration, loop=0, optimize=False)
        logger.info(f"mTSP动画已保存: {save_path}")
    else:
        logger.warning("mTSP动画：没有生成任何帧")


def create_mtsp_comparison_plot(td, actions, save_path, cost=None, 
                                 title="mTSP路线对比图"):
    """
    创建mTSP多代理路线对比图
    
    参数:
        td: TensorDict，包含城市坐标等信息
        actions: numpy数组，访问城市的顺序
        save_path: 图片保存路径
        cost: 可选，总成本或最大路径成本
        title: 图表标题
    """
    # 提取城市坐标
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs'])
    else:
        locs = td['locs']
    
    # 如果是 Tensor，转换为 numpy
    if hasattr(locs, 'cpu'):
        locs = locs.cpu().numpy()
    
    # 确保 locs 是 2D 数组 (num_cities, 2)
    if locs.ndim == 3:
        locs = locs[0]  # 去掉 batch 维度
    
    # 提取每个代理的路径
    agent_routes = extract_agent_routes(actions)
    num_agents = len(agent_routes)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制depot
    depot = locs[0]
    ax.scatter(depot[0], depot[1], c='gold', s=600, 
              marker='D', zorder=10, edgecolors='orange', linewidths=4,
              label='Depot')
    ax.text(depot[0], depot[1], 'D', fontsize=16, ha='center', va='center',
           fontweight='bold', color='black')
    
    # 绘制所有城市
    customer_locs = locs[1:]
    ax.scatter(customer_locs[:, 0], customer_locs[:, 1], 
              c='lightgray', s=200, zorder=3, alpha=0.6,
              edgecolors='gray', linewidths=2)
    
    # 标注城市编号
    for i, (x, y) in enumerate(customer_locs):
        ax.text(x, y, str(i+1), fontsize=10, ha='center', va='center',
               color='darkgray', fontweight='bold')
    
    # 绘制每个代理的路径并计算距离
    distances = []
    for agent_idx, route in enumerate(agent_routes):
        if len(route) == 0:
            continue
        
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        
        # 构建完整路径（depot -> cities -> depot）
        path_locs = [depot] + [locs[city] for city in route] + [depot]
        path_x = [loc[0] for loc in path_locs]
        path_y = [loc[1] for loc in path_locs]
        
        # 计算距离
        dist = calculate_route_distance(locs, route)
        distances.append(dist)
        
        # 绘制路径
        ax.plot(path_x, path_y, color=color, linewidth=3, alpha=0.8, zorder=5,
               label=f'代理 {agent_idx + 1} (距离: {dist:.3f})')
        
        # 添加方向箭头
        for i in range(len(path_locs) - 1):
            mid_x = (path_locs[i][0] + path_locs[i+1][0]) / 2
            mid_y = (path_locs[i][1] + path_locs[i+1][1]) / 2
            dx = path_locs[i+1][0] - path_locs[i][0]
            dy = path_locs[i+1][1] - path_locs[i][1]
            ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1),
                       xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                       arrowprops=dict(arrowstyle='->', color=color, 
                                     lw=2.5, alpha=0.8))
        
        # 标记访问的城市
        for city in route:
            ax.scatter(locs[city, 0], locs[city, 1], 
                     c=color, s=220, zorder=6, alpha=0.9,
                     edgecolors='black', linewidths=2)
    
    # 构建标题信息
    info_lines = [title]
    if distances:
        max_dist = max(distances)
        sum_dist = sum(distances)
        avg_dist = sum_dist / len(distances)
        info_lines.append(f"代理数: {num_agents} | 最长: {max_dist:.3f} | " +
                         f"总计: {sum_dist:.3f} | 平均: {avg_dist:.3f}")
    
    if cost is not None:
        info_lines.append(f"优化成本: {cost:.4f}")
    
    ax.set_title('\n'.join(info_lines), fontsize=14, fontweight='bold', pad=20)
    
    # 设置坐标轴
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X 坐标', fontsize=12)
    ax.set_ylabel('Y 坐标', fontsize=12)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
             bbox_to_anchor=(1.0, 1.0))
    
    # 添加统计信息框
    stats_text = f"城市总数: {len(customer_locs)}\n"
    stats_text += f"代理数量: {num_agents}\n"
    if distances:
        stats_text += f"负载均衡: {min(len(r) for r in agent_routes)}-{max(len(r) for r in agent_routes)} 城市/代理"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 保存图片
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"mTSP对比图已保存: {save_path}")
