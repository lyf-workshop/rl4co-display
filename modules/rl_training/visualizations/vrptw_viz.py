"""
VRPTW问题专用可视化函数
提供VRPTW路线动画、时间调度图等可视化
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


def create_vrptw_route_animation(td, actions, save_path, title="VRPTW路线生成过程（带时间窗）", fps=2):
    """
    创建VRPTW路线逐步生成的动态GIF，展示时间窗约束
    
    参数:
        td: TensorDict，包含客户坐标、需求、时间窗等信息
        actions: numpy数组，访问节点的顺序
        save_path: GIF保存路径
        title: 图表标题
        fps: 帧率（每秒帧数）
    """
    # 提取坐标信息
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs']).cpu().numpy()
        # 去掉batch维度（如果有）
        if locs.ndim == 3:  # shape: (batch, num_nodes, 2)
            locs = locs[0]  # 取第一个batch
        depot = locs[0]  # 仓库位置 - shape: (2,)
        customers = locs[1:]  # 客户位置 - shape: (num_customers, 2)
        
        demands = td.get('demand', td.get('demands', None))
        if demands is not None:
            demands = demands.cpu().numpy()
            if demands.ndim == 2:  # 去掉batch维度
                demands = demands[0]
        
        # 尝试提取时间窗信息
        time_windows = td.get('time_windows', None)
        if time_windows is not None:
            time_windows = time_windows.cpu().numpy()
            if time_windows.ndim == 3:  # 去掉batch维度
                time_windows = time_windows[0]
        else:
            # 如果没有时间窗信息，生成默认的
            time_windows = np.random.rand(len(customers), 2) * 100
            time_windows[:, 1] = time_windows[:, 0] + 50  # end = start + 50
    else:
        locs = td['locs'].cpu().numpy()
        if locs.ndim == 3:
            locs = locs[0]
        depot = locs[0]
        customers = locs[1:]
        
        demands = td.get('demand', td.get('demands', None))
        if demands is not None:
            demands = demands.cpu().numpy()
            if demands.ndim == 2:
                demands = demands[0]
        
        time_windows = td.get('time_windows', None)
        if time_windows is not None:
            time_windows = time_windows.cpu().numpy()
            if time_windows.ndim == 3:
                time_windows = time_windows[0]
        else:
            time_windows = np.random.rand(len(customers), 2) * 100
            time_windows[:, 1] = time_windows[:, 0] + 50
    
    num_nodes = len(locs)
    frames = []
    
    # 计算每一步的累计距离、载重和时间
    def calculate_metrics(locs, actions, step, capacity=1.0, speed=1.0, service_time=10.0):
        """计算到第step步为止的累计距离、当前载重和当前时间"""
        if step < 1:
            return 0.0, 0.0, 0.0
        
        total_dist = 0.0
        current_load = 0.0
        current_time = 0.0
        
        for i in range(step):
            node_a = actions[i]
            
            if i + 1 < len(actions):
                node_b = actions[i + 1]
            else:
                node_b = 0  # 返回仓库
            
            # 计算距离
            pos_a = locs[node_a]
            pos_b = locs[node_b]
            dist = np.sqrt(np.sum((pos_a - pos_b) ** 2))
            total_dist += dist
            
            # 计算旅行时间
            travel_time = dist / speed
            current_time += travel_time
            
            # 更新载重（如果访问客户）
            if node_a > 0 and demands is not None:  # 不是仓库
                demand_idx = node_a if len(demands) > node_a else node_a - 1
                if demand_idx < len(demands):
                    current_load += demands[demand_idx]
                
                # 检查时间窗约束
                if time_windows is not None and demand_idx < len(time_windows):
                    earliest, latest = time_windows[demand_idx]
                    # 如果到达太早，等待到最早开始时间
                    if current_time < earliest:
                        current_time = earliest
                
                # 添加服务时间
                current_time += service_time
            
            # 如果返回仓库，重置载重
            if node_b == 0:
                current_load = 0.0
        
        return total_dist, current_load, current_time
    
    # 为每一步生成一帧图像（使用双子图：地图+时间轴）
    for step in range(len(actions) + 1):
        fig = plt.figure(figsize=(16, 8))
        
        # 左侧：路线地图
        ax1 = plt.subplot(1, 2, 1)
        
        # 绘制仓库（特殊标记）
        ax1.scatter(float(depot[0]), float(depot[1]), c='red', s=400, marker='s', 
                   zorder=5, edgecolors='darkred', linewidths=3, label='仓库')
        
        # 绘制所有客户点，根据时间窗上色
        if time_windows is not None:
            # 根据时间窗早晚上色
            tw_starts = time_windows[:, 0]
            scatter = ax1.scatter(customers[:, 0], customers[:, 1], 
                                 c=tw_starts, s=200, cmap='coolwarm',
                                 zorder=3, alpha=0.7, edgecolors='black', linewidths=2)
            cbar = plt.colorbar(scatter, ax=ax1, label='时间窗起始时间')
        else:
            ax1.scatter(customers[:, 0], customers[:, 1], c='lightblue', s=200, 
                       zorder=3, alpha=0.6, edgecolors='black', linewidths=2)
        
        # 标注节点编号和时间窗
        ax1.text(float(depot[0]), float(depot[1]), '0\n仓库', fontsize=10, ha='center', va='center',
                fontweight='bold', color='white')
        
        for idx, (x, y) in enumerate(customers):
            customer_id = idx + 1
            if time_windows is not None:
                tw_text = f'\n[{time_windows[idx][0]:.0f}-{time_windows[idx][1]:.0f}]'
            else:
                tw_text = ''
            ax1.text(float(x), float(y), f'{customer_id}{tw_text}', fontsize=8, ha='center', va='center',
                    fontweight='bold', color='darkblue')
        
        # 绘制已经构建的路径
        if step > 0:
            for i in range(step):
                start_idx = actions[i]
                if i + 1 < len(actions):
                    end_idx = actions[i + 1]
                else:
                    end_idx = 0  # 最后返回仓库
                
                start_pos = locs[start_idx]
                end_pos = locs[end_idx]
                
                # 如果是返回仓库，使用虚线
                linestyle = '--' if end_idx == 0 else '-'
                linewidth = 2 if end_idx == 0 else 3
                
                # 绘制路径线
                ax1.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                        'b-' if linestyle == '-' else 'g--', 
                        linewidth=linewidth, alpha=0.7, zorder=1)
                
                # 添加箭头表示方向
                mid_x = float((start_pos[0] + end_pos[0]) / 2)
                mid_y = float((start_pos[1] + end_pos[1]) / 2)
                dx = float(end_pos[0] - start_pos[0])
                dy = float(end_pos[1] - start_pos[1])
                ax1.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                           xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                           arrowprops=dict(arrowstyle='->', 
                                         color='blue' if linestyle == '-' else 'green', 
                                         lw=2, alpha=0.7))
        
        # 高亮当前访问的节点
        if step > 0 and step <= len(actions):
            current_node = actions[step - 1]
            current_pos = locs[current_node]
            node_type = '仓库' if current_node == 0 else f'客户 {current_node}'
            ax1.scatter(float(current_pos[0]), float(current_pos[1]), 
                       c='orange', s=500, zorder=6, marker='*', 
                       edgecolors='darkorange', linewidths=2,
                       label=f'当前: {node_type}')
        
        # 计算当前指标
        current_dist, current_load, current_time = calculate_metrics(locs, actions, step)
        
        # 标题和图例
        ax1.set_title(f'{title}\n步骤: {step}/{len(actions)}', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.set_xlabel('X坐标', fontsize=12)
        ax1.set_ylabel('Y坐标', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_aspect('equal')
        
        # 添加统计信息
        info_text = (
            f'累计距离: {current_dist:.2f}\n'
            f'当前载重: {current_load:.2f}\n'
            f'当前时间: {current_time:.1f}'
        )
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 右侧：时间调度甘特图
        ax2 = plt.subplot(1, 2, 2)
        
        if time_windows is not None and step > 0:
            # 绘制时间窗
            for idx in range(len(customers)):
                earliest, latest = time_windows[idx]
                width = latest - earliest
                # 绘制时间窗矩形（浅色）
                rect = Rectangle((earliest, idx), width, 0.8, 
                               facecolor='lightgreen', edgecolor='black', 
                               alpha=0.3, linewidth=1)
                ax2.add_patch(rect)
            
            # 绘制已访问客户的实际到达时间
            temp_time = 0.0
            visited_customers = []
            for i in range(step):
                node = actions[i]
                if node > 0:  # 客户节点
                    customer_idx = node - 1
                    if customer_idx < len(customers):
                        visited_customers.append((customer_idx, temp_time))
                
                # 更新时间
                if i + 1 < len(actions):
                    next_node = actions[i + 1]
                    pos_a = locs[node]
                    pos_b = locs[next_node]
                    dist = np.sqrt(np.sum((pos_a - pos_b) ** 2))
                    temp_time += dist  # 假设速度=1
                    
                    if next_node > 0:  # 下一个是客户
                        customer_idx = next_node - 1
                        earliest = time_windows[customer_idx][0]
                        if temp_time < earliest:
                            temp_time = earliest  # 等待
                        temp_time += 10  # 服务时间
            
            # 绘制访问标记
            for customer_idx, arrival_time in visited_customers:
                ax2.scatter(arrival_time, customer_idx, c='red', s=100, 
                          zorder=5, marker='o', edgecolors='darkred', linewidths=2)
                ax2.text(arrival_time, customer_idx + 0.3, f'{arrival_time:.1f}',
                        fontsize=8, ha='center', color='red', fontweight='bold')
            
            ax2.set_xlabel('时间', fontsize=12)
            ax2.set_ylabel('客户ID', fontsize=12)
            ax2.set_title('时间调度图', fontsize=12, fontweight='bold')
            ax2.set_ylim(-0.5, len(customers) - 0.5)
            if time_windows is not None:
                max_time = np.max(time_windows[:, 1]) * 1.2
                ax2.set_xlim(0, max_time)
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.set_yticks(range(len(customers)))
            ax2.set_yticklabels([f'客户 {i+1}' for i in range(len(customers))])
        else:
            ax2.text(0.5, 0.5, '等待路线生成...', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=14, color='gray')
            ax2.axis('off')
        
        plt.tight_layout()
        
        # 保存到临时文件并读取为图像
        temp_path = save_path.replace('.gif', f'_frame_{step}.png')
        plt.savefig(temp_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 读取图像
        frames.append(Image.open(temp_path))
    
    # 保存为GIF
    if frames:
        duration = int(1000 / fps)  # 每帧持续时间（毫秒）
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
    
    # 清理临时文件
    for step in range(len(actions) + 1):
        temp_path = save_path.replace('.gif', f'_frame_{step}.png')
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    logger.info(f"VRPTW动画已保存: {save_path}")
    return save_path


def create_vrptw_comparison_plot(before_cost, after_cost, save_path, 
                                 title="VRPTW训练前后对比"):
    """
    创建VRPTW训练前后的对比图
    
    参数:
        before_cost: 训练前的总成本（距离+时间惩罚）
        after_cost: 训练后的总成本
        save_path: 图片保存路径
        title: 图表标题
    """
    improvement = ((before_cost - after_cost) / before_cost) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：成本对比
    categories = ['训练前', '训练后']
    costs = [before_cost, after_cost]
    colors = ['#ff6b6b', '#51cf66']
    
    bars = ax1.bar(categories, costs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('总成本（距离+时间惩罚）', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱状图上标注数值
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加改进百分比文本
    ax1.text(0.5, 0.95, f'改进: {improvement:.1f}%', 
            transform=ax1.transAxes, ha='center', va='top',
            fontsize=14, fontweight='bold', color='green' if improvement > 0 else 'red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 右图：改进百分比
    ax2.bar(['改进率'], [improvement], color='#339af0', alpha=0.8, 
           edgecolor='black', linewidth=2)
    ax2.set_ylabel('改进率 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('训练效果', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # 标注改进率数值
    ax2.text(0, improvement, f'{improvement:.1f}%',
            ha='center', va='bottom' if improvement > 0 else 'top',
            fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"VRPTW对比图已保存: {save_path}")
    return save_path


def create_vrptw_time_schedule(td, actions, save_path, title="VRPTW时间调度详情"):
    """
    创建详细的VRPTW时间调度甘特图
    
    参数:
        td: TensorDict，包含时间窗等信息
        actions: numpy数组，访问节点的顺序
        save_path: 图片保存路径
        title: 图表标题
    """
    # 提取信息
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs']).cpu().numpy()
        # 去掉batch维度（如果有）
        if locs.ndim == 3:
            locs = locs[0]
        
        time_windows = td.get('time_windows', None)
        if time_windows is not None:
            time_windows = time_windows.cpu().numpy()
            if time_windows.ndim == 3:
                time_windows = time_windows[0]
        else:
            # 生成默认时间窗
            num_customers = len(locs) - 1
            time_windows = np.random.rand(num_customers, 2) * 100
            time_windows[:, 1] = time_windows[:, 0] + 50
    else:
        locs = td['locs'].cpu().numpy()
        if locs.ndim == 3:
            locs = locs[0]
        
        time_windows = td.get('time_windows', None)
        if time_windows is not None:
            time_windows = time_windows.cpu().numpy()
            if time_windows.ndim == 3:
                time_windows = time_windows[0]
        else:
            num_customers = len(locs) - 1
            time_windows = np.random.rand(num_customers, 2) * 100
            time_windows[:, 1] = time_windows[:, 0] + 50
    
    num_customers = len(time_windows)
    
    # 计算实际访问时间
    current_time = 0.0
    service_time = 10.0
    speed = 1.0
    schedule = []  # [(customer_id, arrival_time, service_start, service_end, wait_time, is_late)]
    
    for i, node in enumerate(actions):
        if node == 0:  # 仓库
            continue
        
        customer_idx = node - 1
        if customer_idx >= num_customers:
            continue
        
        # 计算到达时间
        if i > 0:
            prev_node = actions[i - 1]
            dist = np.sqrt(np.sum((locs[node] - locs[prev_node]) ** 2))
            current_time += dist / speed
        
        arrival_time = current_time
        earliest, latest = time_windows[customer_idx]
        
        # 检查时间窗
        wait_time = 0.0
        is_late = False
        if current_time < earliest:
            wait_time = earliest - current_time
            current_time = earliest
        elif current_time > latest:
            is_late = True
        
        service_start = current_time
        service_end = current_time + service_time
        current_time = service_end
        
        schedule.append((node, arrival_time, service_start, service_end, 
                        wait_time, is_late, earliest, latest))
    
    # 绘制甘特图
    fig, ax = plt.subplots(figsize=(16, max(8, num_customers * 0.5)))
    
    # 绘制时间窗（背景）
    for customer_id in range(1, num_customers + 1):
        idx = customer_id - 1
        if idx < len(time_windows):
            earliest, latest = time_windows[idx]
            width = latest - earliest
            rect = Rectangle((earliest, customer_id - 0.4), width, 0.8,
                           facecolor='lightgreen', edgecolor='black',
                           alpha=0.2, linewidth=1, label='时间窗' if customer_id == 1 else '')
            ax.add_patch(rect)
    
    # 绘制实际服务时间
    colors_service = []
    for customer_id, arrival, start, end, wait, is_late, earliest, latest in schedule:
        # 等待时间（浅蓝色）
        if wait > 0:
            rect_wait = Rectangle((arrival, customer_id - 0.3), wait, 0.6,
                                 facecolor='lightblue', edgecolor='blue',
                                 alpha=0.5, linewidth=1)
            ax.add_patch(rect_wait)
            ax.text(arrival + wait/2, customer_id, '等待',
                   ha='center', va='center', fontsize=8, color='blue')
        
        # 服务时间
        color = 'red' if is_late else 'green'
        colors_service.append(color)
        rect_service = Rectangle((start, customer_id - 0.3), end - start, 0.6,
                                facecolor=color, edgecolor='darkred' if is_late else 'darkgreen',
                                alpha=0.7, linewidth=2,
                                label='延迟服务' if is_late and customer_id == schedule[0][0] else 
                                      ('正常服务' if not is_late and customer_id == schedule[0][0] else ''))
        ax.add_patch(rect_service)
        
        # 标注时间
        ax.text(start + (end - start)/2, customer_id, f'{start:.1f}',
               ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        
        # 如果延迟，标注超时量
        if is_late:
            overtime = start - latest
            ax.text(end + 2, customer_id, f'超时:{overtime:.1f}',
                   ha='left', va='center', fontsize=8, color='red', fontweight='bold')
    
    # 设置坐标轴
    ax.set_xlabel('时间', fontsize=12, fontweight='bold')
    ax.set_ylabel('客户编号', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, num_customers + 0.5)
    
    if time_windows is not None and len(time_windows) > 0:
        max_time = max(np.max(time_windows[:, 1]), max([end for _, _, _, end, _, _, _, _ in schedule]))
        ax.set_xlim(0, max_time * 1.1)
    
    ax.set_yticks(range(1, num_customers + 1))
    ax.set_yticklabels([f'客户 {i}' for i in range(1, num_customers + 1)])
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息
    total_wait = sum([wait for _, _, _, _, wait, _, _, _ in schedule])
    num_late = sum([1 for _, _, _, _, _, is_late, _, _ in schedule if is_late])
    total_time = schedule[-1][3] if schedule else 0
    
    stats_text = (
        f'统计信息:\n'
        f'总访问客户: {len(schedule)}\n'
        f'总等待时间: {total_wait:.1f}\n'
        f'延迟客户数: {num_late}\n'
        f'总完成时间: {total_time:.1f}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"VRPTW时间调度图已保存: {save_path}")
    return save_path



