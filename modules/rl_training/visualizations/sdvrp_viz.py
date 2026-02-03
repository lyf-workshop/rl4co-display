"""
SDVRP问题专用可视化函数
提供SDVRP路线动画、对比图、分割配送分析等可视化
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def analyze_split_deliveries(actions, demands=None):
    """
    分析分割配送情况
    
    参数:
        actions: 动作序列（节点访问顺序）
        demands: 客户需求数组
    
    返回:
        dict: {
            'visit_count': {客户ID: 访问次数},
            'split_customers': [被分割配送的客户ID],
            'total_splits': 总分割次数,
            'split_percentage': 分割配送百分比
        }
    """
    visit_count = defaultdict(int)
    
    for action in actions:
        if action > 0:  # 不统计仓库
            visit_count[action] += 1
    
    # 找出被分割配送的客户
    split_customers = [cust_id for cust_id, count in visit_count.items() if count > 1]
    total_splits = sum(count - 1 for count in visit_count.values() if count > 1)
    
    num_customers = len([a for a in set(actions) if a > 0])
    split_percentage = (len(split_customers) / num_customers * 100) if num_customers > 0 else 0
    
    return {
        'visit_count': dict(visit_count),
        'split_customers': split_customers,
        'total_splits': total_splits,
        'split_percentage': split_percentage,
        'num_customers': num_customers,
    }


def create_sdvrp_route_animation(td, actions, save_path, title="SDVRP路线生成过程", fps=2):
    """
    创建SDVRP路线逐步生成的动态GIF（突出分割配送）
    
    参数:
        td: TensorDict，包含客户坐标、需求、容量等信息
        actions: numpy数组，访问节点的顺序
        save_path: GIF保存路径
        title: 图表标题
        fps: 帧率（每秒帧数）
    """
    # 提取坐标信息
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs']).cpu().numpy()
        depot = locs[0]
        customers = locs[1:]
        demands = td.get('demand', td.get('demands', None))
        if demands is not None:
            demands = demands.cpu().numpy()
    else:
        locs = td['locs'].cpu().numpy()
        depot = locs[0]
        customers = locs[1:]
        demands = td.get('demand', td.get('demands', None))
        if demands is not None:
            demands = demands.cpu().numpy()
    
    num_nodes = len(locs)
    frames = []
    
    # 分析分割配送
    split_info = analyze_split_deliveries(actions, demands)
    visit_count_per_step = defaultdict(int)
    
    # 计算每一步的累计距离和载重
    def calculate_metrics(locs, actions, step, capacity=1.0):
        """计算到第step步为止的累计距离和当前载重"""
        if step < 1:
            return 0.0, 0.0
        
        total_dist = 0.0
        current_load = 0.0
        
        for i in range(step):
            node_a = actions[i]
            
            if i + 1 < len(actions):
                node_b = actions[i + 1]
            else:
                node_b = 0
            
            # 计算距离
            pos_a = locs[node_a]
            pos_b = locs[node_b]
            dist = np.sqrt(np.sum((pos_a - pos_b) ** 2))
            total_dist += dist
            
            # 更新载重
            if node_a > 0 and demands is not None:
                demand_idx = node_a if len(demands) > node_a else node_a - 1
                if demand_idx < len(demands):
                    current_load += demands[demand_idx]
            
            # 如果返回仓库，重置载重
            if node_b == 0:
                current_load = 0.0
        
        return total_dist, current_load
    
    # 为每一步生成一帧图像
    for step in range(len(actions) + 1):
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # 绘制仓库（特殊标记）
        ax.scatter(depot[0], depot[1], c='red', s=500, marker='s', 
                  zorder=5, edgecolors='darkred', linewidths=3, label='仓库')
        
        # 绘制所有客户点（用颜色区分访问次数）
        for i, (x, y) in enumerate(customers, start=1):
            # 统计当前步骤下该客户被访问的次数
            visit_count = sum(1 for j in range(min(step, len(actions))) if actions[j] == i)
            
            # 根据访问次数选择颜色
            if visit_count == 0:
                color = 'lightgray'  # 未访问
                alpha = 0.4
                size = 150
            elif visit_count == 1:
                color = 'lightblue'  # 访问1次
                alpha = 0.7
                size = 200
            else:
                color = 'orange'  # 分割配送（访问多次）
                alpha = 0.9
                size = 250
            
            ax.scatter(x, y, c=color, s=size, 
                      zorder=3, alpha=alpha, edgecolors='black', linewidths=2)
        
        # 标注节点编号和需求
        ax.text(depot[0], depot[1], '0\n仓库', fontsize=11, ha='center', va='center',
               fontweight='bold', color='white')
        
        for i, (x, y) in enumerate(customers, start=1):
            customer_id = i
            visit_count = sum(1 for j in range(min(step, len(actions))) if actions[j] == i)
            
            # 显示需求和访问次数
            if demands is not None:
                demand_idx = i - 1
                if demand_idx < len(demands):
                    demand_val = demands[demand_idx]
                    if visit_count > 1:
                        label = f'{customer_id}\n需求:{demand_val:.2f}\n访问:{visit_count}次'
                    else:
                        label = f'{customer_id}\n({demand_val:.2f})'
                else:
                    label = f'{customer_id}'
            else:
                if visit_count > 1:
                    label = f'{customer_id}\n({visit_count}次)'
                else:
                    label = f'{customer_id}'
            
            # 分割配送的客户用红色文字
            text_color = 'darkred' if visit_count > 1 else 'darkblue'
            ax.text(x, y, label, fontsize=8, ha='center', va='center',
                   fontweight='bold', color=text_color)
        
        # 绘制已经构建的路径
        if step > 0:
            for i in range(step):
                start_idx = actions[i]
                if i + 1 < len(actions):
                    end_idx = actions[i + 1]
                else:
                    end_idx = 0
                
                start_pos = locs[start_idx]
                end_pos = locs[end_idx]
                
                # 如果是返回仓库，使用绿色虚线
                if end_idx == 0:
                    linestyle = '--'
                    color = 'green'
                    linewidth = 2
                # 如果是重复访问同一客户（分割配送），使用橙色线
                elif start_idx > 0 and sum(1 for j in range(i) if actions[j] == end_idx) > 0:
                    linestyle = '-'
                    color = 'orange'
                    linewidth = 4
                else:
                    linestyle = '-'
                    color = 'blue'
                    linewidth = 3
                
                # 绘制路径线
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                       color=color, linestyle=linestyle, linewidth=linewidth, 
                       alpha=0.7, zorder=1)
                
                # 添加箭头
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                          xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                          arrowprops=dict(arrowstyle='->', color=color, 
                                        lw=2, alpha=0.7))
        
        # 高亮当前访问的节点
        if step > 0 and step <= len(actions):
            current_node = actions[step - 1]
            current_pos = locs[current_node]
            node_type = '仓库' if current_node == 0 else f'客户 {current_node}'
            ax.scatter(current_pos[0], current_pos[1], 
                      c='yellow', s=500, zorder=6, marker='*', 
                      edgecolors='darkorange', linewidths=3,
                      label=f'当前: {node_type}')
        
        # 计算当前指标
        current_dist, current_load = calculate_metrics(locs, actions, step)
        
        # 设置标题和信息
        if step == 0:
            info_text = "开始配送..."
        elif step < len(actions):
            info_text = f"第 {step} 步 | 累计距离: {current_dist:.3f} | 当前载重: {current_load:.2f}"
        else:
            info_text = f"完成！总距离: {current_dist:.3f}"
        
        # 添加分割配送统计
        if split_info['total_splits'] > 0:
            split_text = f" | 分割配送: {split_info['total_splits']}次 ({split_info['split_percentage']:.1f}%)"
            info_text += split_text
        
        ax.set_title(f"{title}\n{info_text}", fontsize=13, fontweight='bold', pad=20)
        
        # 设置坐标轴
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X 坐标', fontsize=11)
        ax.set_ylabel('Y 坐标', fontsize=11)
        
        # 添加图例
        # 创建自定义图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='black', label='访问1次'),
            Patch(facecolor='orange', edgecolor='black', label='分割配送(≥2次)'),
            plt.Line2D([0], [0], color='blue', linewidth=3, label='配送路径'),
            plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='返回仓库'),
            plt.Line2D([0], [0], color='orange', linewidth=4, label='重复访问'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
        
        # 保存当前帧
        fig.tight_layout()
        fig.canvas.draw()
        
        try:
            buf = fig.canvas.buffer_rgba()
            image = np.frombuffer(buf, dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:, :, :3]
        except AttributeError:
            try:
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                image = buf[:, :, 1:]
        
        frames.append(Image.fromarray(image))
        plt.close(fig)
    
    # 在最后一帧停留更长时间
    for _ in range(3):
        frames.append(frames[-1])
    
    # 保存为GIF
    duration = int(1000 / fps)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False
    )


def create_sdvrp_comparison_plot(env, td, actions_untrained, rewards_untrained, 
                                 actions_trained, rewards_trained, save_path, index=1):
    """
    创建SDVRP路线对比图（训练前vs训练后）- 增强版
    
    参数:
        env: SDVRP环境
        td: TensorDict
        actions_untrained: 未训练模型的动作序列
        rewards_untrained: 未训练模型的奖励
        actions_trained: 训练后模型的动作序列
        rewards_trained: 训练后模型的奖励
        save_path: 保存路径
        index: 图片索引
    """
    # 创建更大的图形
    fig = plt.figure(figsize=(16, 7))
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.3)
    
    # 左侧：未训练路线
    ax_untrained = fig.add_subplot(gs[:, 0])
    # 中间：训练后路线
    ax_trained = fig.add_subplot(gs[:, 1])
    # 右侧上：统计信息
    ax_stats = fig.add_subplot(gs[0, 2])
    # 右侧下：分割配送分析
    ax_split = fig.add_subplot(gs[1, 2])
    
    # 渲染路线
    env.render(td, actions_untrained, ax=ax_untrained)
    env.render(td, actions_trained, ax=ax_trained)
    
    # 计算成本和改进
    cost_untrained = -rewards_untrained.item()
    cost_trained = -rewards_trained.item()
    improvement = ((cost_untrained - cost_trained) / cost_untrained) * 100
    
    # 分析分割配送
    split_untrained = analyze_split_deliveries(actions_untrained.cpu().numpy())
    split_trained = analyze_split_deliveries(actions_trained.cpu().numpy())
    
    # 计算返回次数
    returns_untrained = (actions_untrained == 0).sum().item() - 1
    returns_trained = (actions_trained == 0).sum().item() - 1
    
    # 获取节点信息
    locs = td.get('locs', td['locs']).cpu().numpy()
    num_customers = len(locs) - 1
    
    # 设置标题
    ax_untrained.set_title(
        f"🎲 随机策略\n成本: {cost_untrained:.3f} | 返回: {returns_untrained}次\n分割: {split_untrained['total_splits']}次",
        fontsize=12, fontweight='bold', color='#e74c3c', pad=15
    )
    ax_trained.set_title(
        f"🎯 训练后策略\n成本: {cost_trained:.3f} | 返回: {returns_trained}次\n分割: {split_trained['total_splits']}次",
        fontsize=12, fontweight='bold', color='#27ae60', pad=15
    )
    
    # 添加边框颜色
    for spine in ax_untrained.spines.values():
        spine.set_edgecolor('#e74c3c')
        spine.set_linewidth(3)
    for spine in ax_trained.spines.values():
        spine.set_edgecolor('#27ae60')
        spine.set_linewidth(3)
    
    # ========== 右侧上：统计信息面板 ==========
    ax_stats.axis('off')
    
    stats_text = f"""
    📊 对比统计 (SDVRP)
    
    问题规模:
    • 客户数: {num_customers}
    • 总节点: {len(locs)}
    
    成本对比:
    • 随机: {cost_untrained:.3f}
    • 训练: {cost_trained:.3f}
    • 改进: {improvement:.1f}%
    
    返回次数:
    • 随机: {returns_untrained}次
    • 训练: {returns_trained}次
    
    分割配送:
    • 随机: {split_untrained['total_splits']}次
    • 训练: {split_trained['total_splits']}次
    """
    
    # 选择改进指示颜色
    if improvement > 20:
        improvement_color = '#27ae60'
        improvement_emoji = '🎉'
    elif improvement > 10:
        improvement_color = '#f39c12'
        improvement_emoji = '👍'
    else:
        improvement_color = '#e74c3c'
        improvement_emoji = '⚠️'
    
    ax_stats.text(0.05, 0.95, stats_text, 
                 transform=ax_stats.transAxes,
                 fontsize=9, 
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.8', 
                          facecolor='#f8f9fa', 
                          edgecolor='#dee2e6',
                          linewidth=2))
    
    # 改进指示器
    improvement_text = f"{improvement_emoji} {improvement:.1f}%\n性能提升"
    ax_stats.text(0.5, 0.12, improvement_text,
                 transform=ax_stats.transAxes,
                 fontsize=13,
                 fontweight='bold',
                 color=improvement_color,
                 ha='center',
                 va='center',
                 bbox=dict(boxstyle='round,pad=0.6',
                          facecolor='white',
                          edgecolor=improvement_color,
                          linewidth=3))
    
    # ========== 右侧下：分割配送分析 ==========
    ax_split.axis('off')
    
    split_text = f"""
    🔄 分割配送分析
    
    随机策略:
    • 分割客户: {len(split_untrained['split_customers'])}个
    • 分割次数: {split_untrained['total_splits']}次
    • 分割比例: {split_untrained['split_percentage']:.1f}%
    
    训练后策略:
    • 分割客户: {len(split_trained['split_customers'])}个
    • 分割次数: {split_trained['total_splits']}次
    • 分割比例: {split_trained['split_percentage']:.1f}%
    
    💡 SDVRP特点:
    • 允许重复访问客户
    • 分批配送大宗需求
    • 减少返仓次数
    """
    
    ax_split.text(0.05, 0.95, split_text,
                 transform=ax_split.transAxes,
                 fontsize=8,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.8',
                          facecolor='#fff3cd',
                          edgecolor='#ffc107',
                          linewidth=2))
    
    # 添加总标题
    fig.suptitle(f'🚚 SDVRP路线对比分析 (分割配送) - 实例 #{index}',
                fontsize=15, fontweight='bold', y=0.98)
    
    # 保存
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close(fig)


def create_sdvrp_split_analysis(actions, demands, save_path, title="SDVRP分割配送分析"):
    """
    创建SDVRP分割配送详细分析图
    
    参数:
        actions: 动作序列
        demands: 客户需求
        save_path: 保存路径
        title: 标题
    """
    split_info = analyze_split_deliveries(actions, demands)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========== 左图：分割配送统计 ==========
    visit_counts = list(split_info['visit_count'].values())
    unique_counts = sorted(set(visit_counts))
    count_freq = [visit_counts.count(c) for c in unique_counts]
    
    colors = ['lightblue' if c == 1 else 'orange' for c in unique_counts]
    ax1.bar(unique_counts, count_freq, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('访问次数', fontsize=12)
    ax1.set_ylabel('客户数量', fontsize=12)
    ax1.set_title('客户访问次数分布', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (count, freq) in enumerate(zip(unique_counts, count_freq)):
        ax1.text(count, freq + 0.5, str(freq), ha='center', va='bottom', fontweight='bold')
    
    # ========== 右图：分割配送客户 ==========
    if split_info['split_customers']:
        split_customers = split_info['split_customers']
        visit_counts_split = [split_info['visit_count'][c] for c in split_customers]
        
        # 按访问次数排序
        sorted_pairs = sorted(zip(split_customers, visit_counts_split), key=lambda x: x[1], reverse=True)
        customers, counts = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        # 只显示前15个（如果太多）
        if len(customers) > 15:
            customers = customers[:15]
            counts = counts[:15]
            truncated = True
        else:
            truncated = False
        
        ax2.barh(range(len(customers)), counts, color='orange', edgecolor='black', linewidth=1.5)
        ax2.set_yticks(range(len(customers)))
        ax2.set_yticklabels([f'客户 {c}' for c in customers])
        ax2.set_xlabel('访问次数', fontsize=12)
        ax2.set_title('分割配送客户详情' + (' (前15)' if truncated else ''), 
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, count in enumerate(counts):
            ax2.text(count + 0.1, i, str(count), va='center', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, '无分割配送\n所有客户仅访问一次', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax2.axis('off')
    
    # 总标题
    fig.suptitle(f"{title}\n总分割次数: {split_info['total_splits']} | "
                f"分割客户: {len(split_info['split_customers'])}个 ({split_info['split_percentage']:.1f}%)",
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


