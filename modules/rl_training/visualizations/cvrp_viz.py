"""
CVRP问题专用可视化函数
提供CVRP路线动画、对比图等可视化
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def create_cvrp_route_animation(td, actions, save_path, title="CVRP路线生成过程", fps=2):
    """
    创建CVRP路线逐步生成的动态GIF
    
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
        depot = locs[0]  # 仓库位置
        customers = locs[1:]  # 客户位置
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
                node_b = 0  # 返回仓库
            
            # 计算距离
            pos_a = locs[node_a]
            pos_b = locs[node_b]
            dist = np.sqrt(np.sum((pos_a - pos_b) ** 2))
            total_dist += dist
            
            # 更新载重（如果访问客户）
            if node_a > 0 and demands is not None:  # 不是仓库
                # demands数组索引：如果包含仓库则用node_a，否则用node_a-1
                demand_idx = node_a if len(demands) > node_a else node_a - 1
                if demand_idx < len(demands):
                    current_load += demands[demand_idx]
            
            # 如果返回仓库，重置载重
            if node_b == 0:
                current_load = 0.0
        
        return total_dist, current_load
    
    # 为每一步生成一帧图像
    for step in range(len(actions) + 1):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制仓库（特殊标记）
        ax.scatter(depot[0], depot[1], c='red', s=400, marker='s', 
                  zorder=5, edgecolors='darkred', linewidths=3, label='仓库')
        
        # 绘制所有客户点
        ax.scatter(customers[:, 0], customers[:, 1], c='lightblue', s=200, 
                  zorder=3, alpha=0.6, edgecolors='black', linewidths=2)
        
        # 标注节点编号和需求
        ax.text(depot[0], depot[1], '0\n仓库', fontsize=10, ha='center', va='center',
               fontweight='bold', color='white')
        
        for idx, (x, y) in enumerate(customers):
            customer_id = idx + 1  # 显示的客户编号（1, 2, 3, ...）
            demand_text = f'\n({demands[idx]:.2f})' if demands is not None else ''
            ax.text(x, y, f'{customer_id}{demand_text}', fontsize=9, ha='center', va='center',
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
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                       'b-' if linestyle == '-' else 'g--', 
                       linewidth=linewidth, alpha=0.7, zorder=1)
                
                # 添加箭头表示方向
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                          xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                          arrowprops=dict(arrowstyle='->', 
                                        color='blue' if linestyle == '-' else 'green', 
                                        lw=2, alpha=0.7))
        
        # 高亮当前访问的节点
        if step > 0 and step <= len(actions):
            current_node = actions[step - 1]
            current_pos = locs[current_node]
            node_type = '仓库' if current_node == 0 else f'客户 {current_node}'
            ax.scatter(current_pos[0], current_pos[1], 
                      c='orange', s=400, zorder=6, marker='*', 
                      edgecolors='darkorange', linewidths=2,
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
        
        ax.set_title(f"{title}\n{info_text}", fontsize=14, fontweight='bold', pad=20)
        
        # 设置坐标轴
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # 添加进度条
        progress = step / len(actions) if len(actions) > 0 else 0
        ax.text(0.5, -0.12, f"进度: {int(progress * 100)}%", 
               ha='center', va='top', transform=ax.transAxes,
               fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 保存当前帧为图像
        fig.tight_layout()
        
        # 将图形转换为PIL Image
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


def create_cvrp_comparison_plot(env, td, actions_untrained, rewards_untrained, 
                                actions_trained, rewards_trained, save_path, index=1):
    """
    创建CVRP路线对比图（训练前vs训练后）- 美化增强版
    
    参数:
        env: CVRP环境
        td: TensorDict
        actions_untrained: 未训练模型的动作序列
        rewards_untrained: 未训练模型的奖励
        actions_trained: 训练后模型的动作序列
        rewards_trained: 训练后模型的奖励
        save_path: 保存路径
        index: 图片索引（用于多实例）
    """
    # 创建更大的图形，使用渐变背景
    fig = plt.figure(figsize=(18, 8), facecolor='#f5f7fa')
    
    # 创建网格布局：2行3列，优化比例
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.45], 
                         height_ratios=[1.2, 1],
                         hspace=0.25, wspace=0.25,
                         left=0.05, right=0.98, top=0.92, bottom=0.05)
    
    # 左侧：未训练路线
    ax_untrained = fig.add_subplot(gs[:, 0], facecolor='#fafbfc')
    # 中间：训练后路线
    ax_trained = fig.add_subplot(gs[:, 1], facecolor='#fafbfc')
    # 右侧上：统计信息
    ax_stats = fig.add_subplot(gs[0, 2])
    # 右侧下：图例说明
    ax_legend = fig.add_subplot(gs[1, 2])
    
    # 渲染路线
    env.render(td, actions_untrained, ax=ax_untrained)
    env.render(td, actions_trained, ax=ax_trained)
    
    # 计算成本和改进
    cost_untrained = -rewards_untrained.item()
    cost_trained = -rewards_trained.item()
    improvement = ((cost_untrained - cost_trained) / cost_untrained) * 100
    
    # 计算返回仓库次数
    returns_untrained = (actions_untrained == 0).sum().item() - 1
    returns_trained = (actions_trained == 0).sum().item() - 1
    
    # 获取节点信息
    locs = td.get('locs', td['locs']).cpu().numpy()
    num_customers = len(locs) - 1
    
    # 设置美化后的标题 - 使用更清晰的文字（避免emoji）
    ax_untrained.set_title(
        f"[随机策略]\n成本: {cost_untrained:.3f} | 返回: {returns_untrained}次",
        fontsize=14, fontweight='bold', color='#c0392b', pad=20,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee', 
                 edgecolor='#e74c3c', linewidth=2.5, alpha=0.8)
    )
    ax_trained.set_title(
        f"[训练后策略]\n成本: {cost_trained:.3f} | 返回: {returns_trained}次",
        fontsize=14, fontweight='bold', color='#186a3b', pad=20,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#efe', 
                 edgecolor='#27ae60', linewidth=2.5, alpha=0.8)
    )
    
    # 添加渐变式边框效果
    for spine in ax_untrained.spines.values():
        spine.set_edgecolor('#e74c3c')
        spine.set_linewidth(4)
        spine.set_alpha(0.8)
    for spine in ax_trained.spines.values():
        spine.set_edgecolor('#27ae60')
        spine.set_linewidth(4)
        spine.set_alpha(0.8)
    
    # 添加网格美化
    ax_untrained.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, color='#95a5a6')
    ax_trained.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, color='#95a5a6')
    
    # ========== 右侧上：美化统计信息面板 ==========
    ax_stats.axis('off')
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    
    # 绘制卡片式背景
    from matplotlib.patches import FancyBboxPatch
    card = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                          boxstyle="round,pad=0.02",
                          facecolor='white',
                          edgecolor='#3498db',
                          linewidth=3,
                          transform=ax_stats.transAxes,
                          zorder=1,
                          alpha=0.95)
    ax_stats.add_patch(card)
    
    # 标题
    ax_stats.text(0.5, 0.92, '对比统计', 
                 transform=ax_stats.transAxes,
                 fontsize=13, fontweight='bold',
                 ha='center', va='top',
                 color='#2c3e50')
    
    # 统计内容 - 使用更清晰的排版
    stats_lines = [
        ('问题规模', '', 0.80),
        (f'  客户数量', f'{num_customers}', 0.74),
        (f'  总节点数', f'{len(locs)}', 0.68),
        ('', '', 0.62),  # 分隔线
        ('成本对比', '', 0.56),
        (f'  随机策略', f'{cost_untrained:.3f}', 0.50),
        (f'  训练策略', f'{cost_trained:.3f}', 0.44),
        ('', '', 0.38),  # 分隔线
        ('返回次数', '', 0.32),
        (f'  随机策略', f'{returns_untrained}次', 0.26),
        (f'  训练策略', f'{returns_trained}次', 0.20),
    ]
    
    for label, value, y_pos in stats_lines:
        if label == '':  # 分隔线
            ax_stats.plot([0.1, 0.9], [y_pos, y_pos], 
                         color='#bdc3c7', linewidth=1, alpha=0.5,
                         transform=ax_stats.transAxes, zorder=2)
        elif value == '':  # 标题行
            ax_stats.text(0.1, y_pos, label,
                         transform=ax_stats.transAxes,
                         fontsize=10, fontweight='bold',
                         color='#34495e', zorder=3)
        else:  # 数据行
            ax_stats.text(0.15, y_pos, label,
                         transform=ax_stats.transAxes,
                         fontsize=9, color='#7f8c8d', zorder=3)
            ax_stats.text(0.85, y_pos, value,
                         transform=ax_stats.transAxes,
                         fontsize=9, fontweight='bold',
                         color='#2c3e50', ha='right', zorder=3)
    
    # 改进指示器 - 大号显示
    if improvement > 20:
        improvement_color = '#27ae60'
        improvement_bg = '#d5f4e6'
        improvement_label = '优秀'
    elif improvement > 10:
        improvement_color = '#f39c12'
        improvement_bg = '#fef5e7'
        improvement_label = '良好'
    elif improvement > 0:
        improvement_color = '#3498db'
        improvement_bg = '#ebf5fb'
        improvement_label = '提升'
    else:
        improvement_color = '#e74c3c'
        improvement_bg = '#fadbd8'
        improvement_label = '需优化'
    
    # 改进百分比大号显示
    ax_stats.text(0.5, 0.08, f'{improvement:.1f}%',
                 transform=ax_stats.transAxes,
                 fontsize=24, fontweight='bold',
                 color=improvement_color,
                 ha='center', va='center', zorder=4)
    ax_stats.text(0.5, 0.02, f'性能{improvement_label}',
                 transform=ax_stats.transAxes,
                 fontsize=10,
                 color=improvement_color,
                 ha='center', va='center', zorder=4)
    
    # ========== 右侧下：美化图例说明 ==========
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    
    # 绘制卡片式背景
    card2 = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                           boxstyle="round,pad=0.02",
                           facecolor='#fffef0',
                           edgecolor='#f39c12',
                           linewidth=3,
                           transform=ax_legend.transAxes,
                           zorder=1,
                           alpha=0.95)
    ax_legend.add_patch(card2)
    
    # 标题
    ax_legend.text(0.5, 0.92, '图例说明', 
                  transform=ax_legend.transAxes,
                  fontsize=12, fontweight='bold',
                  ha='center', va='top',
                  color='#7d6608')
    
    # 图例内容
    legend_items = [
        ('节点类型', 0.78),
        ('  [■] 仓库（红色方块）', 0.70),
        ('  [●] 客户（蓝色圆点）', 0.62),
        ('', 0.56),  # 分隔
        ('路径类型', 0.48),
        ('  [━] 配送路线（蓝色）', 0.40),
        ('  [┈] 返回仓库（绿色虚线）', 0.32),
        ('', 0.26),  # 分隔
        ('成本说明', 0.18),
        ('  总行驶距离越小越好', 0.10),
    ]
    
    for text, y_pos in legend_items:
        if text == '':
            ax_legend.plot([0.1, 0.9], [y_pos, y_pos], 
                          color='#f39c12', linewidth=1, alpha=0.3,
                          transform=ax_legend.transAxes, zorder=2)
        elif text.startswith('  '):
            ax_legend.text(0.12, y_pos, text,
                          transform=ax_legend.transAxes,
                          fontsize=8.5, color='#7f8c8d', zorder=3)
        else:
            ax_legend.text(0.1, y_pos, text,
                          transform=ax_legend.transAxes,
                          fontsize=9.5, fontweight='bold',
                          color='#5d4037', zorder=3)
    
    # 添加美化的总标题
    fig.text(0.5, 0.97, f'CVRP车辆路径问题 - 路线对比分析（实例 #{index}）',
            fontsize=17, fontweight='bold', 
            ha='center', va='top',
            color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.8', 
                     facecolor='white', 
                     edgecolor='#3498db',
                     linewidth=2.5,
                     alpha=0.9))
    
    # 保存高质量图片
    plt.savefig(save_path, dpi=200, bbox_inches="tight", 
               facecolor='#f5f7fa', edgecolor='none')
    plt.close(fig)

