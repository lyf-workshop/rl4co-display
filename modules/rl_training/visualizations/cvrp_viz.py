"""
CVRP问题专用可视化函数
提供CVRP路线动画、对比图等可视化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from .common import (
    DEFAULT_MAX_ANIMATION_FRAMES,
    build_closed_depot_route,
    figure_to_palette_image,
    save_palette_gif,
    select_animation_steps,
)

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei',
    'Microsoft YaHei',
    'Noto Sans CJK SC',
    'Noto Sans CJK JP',
    'WenQuanYi Zen Hei',
    'Arial Unicode MS',
    'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _normalise_node_demands(demands, num_nodes):
    if demands is None:
        return np.zeros(num_nodes, dtype=float)

    values = _to_numpy(demands).reshape(-1).astype(float)
    if values.size == num_nodes:
        return values
    if values.size == num_nodes - 1:
        return np.concatenate(([0.0], values))
    raise ValueError(
        f"CVRP demand length {values.size} does not match "
        f"{num_nodes - 1} customers"
    )


def _prepare_cvrp_route(locs, actions, demands=None):
    """Build a closed route and precompute distance/load at every action step."""
    locs = _to_numpy(locs).squeeze()
    if locs.ndim != 2 or locs.shape[1] != 2 or len(locs) < 2:
        raise ValueError("CVRP locations must have shape [num_nodes, 2]")

    route = build_closed_depot_route(
        _to_numpy(actions),
        num_nodes=len(locs),
        depot_index=0,
    )
    if len(route) < 2:
        raise ValueError("CVRP route must visit at least one customer")

    node_demands = _normalise_node_demands(demands, len(locs))
    segments = np.stack((locs[route[:-1]], locs[route[1:]]), axis=1)
    segment_distances = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    cumulative_distances = np.concatenate(([0.0], np.cumsum(segment_distances)))

    loads = np.zeros(len(route), dtype=float)
    current_load = 0.0
    for step, destination in enumerate(route[1:], start=1):
        if destination == 0:
            current_load = 0.0
        else:
            current_load += node_demands[destination]
        loads[step] = current_load

    return locs, route, node_demands, segments, cumulative_distances, loads


def create_cvrp_route_animation(
    td,
    actions,
    save_path,
    title="CVRP路线生成过程",
    fps=2,
    max_frames=DEFAULT_MAX_ANIMATION_FRAMES,
    progress_callback=None,
):
    """Create a bounded-memory CVRP route animation."""
    locs_value = td.get("locs") if hasattr(td, "get") else td["locs"]
    demands = None
    if hasattr(td, "get"):
        demands = td.get("demand", td.get("demands", None))
    elif "demand" in td:
        demands = td["demand"]
    elif "demands" in td:
        demands = td["demands"]

    locs, route, node_demands, segments, distances, loads = _prepare_cvrp_route(
        locs_value,
        actions,
        demands,
    )
    customers = locs[1:]
    route_steps = len(route) - 1
    frame_steps = select_animation_steps(route_steps, max_frames=max_frames)
    num_customers = len(customers)

    marker_size = max(8, min(90, 1600 / max(num_customers, 1)))
    line_width = max(0.7, min(2.5, 16 / np.sqrt(max(num_customers, 1))))
    label_nodes = num_customers <= 100

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    frames = []
    try:
        ax.scatter(
            customers[:, 0],
            customers[:, 1],
            c="#90caf9",
            s=marker_size,
            alpha=0.75,
            edgecolors="#1f2937" if num_customers <= 200 else "none",
            linewidths=0.5,
            zorder=3,
            label="客户",
        )
        ax.scatter(
            locs[0, 0],
            locs[0, 1],
            c="#d32f2f",
            s=max(90, marker_size * 2.2),
            marker="s",
            edgecolors="#7f1d1d",
            linewidths=1.5,
            zorder=5,
            label="仓库",
        )
        if label_nodes:
            ax.text(
                locs[0, 0],
                locs[0, 1],
                "0",
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=6,
            )
            for node, (x_coord, y_coord) in enumerate(customers, start=1):
                ax.text(
                    x_coord,
                    y_coord,
                    f"{node}\n({node_demands[node]:.2f})",
                    fontsize=6,
                    color="#0f3b66",
                    ha="center",
                    va="center",
                    zorder=4,
                )

        route_lines = LineCollection(
            [],
            linewidths=line_width,
            alpha=0.8,
            zorder=2,
            label="配送路线",
        )
        ax.add_collection(route_lines)
        current_marker = ax.scatter(
            [locs[0, 0]],
            [locs[0, 1]],
            c="#ff9800",
            s=max(110, marker_size * 2.6),
            marker="*",
            edgecolors="#9a4d00",
            linewidths=1,
            zorder=7,
            label="当前位置",
        )
        direction_arrow = FancyArrowPatch(
            locs[0],
            locs[0],
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=max(1.0, line_width),
            color="#1565c0",
            alpha=0.9,
            zorder=6,
        )
        direction_arrow.set_visible(False)
        ax.add_patch(direction_arrow)

        minimum = locs.min(axis=0)
        maximum = locs.max(axis=0)
        span = np.maximum(maximum - minimum, 1e-3)
        margin = np.maximum(span * 0.06, 0.02)
        ax.set_xlim(minimum[0] - margin[0], maximum[0] + margin[0])
        ax.set_ylim(minimum[1] - margin[1], maximum[1] + margin[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        title_artist = ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        progress_artist = ax.text(
            0.5,
            -0.10,
            "",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )
        fig.tight_layout()

        notify_stride = max(1, len(frame_steps) // 10)
        for frame_index, step in enumerate(frame_steps):
            route_lines.set_segments(segments[:step])
            destinations = route[1:step + 1]
            route_lines.set_colors(
                ["#2e7d32" if node == 0 else "#1565c0" for node in destinations]
            )
            route_lines.set_linestyles(
                ["dashed" if node == 0 else "solid" for node in destinations]
            )

            current_position = locs[route[step]]
            current_marker.set_offsets(current_position.reshape(1, 2))
            if step:
                direction_arrow.set_positions(segments[step - 1, 0], segments[step - 1, 1])
                direction_arrow.set_color(
                    "#2e7d32" if route[step] == 0 else "#1565c0"
                )
                direction_arrow.set_visible(True)
            else:
                direction_arrow.set_visible(False)

            percent = int(round(step * 100 / route_steps))
            if step == 0:
                status = "开始配送"
            elif step == route_steps:
                status = f"配送完成 | 总距离: {distances[step]:.3f}"
            else:
                status = (
                    f"第 {step}/{route_steps} 步 | "
                    f"累计距离: {distances[step]:.3f} | "
                    f"当前载重: {loads[step]:.2f}"
                )
            title_artist.set_text(f"{title}\n{status}")
            progress_artist.set_text(f"进度: {percent}%")
            frames.append(figure_to_palette_image(fig))

            if progress_callback and (
                frame_index == 0
                or frame_index == len(frame_steps) - 1
                or frame_index % notify_stride == 0
            ):
                progress_callback(percent, int(step), route_steps)
    except Exception:
        for frame in frames:
            frame.close()
        raise
    finally:
        plt.close(fig)

    save_palette_gif(frames, save_path, fps)
    return save_path


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

