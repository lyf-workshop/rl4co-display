"""
ATSP问题专用可视化函数
提供费用矩阵热力图、训练前后对比图，以及路径构建动态GIF
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
import logging

logger = logging.getLogger('rl4co_display')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 内部辅助函数
# ============================================================

def _compute_tour_cost(cost_matrix, actions):
    """计算给定路径的总费用"""
    n = len(actions)
    total = 0.0
    for t in range(n):
        src = int(actions[t])
        dst = int(actions[(t + 1) % n])
        total += float(cost_matrix[src, dst])
    return total


def _fig_to_pil(fig):
    """将 matplotlib Figure 转换为 PIL Image（兼容多个版本）"""
    fig.canvas.draw()
    try:
        buf = fig.canvas.buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(image[:, :, :3])
    except AttributeError:
        pass
    try:
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return Image.fromarray(image)
    except AttributeError:
        pass
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return Image.fromarray(buf[:, :, 1:])  # ARGB → RGB


def _draw_heatmap_with_path(ax, cost_matrix, actions, path_color, vmin, vmax):
    """在指定轴上绘制热力图并高亮路径"""
    n = len(cost_matrix)
    im = ax.imshow(cost_matrix, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)

    for t in range(len(actions)):
        src = int(actions[t])
        dst = int(actions[(t + 1) % len(actions)])
        rect = plt.Rectangle(
            (dst - 0.5, src - 0.5), 1, 1,
            fill=False, edgecolor=path_color, linewidth=2.5, zorder=3
        )
        ax.add_patch(rect)

    start = int(actions[0])
    ax.add_patch(plt.Rectangle(
        (start - 0.5, start - 0.5), 1, 1,
        fill=True, facecolor='lime', alpha=0.5,
        edgecolor='green', linewidth=3, zorder=4
    ))

    tick_step = max(1, n // 10)
    ticks = list(range(0, n, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel('目标节点', fontsize=10)
    ax.set_ylabel('源节点', fontsize=10)
    return im


# ============================================================
# 公共接口 1：对比静态图
# ============================================================

def create_atsp_comparison_plot(cost_matrix, actions_random, actions_trained,
                                save_path, title="ATSP训练前后对比"):
    """
    创建ATSP训练前后的费用矩阵热力图对比图（静态PNG）

    参数:
        cost_matrix:     [N, N] 费用矩阵（numpy 或 torch.Tensor）
        actions_random:  [N]    随机策略的访问顺序
        actions_trained: [N]    训练后贪心解码的访问顺序
        save_path:       图片保存路径
        title:           图表标题

    返回:
        dict: {'cost_random', 'cost_trained', 'improvement'}
    """
    if isinstance(cost_matrix, torch.Tensor):
        cost_matrix = cost_matrix.cpu().numpy()
    if isinstance(actions_random, torch.Tensor):
        actions_random = actions_random.cpu().numpy()
    if isinstance(actions_trained, torch.Tensor):
        actions_trained = actions_trained.cpu().numpy()

    cost_random  = _compute_tour_cost(cost_matrix, actions_random)
    cost_trained = _compute_tour_cost(cost_matrix, actions_trained)
    improvement  = (cost_random - cost_trained) / cost_random * 100 if cost_random > 0 else 0.0

    n = len(cost_matrix)
    fig_w = max(14, n * 0.5)
    fig_h = max(7,  n * 0.35)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    vmin, vmax = float(cost_matrix.min()), float(cost_matrix.max())

    configs = [
        (axes[0], actions_random,  '训练前（随机采样）', 'red'),
        (axes[1], actions_trained, '训练后（贪心解码）', 'blue'),
    ]

    for ax, actions, label, color in configs:
        im = _draw_heatmap_with_path(ax, cost_matrix, actions, color, vmin, vmax)
        cost = cost_random if label.startswith('训练前') else cost_trained
        title_color = 'red' if label.startswith('训练前') else 'green'
        ax.set_title(f'{label}\n总费用: {cost:.4f}',
                     fontsize=12, fontweight='bold', color=title_color)

    plt.colorbar(im, ax=axes[1], label='边费用', shrink=0.8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lime', edgecolor='green', label='起始/终止节点'),
        Patch(facecolor='none', edgecolor='red',  linewidth=2, label='路径边（训练前）'),
        Patch(facecolor='none', edgecolor='blue', linewidth=2, label='路径边（训练后）'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        f"{title}\n改进幅度: {improvement:.2f}%  （{cost_random:.4f} → {cost_trained:.4f}）",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'cost_random':  cost_random,
        'cost_trained': cost_trained,
        'improvement':  improvement,
    }


# ============================================================
# 公共接口 2：路径构建动态GIF
# ============================================================

def create_atsp_route_animation(cost_matrix, actions, save_path,
                                title="ATSP路径构建过程", fps=2):
    """
    创建ATSP路径逐步构建的动态GIF

    由于ATSP无2D坐标，采用双面板设计：
      - 左图：圆形布局节点图，带方向箭头、颜色编码边费用
      - 右图：费用矩阵热力图，实时高亮当前选择的边

    参数:
        cost_matrix: [N, N] 费用矩阵（numpy 或 torch.Tensor）
        actions:     [N]    节点访问顺序
        save_path:   GIF 保存路径
        title:       图表标题
        fps:         帧率（每秒帧数）
    """
    if isinstance(cost_matrix, torch.Tensor):
        cost_matrix = cost_matrix.cpu().numpy()
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()

    n = len(actions)
    vmin, vmax = float(cost_matrix.min()), float(cost_matrix.max())

    # ── 圆形布局坐标（从顶部12点钟方向顺时针排列）──────────────────
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    node_x = np.cos(angles)
    node_y = np.sin(angles)

    # ── 根据节点数量自适应尺寸参数 ────────────────────────────────────
    node_s      = max(80,  1200 // n)       # 节点散点大小
    node_s_cur  = node_s * 2               # 当前节点放大
    arrow_lw    = max(1.0, 2.5 - n / 40)  # 箭头线宽
    label_fs    = max(5,   11  - n // 10)  # 节点标签字号（节点多时缩小）
    show_labels = (n <= 40)               # 节点超过40个时关闭标签

    # ── 节点偏移半径（箭头从节点边缘出发）────────────────────────────
    node_r = 0.07

    frames = []

    # n+1 步：步骤 0 = 仅显示节点，步骤 1~n = 逐条加边（最后一条回到起点）
    for step in range(n + 1):

        fig, (ax_g, ax_h) = plt.subplots(1, 2, figsize=(14, 7))

        # ────────────────────────────────────────────────
        # 左图：圆形布局图
        # ────────────────────────────────────────────────
        ax_g.set_aspect('equal')
        ax_g.set_xlim(-1.45, 1.45)
        ax_g.set_ylim(-1.45, 1.45)
        ax_g.axis('off')

        # 绘制所有边的"底色"（极浅灰色，帮助感知整体结构）
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                alpha = 0.04
                ax_g.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]],
                          color='gray', lw=0.3, alpha=alpha, zorder=0)

        # 绘制已完成的路径边（带颜色 + 方向箭头）
        for s in range(step):
            src = int(actions[s])
            dst = int(actions[(s + 1) % n])
            cost = float(cost_matrix[src, dst])
            c_norm = (cost - vmin) / (vmax - vmin + 1e-8)
            # 绿色=低费用，红色=高费用
            edge_color = plt.cm.RdYlGn_r(c_norm)

            x1, y1 = node_x[src], node_y[src]
            x2, y2 = node_x[dst], node_y[dst]
            dx, dy = x2 - x1, y2 - y1
            dist = np.hypot(dx, dy) + 1e-8

            ax_g.annotate(
                '',
                xy=(x2 - dx / dist * node_r, y2 - dy / dist * node_r),
                xytext=(x1 + dx / dist * node_r, y1 + dy / dist * node_r),
                arrowprops=dict(
                    arrowstyle='->', color=edge_color,
                    lw=arrow_lw, mutation_scale=14
                ),
                zorder=3
            )

        # 绘制节点
        start_idx = int(actions[0])
        cur_idx   = int(actions[step - 1]) if step > 0 else -1

        for i in range(n):
            if i == start_idx:
                color, marker, size = '#2ecc71', 's', node_s * 1.8
            elif i == cur_idx and step > 0:
                color, marker, size = '#e74c3c', '*', node_s_cur
            else:
                color, marker, size = '#aed6f1', 'o', node_s

            ax_g.scatter(node_x[i], node_y[i], c=color, s=size,
                         marker=marker, zorder=5,
                         edgecolors='#2c3e50', linewidths=1.2)

            if show_labels:
                lx = node_x[i] * 1.18
                ly = node_y[i] * 1.18
                ax_g.text(lx, ly, str(i), ha='center', va='center',
                          fontsize=label_fs, fontweight='bold', color='#1a252f')

        # 节点图图例
        from matplotlib.lines import Line2D
        legend_g = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
                   markersize=9, label=f'起点 (节点 {start_idx})'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#e74c3c',
                   markersize=11, label='当前节点'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#aed6f1',
                   markersize=8, label='未访问节点'),
        ]
        ax_g.legend(handles=legend_g, loc='lower left', fontsize=8,
                    framealpha=0.85, bbox_to_anchor=(-0.02, -0.02))

        ax_g.set_title('路径构建（圆形布局）\n颜色越红=边费用越高',
                       fontsize=11, fontweight='bold')

        # ────────────────────────────────────────────────
        # 右图：费用矩阵热力图
        # ────────────────────────────────────────────────
        im = ax_h.imshow(cost_matrix, cmap='YlOrRd', aspect='auto',
                         vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax_h, label='边费用', shrink=0.82)

        # 已选边：蓝色框
        for s in range(step):
            src = int(actions[s])
            dst = int(actions[(s + 1) % n])
            ax_h.add_patch(plt.Rectangle(
                (dst - 0.5, src - 0.5), 1, 1,
                fill=False, edgecolor='#2980b9', linewidth=1.8, zorder=3
            ))

        # 最新添加的边：红色高亮填充
        if step > 0:
            src = int(actions[step - 1])
            dst = int(actions[step % n])
            ax_h.add_patch(plt.Rectangle(
                (dst - 0.5, src - 0.5), 1, 1,
                fill=True, facecolor='#e74c3c', alpha=0.45,
                edgecolor='#c0392b', linewidth=2.5, zorder=4
            ))

        tick_step = max(1, n // 10)
        ticks = list(range(0, n, tick_step))
        ax_h.set_xticks(ticks)
        ax_h.set_yticks(ticks)
        ax_h.set_xlabel('目标节点', fontsize=10)
        ax_h.set_ylabel('源节点', fontsize=10)
        ax_h.set_title('费用矩阵热力图\n蓝框=已选边，红框=当前边',
                       fontsize=11, fontweight='bold')

        # ────────────────────────────────────────────────
        # 总标题 & 进度信息
        # ────────────────────────────────────────────────
        current_cost = sum(
            float(cost_matrix[int(actions[s]), int(actions[(s + 1) % n])])
            for s in range(step)
        )

        if step == 0:
            info = f"准备构建路径 | 共 {n} 个节点"
        elif step < n:
            src_node = int(actions[step - 1])
            dst_node = int(actions[step])
            edge_c   = float(cost_matrix[src_node, dst_node])
            info = (f"第 {step} 步: {src_node} → {dst_node}  "
                    f"(边费用: {edge_c:.3f}) | 累计费用: {current_cost:.3f}")
        else:
            info = f"路径完成！总费用: {current_cost:.4f} | 所有 {n} 个节点已访问"

        fig.suptitle(f"{title}\n{info}", fontsize=12, fontweight='bold')

        progress_pct = int(step / n * 100)
        fig.text(0.5, 0.01,
                 f"构建进度: {step}/{n}  ({progress_pct}%)",
                 ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='#fef9e7', alpha=0.8))

        plt.tight_layout(rect=[0, 0.04, 1, 0.95])

        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    # 最后一帧多停留3帧
    for _ in range(3):
        frames.append(frames[-1])

    duration = int(1000 / fps)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )

    logger.info(f"ATSP动态GIF已保存: {save_path}（{len(frames)}帧，fps={fps}）")
