"""
SPCTSP (Stochastic Prize Collecting TSP) 可视化函数

在 PCTSP 可视化基础上的新增内容：
  - create_spctsp_comparison_plot():
      左图：路线图（用 deterministic_prize 控制节点大小，真实奖励用颜色注释）
      右图：随机奖励对比柱形图（每个访问节点的期望奖励 vs 真实奖励）
  - create_spctsp_route_animation():
      与 PCTSP 动画相同，但标题额外显示「期望奖励总和 vs 真实奖励总和」

视觉约定：
  - 节点大小 ↔ deterministic_prize（期望奖励，决策时已知）
  - 已访问节点颜色 ↔ real_prize（真实奖励，访问后揭晓）
  - 节点边框深浅 ↔ penalty 值
"""

import os
import uuid
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional


# ──────────────────────────────────────────────
# 内部辅助函数（复用 pctsp_viz 相同逻辑）
# ──────────────────────────────────────────────

def _normalize(arr: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return lo + (hi - lo) * (arr - mn) / (mx - mn)
    return np.full_like(arr, (lo + hi) / 2.0)


def _parse_actions(actions: np.ndarray) -> list:
    visited = []
    for a in actions:
        a = int(a)
        if a == 0:
            break
        visited.append(a - 1)
    return visited


def _calc_tour_length(depot: np.ndarray, locs: np.ndarray, visited_idx: list) -> float:
    if not visited_idx:
        return 0.0
    pts = [depot] + [locs[i] for i in visited_idx] + [depot]
    return float(sum(np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i])) for i in range(len(pts)-1)))


# ──────────────────────────────────────────────
# 对比图（含随机奖励对比子图）
# ──────────────────────────────────────────────

def create_spctsp_comparison_plot(
    depot: np.ndarray,
    locs: np.ndarray,
    det_prizes: np.ndarray,
    real_prizes: np.ndarray,
    penalties: np.ndarray,
    actions: np.ndarray,
    model_reward: float,
    save_dir: str,
    instance_id: int = 1,
) -> Optional[str]:
    """
    生成 SPCTSP 路线对比图（静态 PNG，2 列布局）

    左图: 路线图
      - 节点大小 = deterministic_prize（决策时已知）
      - 已访问节点颜色 = real_prize（访问后揭晓），越红越高
      - 未访问节点：灰色半透明
      - 注释显示「期望/真实」两个值

    右图: 随机奖励对比柱形图
      - 仅展示已访问节点
      - 蓝色柱 = deterministic_prize（期望）
      - 橙色柱 = real_prize（真实）
      - 直观展示随机波动程度

    Args:
        depot       [2]          depot 坐标
        locs        [num_loc, 2] 客户节点坐标
        det_prizes  [num_loc]    期望奖励（预先已知）
        real_prizes [num_loc]    真实奖励（访问后才知）
        penalties   [num_loc]    惩罚值
        actions     [seq_len]    动作序列
        model_reward             模型总奖励
        save_dir                 保存目录
        instance_id              实例编号

    Returns:
        保存路径，失败返回 None
    """
    try:
        depot = np.array(depot).flatten()
        det_prizes = np.array(det_prizes).flatten()
        real_prizes = np.array(real_prizes).flatten()
        penalties = np.array(penalties).flatten()
        num_loc = len(locs)

        visited_idx = _parse_actions(actions)
        visited_set = set(visited_idx)

        # ── 统计信息 ──
        det_collected = float(sum(det_prizes[i] for i in visited_idx))
        real_collected = float(sum(real_prizes[i] for i in visited_idx))
        saved_penalty = float(sum(penalties[i] for i in visited_set))
        tour_length = _calc_tour_length(depot, locs, visited_idx)

        # ── 视觉属性 ──
        norm_det = _normalize(det_prizes, 80, 500)      # 节点面积（按期望奖励）
        norm_penalty = _normalize(penalties, 0.5, 4.0)
        penalty_cmap = colormaps.get_cmap('BuPu')
        penalty_colors = penalty_cmap(_normalize(penalties, 0.2, 0.9))

        # ── 布局：左路线图 + 右柱形图 ──
        fig, (ax_route, ax_bar) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('#ffffff')

        # ========== 左图：路线图 ==========
        ax = ax_route
        ax.set_facecolor('#f8f9fa')

        # 未访问节点（灰色空心）
        unvisited = [i for i in range(num_loc) if i not in visited_set]
        if unvisited:
            uv_locs = locs[unvisited]
            ax.scatter(
                uv_locs[:, 0], uv_locs[:, 1],
                s=norm_det[unvisited],
                facecolors='#cccccc', edgecolors=penalty_colors[unvisited],
                linewidths=norm_penalty[unvisited],
                alpha=0.5, zorder=3, label='未访问节点'
            )

        # 已访问节点（颜色=real_prize）
        if visited_idx:
            v_locs = locs[visited_idx]
            real_vals = real_prizes[visited_idx]
            node_colors = plt.cm.YlOrRd(_normalize(real_vals, 0.2, 1.0))
            ax.scatter(
                v_locs[:, 0], v_locs[:, 1],
                s=norm_det[visited_idx],
                c=node_colors,
                edgecolors=penalty_colors[visited_idx],
                linewidths=norm_penalty[visited_idx],
                alpha=1.0, zorder=5, label='已访问节点'
            )
            for j, idx in enumerate(visited_idx):
                ax.annotate(
                    f'E:{det_prizes[idx]:.2f}\nR:{real_prizes[idx]:.2f}',
                    xy=(locs[idx, 0], locs[idx, 1]),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=6.5, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
                )

        # Depot
        ax.scatter(depot[0], depot[1], c='tab:green', marker='s', s=220,
                   edgecolors='black', linewidths=1.5, zorder=10, label='Depot')
        ax.annotate('Depot', xy=(depot[0], depot[1]), xytext=(0, -14),
                    textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

        # 路径箭头
        if visited_idx:
            route_pts = np.array([depot] + [locs[i] for i in visited_idx] + [depot])
            dx = np.diff(route_pts[:, 0])
            dy = np.diff(route_pts[:, 1])
            ax.quiver(
                route_pts[:-1, 0], route_pts[:-1, 1], dx, dy,
                scale_units='xy', angles='xy', scale=1,
                color='#2196F3', width=0.004, alpha=0.85, zorder=6
            )

        legend_handles = [
            mpatches.Patch(facecolor='#cccccc', edgecolor='gray', label='未访问节点'),
            mpatches.Patch(facecolor='#ff7043', edgecolor='gray', label='已访问节点（颜色=真实奖励）'),
            mpatches.Patch(facecolor='tab:green', edgecolor='black', label='Depot'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8.5, framealpha=0.9)
        ax.set_title(
            f'SPCTSP 路线图（实例 {instance_id}）\n'
            f'模型奖励: {model_reward:.4f}  |  访问节点: {len(visited_idx)}/{num_loc}\n'
            f'路径长度: {tour_length:.3f}  |  节省惩罚: {saved_penalty:.3f}',
            fontsize=10, pad=8
        )
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.text(0.01, 0.01,
                '节点大小=期望奖励  |  节点颜色=真实奖励  |  注释: E=期望, R=真实',
                transform=ax.transAxes, fontsize=7.5, color='#666666',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

        # ========== 右图：随机奖励对比柱形图 ==========
        ax2 = ax_bar
        ax2.set_facecolor('#f8f9fa')

        if visited_idx:
            x = np.arange(len(visited_idx))
            bar_w = 0.35
            det_vals = det_prizes[visited_idx]
            real_vals = real_prizes[visited_idx]

            bars1 = ax2.bar(x - bar_w/2, det_vals, bar_w,
                            color='#42A5F5', alpha=0.85,
                            label=f'期望奖励（决策时已知）\n总计: {det_collected:.3f}')
            bars2 = ax2.bar(x + bar_w/2, real_vals, bar_w,
                            color='#FF7043', alpha=0.85,
                            label=f'真实奖励（访问后揭晓）\n总计: {real_collected:.3f}')

            # 差值标注（real - det）
            for j, (d, r) in enumerate(zip(det_vals, real_vals)):
                diff = r - d
                color = '#d32f2f' if diff < 0 else '#388e3c'
                ax2.annotate(
                    f'{diff:+.2f}',
                    xy=(x[j] + bar_w/2, r),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=8, color=color, fontweight='bold'
                )

            ax2.set_xticks(x)
            ax2.set_xticklabels([f'节点{visited_idx[j]+1}' for j in range(len(visited_idx))],
                                 rotation=30, ha='right', fontsize=8)
            ax2.legend(fontsize=9, framealpha=0.9)
            ax2.set_ylabel('奖励值', fontsize=10)

            # 标注最低奖励要求线
            ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        else:
            ax2.text(0.5, 0.5, '无已访问节点', transform=ax2.transAxes,
                     ha='center', va='center', fontsize=14, color='gray')

        ax2.set_title(
            f'随机奖励对比（期望 vs 真实）\n'
            f'期望总计: {det_collected:.3f}  |  真实总计: {real_collected:.3f}\n'
            f'差值 = 真实 - 期望，正值=好运，负值=坏运',
            fontsize=10, pad=8
        )
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor('#f8f9fa')

        plt.tight_layout(pad=2.0)

        os.makedirs(save_dir, exist_ok=True)
        uid = uuid.uuid4().hex[:8]
        filename = f'spctsp_comparison_{uid}_inst{instance_id}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return filepath

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────
# 动画（GIF）
# ──────────────────────────────────────────────

def create_spctsp_route_animation(
    depot: np.ndarray,
    locs: np.ndarray,
    det_prizes: np.ndarray,
    real_prizes: np.ndarray,
    penalties: np.ndarray,
    actions: np.ndarray,
    model_reward: float,
    save_dir: str,
    instance_id: int = 1,
    fps: int = 2,
) -> Optional[str]:
    """
    生成 SPCTSP 路线动画（GIF）

    每帧标题额外显示：
      - 累计期望奖励 vs 累计真实奖励
    节点颜色用真实奖励区分，大小用期望奖励区分。

    Args:
        depot / locs / det_prizes / real_prizes / penalties / actions 同对比图函数
        fps      动画帧率
    Returns:
        GIF 文件路径，失败返回 None
    """
    try:
        depot = np.array(depot).flatten()
        det_prizes = np.array(det_prizes).flatten()
        real_prizes = np.array(real_prizes).flatten()
        penalties = np.array(penalties).flatten()
        num_loc = len(locs)

        visited_idx = _parse_actions(actions)
        route_pts = np.array([depot] + [locs[i] for i in visited_idx] + [depot])

        norm_det = _normalize(det_prizes, 80, 500)
        norm_penalty = _normalize(penalties, 0.5, 4.0)
        penalty_cmap = colormaps.get_cmap('BuPu')
        penalty_colors = penalty_cmap(_normalize(penalties, 0.2, 0.9))

        total_frames = len(route_pts)

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#ffffff')

        def draw_frame(frame_idx):
            ax.cla()
            ax.set_facecolor('#f8f9fa')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.2, linestyle='--')

            currently_visited = set(visited_idx[:max(0, frame_idx - 1)])

            # 未访问节点
            unvisited = [i for i in range(num_loc) if i not in currently_visited]
            if unvisited:
                uv = locs[unvisited]
                ax.scatter(uv[:, 0], uv[:, 1],
                           s=norm_det[unvisited],
                           facecolors='#cccccc',
                           edgecolors=penalty_colors[unvisited],
                           linewidths=norm_penalty[unvisited],
                           alpha=0.4, zorder=3)

            # 已访问节点（颜色=真实奖励）
            if currently_visited:
                v_list = list(currently_visited)
                v_locs = locs[v_list]
                node_c = plt.cm.YlOrRd(_normalize(real_prizes[v_list], 0.2, 1.0))
                ax.scatter(v_locs[:, 0], v_locs[:, 1],
                           s=norm_det[v_list],
                           c=node_c,
                           edgecolors=penalty_colors[v_list],
                           linewidths=norm_penalty[v_list],
                           alpha=1.0, zorder=5)

            # Depot
            ax.scatter(depot[0], depot[1], c='tab:green', marker='s', s=200,
                       edgecolors='black', linewidths=1.5, zorder=10)
            ax.annotate('Depot', xy=(depot[0], depot[1]), xytext=(0, -13),
                        textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

            # 已走路段
            if frame_idx > 0:
                seg = route_pts[:frame_idx + 1]
                ax.plot(seg[:, 0], seg[:, 1], 'b-', linewidth=2.2, alpha=0.8, zorder=6)
                dx = np.diff(seg[-2:, 0])
                dy = np.diff(seg[-2:, 1])
                if len(dx) > 0:
                    ax.quiver(seg[-2, 0], seg[-2, 1], dx[0], dy[0],
                              scale_units='xy', angles='xy', scale=1,
                              color='#1565C0', width=0.006, zorder=7)

            # 当前位置
            cur_pos = route_pts[min(frame_idx, len(route_pts)-1)]
            ax.scatter(cur_pos[0], cur_pos[1], c='red', marker='o', s=120,
                       edgecolors='darkred', linewidths=1.5, zorder=11)

            # 标题：期望 vs 真实累计奖励
            cur_visited = visited_idx[:max(0, frame_idx - 1)]
            cur_det = sum(det_prizes[i] for i in cur_visited) if cur_visited else 0.0
            cur_real = sum(real_prizes[i] for i in cur_visited) if cur_visited else 0.0
            ax.set_title(
                f'SPCTSP 路线动画  实例 {instance_id}  步骤 {frame_idx}/{total_frames-1}\n'
                f'期望奖励: {cur_det:.3f}  真实奖励: {cur_real:.3f}  最终奖励: {model_reward:.4f}',
                fontsize=9.5
            )
            ax.text(0.01, 0.01, '大小=期望奖励  颜色=真实奖励',
                    transform=ax.transAxes, fontsize=7.5, color='#888888')

        anim = FuncAnimation(fig, draw_frame, frames=total_frames, interval=600, repeat=True)

        os.makedirs(save_dir, exist_ok=True)
        uid = uuid.uuid4().hex[:8]
        filename = f'spctsp_animation_{uid}_inst{instance_id}.gif'
        filepath = os.path.join(save_dir, filename)

        writer = PillowWriter(fps=fps)
        anim.save(filepath, writer=writer)
        plt.close(fig)

        return filepath

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


__all__ = ['create_spctsp_comparison_plot', 'create_spctsp_route_animation']
