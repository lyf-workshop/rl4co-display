"""
PCTSP (Prize Collecting TSP) 可视化函数

视觉约定（参考 rl4co pctsp/render.py）：
  - 节点大小  ↔  prize 值（奖励越高节点越大）
  - 节点边框颜色/宽度  ↔  penalty 值（惩罚越高边框越深越粗）
  - 已访问节点：实心彩色（autumn_r colormap）
  - 未访问节点：灰色半透明空心
  - Depot：绿色方形标记
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
# 内部辅助函数
# ──────────────────────────────────────────────

def _normalize(arr: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """将数组线性归一化到 [lo, hi]"""
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return lo + (hi - lo) * (arr - mn) / (mx - mn)
    return np.full_like(arr, (lo + hi) / 2.0)


def _parse_actions(actions: np.ndarray) -> list:
    """
    解析动作序列，返回实际访问的客户节点索引列表（不含 depot=0）。
    action 数组里 0 代表 depot，其余为 1-indexed 客户节点。
    """
    visited = []
    for a in actions:
        a = int(a)
        if a == 0:
            break  # 返回 depot 即结束
        visited.append(a - 1)  # 转为 0-indexed 客户节点
    return visited


def _calc_tour_length(depot: np.ndarray, locs: np.ndarray, visited_idx: list) -> float:
    """计算路径总长度（depot→v0→v1→...→depot）"""
    if not visited_idx:
        return 0.0
    pts = [depot] + [locs[i] for i in visited_idx] + [depot]
    return float(sum(np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i])) for i in range(len(pts)-1)))


# ──────────────────────────────────────────────
# 对比图
# ──────────────────────────────────────────────

def create_pctsp_comparison_plot(
    depot: np.ndarray,
    locs: np.ndarray,
    prizes: np.ndarray,
    penalties: np.ndarray,
    actions: np.ndarray,
    model_reward: float,
    save_dir: str,
    instance_id: int = 1,
) -> Optional[str]:
    """
    生成 PCTSP 路线对比图（静态 PNG）

    Args:
        depot     [2]         depot 坐标
        locs      [num_loc,2] 客户节点坐标
        prizes    [num_loc]   各客户节点奖励值
        penalties [num_loc]   各客户节点惩罚值
        actions   [seq_len]   动作序列（0=depot，1..n=客户节点）
        model_reward          模型获得的总奖励（标量）
        save_dir              保存目录
        instance_id           实例编号（用于文件名）

    Returns:
        保存路径，失败返回 None
    """
    try:
        depot = np.array(depot).flatten()
        prizes = np.array(prizes).flatten()
        penalties = np.array(penalties).flatten()
        num_loc = len(locs)

        visited_idx = _parse_actions(actions)
        visited_set = set(visited_idx)

        # ── 统计信息 ──
        collected_prize = float(sum(prizes[i] for i in visited_idx))
        saved_penalty = float(sum(penalties[i] for i in visited_set))
        total_penalty = float(penalties.sum())
        tour_length = _calc_tour_length(depot, locs, visited_idx)

        # ── 节点视觉属性 ──
        norm_prize = _normalize(prizes, 80, 500)          # 节点面积
        norm_penalty = _normalize(penalties, 0.5, 4.0)    # 边框宽度
        penalty_cmap = colormaps.get_cmap('BuPu')
        penalty_colors = penalty_cmap(_normalize(penalties, 0.2, 0.9))

        # ── 绘图 ──
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#ffffff')

        # 未访问节点（灰色空心）
        unvisited = [i for i in range(num_loc) if i not in visited_set]
        if unvisited:
            uv_locs = locs[unvisited]
            ax.scatter(
                uv_locs[:, 0], uv_locs[:, 1],
                s=norm_prize[unvisited],
                facecolors='#cccccc', edgecolors=penalty_colors[unvisited],
                linewidths=norm_penalty[unvisited],
                alpha=0.5, zorder=3, label='未访问节点'
            )

        # 已访问节点（autumn_r，实心）
        if visited_idx:
            v_locs = locs[visited_idx]
            v_norm_prize = norm_prize[visited_idx]
            v_penalty_colors = penalty_colors[visited_idx]
            v_norm_penalty = norm_penalty[visited_idx]
            prize_vals_visited = prizes[visited_idx]
            node_colors = plt.cm.autumn_r(_normalize(prize_vals_visited, 0.2, 1.0))
            ax.scatter(
                v_locs[:, 0], v_locs[:, 1],
                s=v_norm_prize,
                c=node_colors,
                edgecolors=v_penalty_colors,
                linewidths=v_norm_penalty,
                alpha=1.0, zorder=5, label='已访问节点'
            )
            # 标注奖励值
            for j, idx in enumerate(visited_idx):
                ax.annotate(
                    f'p={prizes[idx]:.2f}',
                    xy=(locs[idx, 0], locs[idx, 1]),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=7, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none')
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

        # ── 图例与标题 ──
        legend_handles = [
            mpatches.Patch(facecolor='#cccccc', edgecolor='gray', label='未访问节点（承担惩罚）'),
            mpatches.Patch(facecolor='#ff7043', edgecolor='gray', label='已访问节点（获得奖励）'),
            mpatches.Patch(facecolor='tab:green', edgecolor='black', label='Depot'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)

        title_lines = [
            f'PCTSP 路线图（实例 {instance_id}）',
            f'奖励: {model_reward:.4f}  |  收集奖励: {collected_prize:.3f} / 要求≥1.0',
            f'节省惩罚: {saved_penalty:.3f}  |  路径长度: {tour_length:.3f}  |  总惩罚: {total_penalty:.3f}',
            f'访问节点: {len(visited_idx)}/{num_loc}',
        ]
        ax.set_title('\n'.join(title_lines), fontsize=11, pad=10)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # ── 说明文字（左下角） ──
        note = '节点大小 = 奖励值  |  边框深浅 = 惩罚值'
        ax.text(0.01, 0.01, note, transform=ax.transAxes,
                fontsize=8, color='#666666',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        uid = uuid.uuid4().hex[:8]
        filename = f'pctsp_comparison_{uid}_inst{instance_id}.png'
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

def create_pctsp_route_animation(
    depot: np.ndarray,
    locs: np.ndarray,
    prizes: np.ndarray,
    penalties: np.ndarray,
    actions: np.ndarray,
    model_reward: float,
    save_dir: str,
    instance_id: int = 1,
    fps: int = 2,
) -> Optional[str]:
    """
    生成 PCTSP 路线动画（GIF）

    动画帧：
      帧 0     — 显示所有节点（初始状态）
      帧 1..n  — 逐步添加路径段，节点变为已访问状态
      帧 n+1   — 返回 depot，完整路线

    Args:
        depot / locs / prizes / penalties / actions  同 create_pctsp_comparison_plot
        fps      动画帧率
    Returns:
        GIF 文件路径，失败返回 None
    """
    try:
        depot = np.array(depot).flatten()
        prizes = np.array(prizes).flatten()
        penalties = np.array(penalties).flatten()
        num_loc = len(locs)

        visited_idx = _parse_actions(actions)
        # 完整路径点（包括起/终 depot）
        route_pts = np.array([depot] + [locs[i] for i in visited_idx] + [depot])

        # ── 视觉属性 ──
        norm_prize = _normalize(prizes, 80, 500)
        norm_penalty = _normalize(penalties, 0.5, 4.0)
        penalty_cmap = colormaps.get_cmap('BuPu')
        penalty_colors = penalty_cmap(_normalize(penalties, 0.2, 0.9))

        total_frames = len(route_pts)  # depot + n visited + depot

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#ffffff')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2, linestyle='--')

        def draw_frame(frame_idx):
            ax.cla()
            ax.set_facecolor('#f8f9fa')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.2, linestyle='--')

            # 当前已访问节点集合
            currently_visited = set(visited_idx[:max(0, frame_idx - 1)])

            # 未访问节点
            unvisited = [i for i in range(num_loc) if i not in currently_visited]
            if unvisited:
                uv = locs[unvisited]
                ax.scatter(uv[:, 0], uv[:, 1],
                           s=norm_prize[unvisited],
                           facecolors='#cccccc',
                           edgecolors=penalty_colors[unvisited],
                           linewidths=norm_penalty[unvisited],
                           alpha=0.4, zorder=3)

            # 已访问节点
            if currently_visited:
                v_list = list(currently_visited)
                v_locs = locs[v_list]
                node_c = plt.cm.autumn_r(_normalize(prizes[v_list], 0.2, 1.0))
                ax.scatter(v_locs[:, 0], v_locs[:, 1],
                           s=norm_prize[v_list],
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

            # 当前位置标记
            cur_pos = route_pts[min(frame_idx, len(route_pts)-1)]
            ax.scatter(cur_pos[0], cur_pos[1], c='red', marker='o', s=120,
                       edgecolors='darkred', linewidths=1.5, zorder=11)

            # 统计信息
            cur_visited = visited_idx[:max(0, frame_idx - 1)]
            cur_prize = sum(prizes[i] for i in cur_visited) if cur_visited else 0.0
            ax.set_title(
                f'PCTSP 路线动画  实例 {instance_id}  步骤 {frame_idx}/{total_frames-1}\n'
                f'已收集奖励: {cur_prize:.3f} / 要求≥1.0  |  最终奖励: {model_reward:.4f}',
                fontsize=10
            )

        anim = FuncAnimation(fig, draw_frame, frames=total_frames, interval=600, repeat=True)

        os.makedirs(save_dir, exist_ok=True)
        uid = uuid.uuid4().hex[:8]
        filename = f'pctsp_animation_{uid}_inst{instance_id}.gif'
        filepath = os.path.join(save_dir, filename)

        writer = PillowWriter(fps=fps)
        anim.save(filepath, writer=writer)
        plt.close(fig)

        return filepath

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


__all__ = ['create_pctsp_comparison_plot', 'create_pctsp_route_animation']
