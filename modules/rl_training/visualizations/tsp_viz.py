"""
TSP问题专用可视化函数
提供TSP路线动画、对比图等可视化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

DEFAULT_MAX_ANIMATION_FRAMES = 100
CITY_LABEL_LIMIT = 100

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


def _select_animation_steps(num_steps, max_frames=DEFAULT_MAX_ANIMATION_FRAMES):
    """Sample route steps while retaining the initial and completed frames."""
    if num_steps < 1:
        raise ValueError("TSP animation requires at least one node")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")

    frame_count = min(num_steps + 1, max_frames)
    steps = np.rint(np.linspace(0, num_steps, num=frame_count)).astype(int)
    steps = np.unique(steps)
    if steps[0] != 0:
        steps = np.insert(steps, 0, 0)
    if steps[-1] != num_steps:
        steps = np.append(steps, num_steps)
    return steps


def _calculate_cumulative_route_costs(locs, actions):
    """Return route points and cumulative costs, including the closing edge."""
    route_locs = locs[actions]
    num_steps = len(route_locs)
    costs = np.zeros(num_steps + 1, dtype=np.float64)

    if num_steps > 1:
        segment_costs = np.linalg.norm(np.diff(route_locs, axis=0), axis=1)
        costs[2:] = np.cumsum(segment_costs)
        costs[-1] += np.linalg.norm(route_locs[-1] - route_locs[0])

    return route_locs, costs


def _figure_to_palette_image(fig):
    """Copy the current canvas into a memory-efficient palette image."""
    fig.canvas.draw()
    try:
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3].copy()
    except AttributeError:
        try:
            width, height = fig.canvas.get_width_height()
            rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            rgb = rgb.reshape((height, width, 3)).copy()
        except AttributeError:
            width, height = fig.canvas.get_width_height()
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            argb = argb.reshape((height, width, 4))
            rgb = argb[:, :, 1:].copy()

    palette_group = getattr(Image, "Palette", None)
    adaptive_palette = getattr(
        palette_group,
        "ADAPTIVE",
        getattr(Image, "ADAPTIVE", 1),
    )
    return Image.fromarray(rgb).convert(
        "P",
        palette=adaptive_palette,
        colors=128,
    )


def create_tsp_route_animation(
    td,
    actions,
    save_path,
    title="TSP路线生成过程",
    fps=2,
    max_frames=DEFAULT_MAX_ANIMATION_FRAMES,
    progress_callback=None,
):
    """Create an adaptive TSP route GIF without buffering full RGB frames."""
    locs_value = td.get("locs") if hasattr(td, "get") else td["locs"]
    if hasattr(locs_value, "detach"):
        locs_value = locs_value.detach()
    if hasattr(locs_value, "cpu"):
        locs_value = locs_value.cpu()
    locs = np.asarray(locs_value)
    if locs.ndim == 3 and locs.shape[0] == 1:
        locs = locs[0]
    if locs.ndim != 2 or locs.shape[1] != 2:
        raise ValueError(f"TSP坐标应为 [节点数, 2]，实际形状为 {locs.shape}")

    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if actions.size == 0:
        raise ValueError("TSP动画动作序列不能为空")
    if actions.min() < 0 or actions.max() >= len(locs):
        raise ValueError("TSP动画动作序列包含越界节点")
    if fps <= 0:
        raise ValueError("fps 必须大于 0")

    num_cities = len(locs)
    num_steps = len(actions)
    frame_steps = _select_animation_steps(num_steps, max_frames=max_frames)
    route_locs, cumulative_costs = _calculate_cumulative_route_costs(locs, actions)
    frames = []

    point_size = max(8, min(200, 12000 / max(num_cities, 1)))
    line_width = 3 if num_cities <= CITY_LABEL_LIMIT else 1.2
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    ax.scatter(
        locs[:, 0],
        locs[:, 1],
        c="lightblue",
        s=point_size,
        zorder=3,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.6 if num_cities > CITY_LABEL_LIMIT else 1.5,
    )

    if num_cities <= CITY_LABEL_LIMIT:
        for city_index, (x_coord, y_coord) in enumerate(locs):
            ax.text(
                x_coord,
                y_coord,
                str(city_index),
                fontsize=8,
                ha="center",
                va="center",
                fontweight="bold",
                color="darkblue",
            )

    route_line, = ax.plot(
        [],
        [],
        color="blue",
        linewidth=line_width,
        alpha=0.75,
        zorder=1,
    )
    current_marker = ax.scatter(
        [],
        [],
        c="red",
        s=max(point_size * 2, 80),
        zorder=5,
        marker="*",
        edgecolors="darkred",
        linewidths=1.2,
        label="当前城市",
    )
    start_city = actions[0]
    ax.scatter(
        locs[start_city, 0],
        locs[start_city, 1],
        c="green",
        s=max(point_size * 1.5, 60),
        zorder=4,
        marker="s",
        edgecolors="darkgreen",
        linewidths=1.2,
        label=f"起点: 城市 {start_city}",
    )
    direction_arrow = FancyArrowPatch(
        (0, 0),
        (0, 0),
        arrowstyle="->",
        mutation_scale=12,
        color="blue",
        linewidth=1.5,
        alpha=0.8,
        zorder=2,
        visible=False,
    )
    ax.add_patch(direction_arrow)

    x_span = float(np.ptp(locs[:, 0]))
    y_span = float(np.ptp(locs[:, 1]))
    x_margin = max(x_span * 0.05, 0.02)
    y_margin = max(y_span * 0.05, 0.02)
    ax.set_xlim(
        float(locs[:, 0].min()) - x_margin,
        float(locs[:, 0].max()) + x_margin,
    )
    ax.set_ylim(
        float(locs[:, 1].min()) - y_margin,
        float(locs[:, 1].max()) + y_margin,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("X 坐标", fontsize=11)
    ax.set_ylabel("Y 坐标", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    title_artist = ax.set_title(
        f"{title}\n开始构建路线...",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    progress_artist = ax.text(
        0.5,
        -0.12,
        "进度: 0%",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()

    next_progress_report = 0
    try:
        for frame_index, step in enumerate(frame_steps):
            if step == 0:
                route_points = np.empty((0, 2))
                info_text = "开始构建路线..."
                current_marker.set_offsets(np.empty((0, 2)))
                direction_arrow.set_visible(False)
            else:
                route_points = route_locs[:step]
                if step == num_steps and num_steps > 1:
                    route_points = np.vstack((route_points, route_locs[0]))

                current_marker.set_offsets(route_locs[step - 1].reshape(1, 2))
                if len(route_points) >= 2:
                    direction_arrow.set_positions(route_points[-2], route_points[-1])
                    direction_arrow.set_visible(True)
                else:
                    direction_arrow.set_visible(False)

                if step < num_steps:
                    info_text = (
                        f"第 {step} 步 | 已访问 {step} 个城市 | "
                        f"累计成本: {cumulative_costs[step]:.3f}"
                    )
                else:
                    info_text = (
                        f"完成！总共 {num_steps} 个城市 | "
                        f"总成本: {cumulative_costs[step]:.3f}"
                    )

            if len(route_points) > 0:
                route_line.set_data(route_points[:, 0], route_points[:, 1])
            else:
                route_line.set_data([], [])

            progress = int(round(step / num_steps * 100))
            title_artist.set_text(f"{title}\n{info_text}")
            progress_artist.set_text(f"进度: {progress}%")
            frames.append(_figure_to_palette_image(fig))

            if progress_callback and (
                progress >= next_progress_report
                or frame_index == len(frame_steps) - 1
            ):
                progress_callback(progress, int(step), num_steps)
                next_progress_report = min(100, progress + 10)
    except Exception:
        for frame in frames:
            frame.close()
        raise
    finally:
        plt.close(fig)

    duration = max(1, int(1000 / fps))
    durations = [duration] * len(frames)
    durations[-1] = duration * 4
    try:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
            disposal=2,
        )
    finally:
        for frame in frames:
            frame.close()


def create_tsp_comparison_plot(env, td, actions_untrained, rewards_untrained,
                               actions_trained, rewards_trained, save_path, index=1,
                               left_label="Random"):
    """
    创建TSP路线对比图（基线 vs 训练后）

    参数:
        env: TSP环境
        td: TensorDict
        actions_untrained: 基线模型的动作序列
        rewards_untrained: 基线模型的奖励
        actions_trained: 训练后模型的动作序列
        rewards_trained: 训练后模型的奖励
        save_path: 保存路径
        index: 图片索引（用于多实例）
        left_label: 左图标题前缀，默认"Random"；
                    DeepACO 等非自回归策略传 "Random (ACO Baseline)"
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # 渲染基线和训练后的路线
    env.render(td, actions_untrained, ax=axs[0])
    env.render(td, actions_trained, ax=axs[1])

    # 设置标题
    axs[0].set_title(f"{left_label} | Cost = {-rewards_untrained.item():.3f}")
    axs[1].set_title(f"Trained (Greedy) | Cost = {-rewards_trained.item():.3f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



