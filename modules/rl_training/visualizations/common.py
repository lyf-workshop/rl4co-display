"""
通用可视化函数
提供所有问题类型共享的可视化工具
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_MAX_ANIMATION_FRAMES = 100


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


def select_animation_steps(num_steps, max_frames=DEFAULT_MAX_ANIMATION_FRAMES):
    """Uniformly sample animation steps while preserving both endpoints."""
    if num_steps < 1:
        raise ValueError("animation requires at least one route step")
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


def build_closed_depot_route(actions, num_nodes, depot_index=0):
    """Normalize a VRP action sequence into a closed depot-to-depot route."""
    route = np.asarray(actions, dtype=np.int64).reshape(-1)
    if route.size == 0:
        raise ValueError("route action sequence cannot be empty")
    if route.min() < 0 or route.max() >= num_nodes:
        raise ValueError("route action sequence contains an out-of-range node")

    # Batched decoders may pad completed routes with repeated depot actions.
    keep = np.concatenate(([True], route[1:] != route[:-1]))
    route = route[keep]
    if route[0] != depot_index:
        route = np.insert(route, 0, depot_index)
    if route[-1] != depot_index:
        route = np.append(route, depot_index)
    return route


def figure_to_palette_image(fig, colors=128):
    """Copy a Matplotlib canvas into a memory-efficient palette image."""
    from PIL import Image

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
        colors=colors,
    )


def save_palette_gif(frames, save_path, fps):
    """Save and close palette frames, holding the final frame four times longer."""
    if not frames:
        raise ValueError("animation did not produce any frames")
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

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


def create_training_curve_plot(history_epochs, history_losses, history_rewards, save_path, best_reward):
    """
    创建训练曲线图（Loss和Reward）
    
    参数:
        history_epochs: epoch编号列表
        history_losses: loss历史数据
        history_rewards: reward历史数据
        save_path: 保存路径
        best_reward: 最佳reward值
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制Loss曲线
    ax1.plot(history_epochs, history_losses, 'b-o', linewidth=2, markersize=6, label='Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练Loss变化曲线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    
    # 绘制Reward曲线
    ax2.plot(history_epochs, history_rewards, 'g-o', linewidth=2, markersize=6, label='Reward')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('训练Reward变化曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=10)
    
    # 在reward图上标注最佳reward
    if history_rewards:
        best_epoch_idx = history_rewards.index(max(history_rewards))
        best_epoch_num = history_epochs[best_epoch_idx]
        ax2.axhline(y=best_reward, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_reward:.4f}')
        ax2.scatter([best_epoch_num], [best_reward], color='red', s=100, zorder=5, marker='*')
        ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



