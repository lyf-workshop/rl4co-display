"""
VRPTW问题专用可视化函数
提供VRPTW路线动画、时间调度图等可视化
"""

import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Rectangle

from .common import (
    DEFAULT_MAX_ANIMATION_FRAMES,
    build_closed_depot_route,
    figure_to_palette_image,
    save_palette_gif,
    select_animation_steps,
)

logger = logging.getLogger('rl4co_display')

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


def _remove_single_batch_dimension(value, target_dimensions):
    values = _to_numpy(value)
    while values.ndim > target_dimensions and values.shape[0] == 1:
        values = values[0]
    return values


def _normalise_node_values(values, num_nodes, default, field_name):
    if values is None:
        result = np.full(num_nodes, default, dtype=float)
        result[0] = 0.0
        return result

    array = _remove_single_batch_dimension(values, 1).reshape(-1).astype(float)
    if array.size == 1:
        result = np.full(num_nodes, float(array[0]), dtype=float)
        result[0] = 0.0
        return result
    if array.size == num_nodes:
        return array
    if array.size == num_nodes - 1:
        return np.concatenate(([0.0], array))
    raise ValueError(
        f"VRPTW {field_name} length {array.size} does not match "
        f"{num_nodes - 1} customers"
    )


def _normalise_time_windows(values, num_nodes):
    if values is None:
        customer_windows = np.tile(np.array([0.0, 50.0]), (num_nodes - 1, 1))
        return np.vstack((np.array([[0.0, 50.0]]), customer_windows))

    windows = _remove_single_batch_dimension(values, 2).astype(float)
    if windows.shape == (num_nodes - 1, 2):
        depot_end = max(float(windows[:, 1].max()), 1.0)
        windows = np.vstack((np.array([[0.0, depot_end]]), windows))
    if windows.shape != (num_nodes, 2):
        raise ValueError(
            f"VRPTW time_windows shape {windows.shape} does not match "
            f"{num_nodes - 1} customers"
        )
    if np.any(windows[:, 1] < windows[:, 0]):
        raise ValueError("VRPTW time window end must not precede its start")
    return windows


def _prepare_vrptw_route(
    locs,
    actions,
    demands=None,
    time_windows=None,
    service_times=None,
    speed=1.0,
):
    """Build a closed route and precompute distance, load, time and visits."""
    locs = _remove_single_batch_dimension(locs, 2).astype(float)
    if locs.ndim != 2 or locs.shape[1] != 2 or len(locs) < 2:
        raise ValueError("VRPTW locations must have shape [num_nodes, 2]")
    if speed <= 0:
        raise ValueError("VRPTW speed must be greater than zero")

    route = build_closed_depot_route(
        _to_numpy(actions),
        num_nodes=len(locs),
        depot_index=0,
    )
    if len(route) < 2:
        raise ValueError("VRPTW route must visit at least one customer")

    node_demands = _normalise_node_values(
        demands,
        len(locs),
        default=0.0,
        field_name="demand",
    )
    node_service_times = _normalise_node_values(
        service_times,
        len(locs),
        default=10.0,
        field_name="service_time",
    )
    node_time_windows = _normalise_time_windows(time_windows, len(locs))

    segments = np.stack((locs[route[:-1]], locs[route[1:]]), axis=1)
    segment_distances = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    cumulative_distances = np.concatenate(([0.0], np.cumsum(segment_distances)))
    loads = np.zeros(len(route), dtype=float)
    elapsed_times = np.zeros(len(route), dtype=float)
    visits = []

    current_load = 0.0
    current_time = 0.0
    for step, destination in enumerate(route[1:], start=1):
        current_time += segment_distances[step - 1] / speed
        if destination == 0:
            current_load = 0.0
        else:
            current_load += node_demands[destination]
            arrival_time = current_time
            earliest, latest = node_time_windows[destination]
            service_start = max(arrival_time, earliest)
            current_time = service_start + node_service_times[destination]
            visits.append(
                (step, destination - 1, arrival_time, service_start, latest)
            )
        loads[step] = current_load
        elapsed_times[step] = current_time

    return (
        locs,
        route,
        node_demands,
        node_time_windows,
        segments,
        cumulative_distances,
        loads,
        elapsed_times,
        visits,
    )


def create_vrptw_route_animation(
    td,
    actions,
    save_path,
    title="VRPTW路线生成过程（带时间窗）",
    fps=2,
    max_frames=DEFAULT_MAX_ANIMATION_FRAMES,
    progress_callback=None,
):
    """Create a bounded-memory VRPTW route and schedule animation."""
    locs_value = td.get("locs") if hasattr(td, "get") else td["locs"]
    demands = None
    time_windows = None
    service_times = None
    if hasattr(td, "get"):
        demands = td.get("demand", td.get("demands", None))
        time_windows = td.get("time_windows", None)
        service_times = td.get("service_time", None)
    else:
        demands = td["demand"] if "demand" in td else td.get("demands")
        time_windows = td.get("time_windows")
        service_times = td.get("service_time")

    (
        locs,
        route,
        node_demands,
        node_time_windows,
        segments,
        distances,
        loads,
        elapsed_times,
        visits,
    ) = _prepare_vrptw_route(
        locs_value,
        actions,
        demands=demands,
        time_windows=time_windows,
        service_times=service_times,
    )

    customers = locs[1:]
    customer_windows = node_time_windows[1:]
    route_steps = len(route) - 1
    frame_steps = select_animation_steps(route_steps, max_frames=max_frames)
    num_customers = len(customers)
    marker_size = max(8, min(80, 1400 / max(num_customers, 1)))
    line_width = max(0.7, min(2.4, 15 / np.sqrt(max(num_customers, 1))))

    fig, (route_ax, schedule_ax) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    frames = []
    try:
        customer_scatter = route_ax.scatter(
            customers[:, 0],
            customers[:, 1],
            c=customer_windows[:, 0],
            s=marker_size,
            cmap="coolwarm",
            alpha=0.8,
            edgecolors="#1f2937" if num_customers <= 100 else "none",
            linewidths=0.5,
            zorder=3,
        )
        colorbar = fig.colorbar(customer_scatter, ax=route_ax, fraction=0.046, pad=0.04)
        colorbar.set_label("时间窗起始时间")
        route_ax.scatter(
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
        if num_customers <= 50:
            route_ax.text(
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
                earliest, latest = node_time_windows[node]
                route_ax.text(
                    x_coord,
                    y_coord,
                    f"{node}\n[{earliest:.0f}-{latest:.0f}]",
                    fontsize=5.5,
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
        route_ax.add_collection(route_lines)
        current_marker = route_ax.scatter(
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
        route_ax.add_patch(direction_arrow)

        minimum = locs.min(axis=0)
        maximum = locs.max(axis=0)
        span = np.maximum(maximum - minimum, 1e-3)
        margin = np.maximum(span * 0.06, 0.02)
        route_ax.set_xlim(minimum[0] - margin[0], maximum[0] + margin[0])
        route_ax.set_ylim(minimum[1] - margin[1], maximum[1] + margin[1])
        route_ax.set_aspect("equal", adjustable="box")
        route_ax.grid(True, alpha=0.25, linestyle="--")
        route_ax.set_xlabel("X 坐标")
        route_ax.set_ylabel("Y 坐标")
        route_ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        route_title = route_ax.set_title("", fontsize=11, fontweight="bold")
        route_info = route_ax.text(
            0.02,
            0.98,
            "",
            transform=route_ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        for customer_index, (earliest, latest) in enumerate(customer_windows):
            schedule_ax.add_patch(
                Rectangle(
                    (earliest, customer_index - 0.35),
                    latest - earliest,
                    0.7,
                    facecolor="#a5d6a7",
                    edgecolor="#4b5563",
                    alpha=0.35,
                    linewidth=0.6,
                )
            )
        service_markers = schedule_ax.scatter(
            [],
            [],
            c="#d32f2f",
            s=max(12, marker_size * 0.7),
            edgecolors="#7f1d1d",
            linewidths=0.5,
            zorder=5,
            label="服务开始",
        )
        max_schedule_time = max(
            float(customer_windows[:, 1].max()),
            float(elapsed_times[-1]),
            1.0,
        )
        schedule_ax.set_xlim(0, max_schedule_time * 1.08)
        schedule_ax.set_ylim(-0.75, num_customers - 0.25)
        tick_count = min(num_customers, 20)
        tick_indices = np.unique(
            np.linspace(0, num_customers - 1, tick_count).astype(int)
        )
        schedule_ax.set_yticks(tick_indices)
        schedule_ax.set_yticklabels(
            [f"客户 {index + 1}" for index in tick_indices],
            fontsize=7,
        )
        schedule_ax.set_xlabel("时间")
        schedule_ax.set_ylabel("客户")
        schedule_ax.set_title("时间窗与服务开始时间", fontsize=11, fontweight="bold")
        schedule_ax.grid(True, alpha=0.25, axis="x")
        schedule_ax.legend(loc="upper right", fontsize=8)
        progress_artist = fig.text(0.5, 0.02, "", ha="center", fontsize=9)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))

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

            visible_visits = [visit for visit in visits if visit[0] <= step]
            if visible_visits:
                marker_offsets = np.array(
                    [[visit[3], visit[1]] for visit in visible_visits],
                    dtype=float,
                )
                service_markers.set_offsets(marker_offsets)
                service_markers.set_color(
                    [
                        "#d32f2f" if visit[3] > visit[4] else "#2e7d32"
                        for visit in visible_visits
                    ]
                )
            else:
                service_markers.set_offsets(np.empty((0, 2)))

            percent = int(round(step * 100 / route_steps))
            current_node = route[step]
            node_label = "仓库" if current_node == 0 else f"客户 {current_node}"
            route_title.set_text(f"步骤 {step}/{route_steps} | 当前：{node_label}")
            route_info.set_text(
                f"累计距离: {distances[step]:.2f}\n"
                f"当前载重: {loads[step]:.2f}\n"
                f"当前时间: {elapsed_times[step]:.1f}"
            )
            progress_artist.set_text(f"动画进度: {percent}%")
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
            if len(time_windows) == len(locs):
                time_windows = time_windows[1:]
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
            if len(time_windows) == len(locs):
                time_windows = time_windows[1:]
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



