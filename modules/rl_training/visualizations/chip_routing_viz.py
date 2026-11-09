"""Visualizations for the grid-based chip-routing prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

from modules.envs.chip_routing_env import (
    ChipRoutingInstance,
    ChipRoutingSolution,
)


_ROUTE_COLORS = list(plt.get_cmap("tab20").colors)


def _prepare_axis(ax, instance: ChipRoutingInstance, title: str) -> None:
    ax.set_xlim(-0.5, instance.width - 0.5)
    ax.set_ylim(-0.5, instance.height - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, instance.width, max(1, instance.width // 8)))
    ax.set_yticks(np.arange(0, instance.height, max(1, instance.height // 8)))
    ax.set_xticks(np.arange(-0.5, instance.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, instance.height, 1), minor=True)
    ax.grid(which="minor", color="#dfe3e8", linewidth=0.45)
    ax.grid(which="major", visible=False)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    for y, x in np.argwhere(instance.obstacles):
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor="#30343b",
                edgecolor="#17191d",
                linewidth=0.3,
                zorder=1,
            )
        )


def _draw_pins(ax, instance: ChipRoutingInstance, alpha: float = 1.0) -> None:
    for net_id, (sx, sy, tx, ty) in enumerate(instance.nets):
        color = _ROUTE_COLORS[net_id % len(_ROUTE_COLORS)]
        ax.scatter(
            [sx],
            [sy],
            marker="o",
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=alpha,
            zorder=5,
        )
        ax.scatter(
            [tx],
            [ty],
            marker="s",
            s=32,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=alpha,
            zorder=5,
        )
        ax.text(
            sx + 0.18,
            sy + 0.18,
            str(net_id + 1),
            color="#20242a",
            fontsize=6,
            zorder=6,
        )


def _draw_routes(
    ax,
    solution: ChipRoutingSolution,
    net_ids: Optional[Iterable[int]] = None,
) -> None:
    ids = solution.order if net_ids is None else list(net_ids)
    for net_id in ids:
        path = solution.paths.get(int(net_id))
        if not path:
            continue
        color = _ROUTE_COLORS[int(net_id) % len(_ROUTE_COLORS)]
        xs, ys = zip(*path)
        ax.plot(
            xs,
            ys,
            color="white",
            linewidth=4.2,
            alpha=0.85,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.3,
            solid_capstyle="round",
            zorder=3,
        )


def _metrics_text(solution: ChipRoutingSolution) -> str:
    return (
        f"Objective {solution.objective:.1f} | "
        f"Wirelength {solution.wirelength} | "
        f"Overflow {solution.overflow} | "
        f"Bends {solution.bend_count} | "
        f"Routed {solution.routed_count}/{len(solution.order)}"
    )


def create_chip_routing_comparison(
    instance: ChipRoutingInstance,
    baseline_solution: ChipRoutingSolution,
    trained_solution: ChipRoutingSolution,
    save_path,
    index: int = 1,
) -> None:
    """Create before/after route views plus a trained congestion heatmap."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    _prepare_axis(
        axes[0],
        instance,
        f"Before training\n{_metrics_text(baseline_solution)}",
    )
    _draw_routes(axes[0], baseline_solution)
    _draw_pins(axes[0], instance)

    _prepare_axis(
        axes[1],
        instance,
        f"After training\n{_metrics_text(trained_solution)}",
    )
    _draw_routes(axes[1], trained_solution)
    _draw_pins(axes[1], instance)

    heatmap = trained_solution.cell_usage(instance)
    masked = np.ma.masked_where(instance.obstacles, heatmap)
    image = axes[2].imshow(
        masked,
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=max(1.0, float(heatmap.max())),
    )
    obstacle_overlay = np.ma.masked_where(~instance.obstacles, instance.obstacles)
    axes[2].imshow(
        obstacle_overlay,
        origin="lower",
        interpolation="nearest",
        cmap=ListedColormap(["#30343b"]),
        alpha=1.0,
    )
    _prepare_axis(
        axes[2],
        instance,
        f"After-training congestion\nMax edge usage {trained_solution.max_congestion}",
    )
    _draw_pins(axes[2], instance, alpha=0.8)
    colorbar = fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.set_label("Routes touching cell")

    improvement = baseline_solution.objective - trained_solution.objective
    fig.suptitle(
        f"Chip Global Routing - Instance {index} | "
        f"Objective improvement: {improvement:+.1f}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_chip_routing_animation(
    instance: ChipRoutingInstance,
    solution: ChipRoutingSolution,
    save_path,
    fps: int = 2,
) -> None:
    """Animate the learned net order, adding one routed net per frame."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    def update(frame: int):
        ax.clear()
        completed = solution.order[:frame]
        _prepare_axis(
            ax,
            instance,
            f"Learned routing order | {frame}/{len(solution.order)} nets",
        )
        _draw_routes(ax, solution, completed)
        _draw_pins(ax, instance, alpha=0.82)
        if frame:
            last_net = solution.order[frame - 1]
            ax.text(
                0.01,
                1.01,
                f"Latest net: {last_net + 1}",
                transform=ax.transAxes,
                fontsize=9,
                color="#444",
            )
        return ax.lines + ax.collections + ax.patches + ax.texts

    animation = FuncAnimation(
        fig,
        update,
        frames=len(solution.order) + 1,
        interval=max(100, int(1000 / max(1, fps))),
        blit=False,
        repeat=True,
    )
    animation.save(save_path, writer=PillowWriter(fps=max(1, fps)), dpi=110)
    plt.close(fig)

