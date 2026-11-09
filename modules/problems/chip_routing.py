"""Chip global-routing problem definition."""

from typing import Any, Callable, Dict

from .base_problem import BaseProblem


class ChipRoutingProblem(BaseProblem):
    """
    Grid-based two-pin global routing.

    A policy orders nets; an A* router realizes the paths while minimizing
    wirelength, bends, capacity overflow, and unrouted nets.
    """

    def _init_problem_params(self):
        self.grid_width = int(self.config.get("grid_width", 18))
        self.grid_height = int(self.config.get("grid_height", 18))
        self.num_nets = int(
            self.config.get("num_nets", self.config.get("num_loc", 12))
        )
        self.num_loc = self.num_nets
        self.obstacle_ratio = float(self.config.get("obstacle_ratio", 0.10))
        self.edge_capacity = int(self.config.get("edge_capacity", 1))

    def get_problem_type(self) -> str:
        return "chip_routing"

    def get_problem_name(self) -> str:
        return "Chip Global Routing"

    def create_environment(self):
        from modules.envs.chip_routing_env import (
            ChipRoutingGenerator,
            GridRoutingEnv,
        )

        generator = ChipRoutingGenerator(
            width=self.grid_width,
            height=self.grid_height,
            num_nets=self.num_nets,
            obstacle_ratio=self.obstacle_ratio,
            edge_capacity=self.edge_capacity,
            seed=int(self.config.get("seed", 1234)),
        )
        return GridRoutingEnv(
            generator,
            congestion_weight=float(self.config.get("congestion_weight", 2.0)),
            overflow_weight=float(self.config.get("overflow_weight", 12.0)),
            bend_weight=float(self.config.get("bend_weight", 0.25)),
            unrouted_penalty=float(self.config.get("unrouted_penalty", 100.0)),
        )

    def get_visualization_functions(self) -> Dict[str, Callable]:
        from modules.rl_training.visualizations.chip_routing_viz import (
            create_chip_routing_animation,
            create_chip_routing_comparison,
        )

        return {
            "animation": create_chip_routing_animation,
            "comparison": create_chip_routing_comparison,
        }

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "num_nets": self.num_nets,
            "obstacle_ratio": self.obstacle_ratio,
            "edge_capacity": self.edge_capacity,
            "batch_size": self.batch_size,
        }

    def _validate_problem_params(self):
        if not 8 <= self.grid_width <= 40:
            return False, "芯片布线网格宽度必须在 8 到 40 之间"
        if not 8 <= self.grid_height <= 40:
            return False, "芯片布线网格高度必须在 8 到 40 之间"
        if not 4 <= self.num_nets <= 40:
            return False, "芯片布线网络数量必须在 4 到 40 之间"
        if not 0.0 <= self.obstacle_ratio <= 0.30:
            return False, "障碍率必须在 0 到 0.30 之间"
        if not 1 <= self.edge_capacity <= 4:
            return False, "边容量必须在 1 到 4 之间"
        return True, ""

    def get_problem_description(self) -> str:
        return (
            f"芯片全局布线：在 {self.grid_width}x{self.grid_height} G-Cell 网格中"
            f"连接 {self.num_nets} 个两端点网络；强化学习决定网络顺序，"
            "A* 完成路径搜索。"
        )

    def get_problem_features(self):
        return [
            "两端点网络",
            "G-Cell 网格",
            "宏单元禁布区",
            "边容量约束",
            "线长与拥塞联合优化",
        ]


__all__ = ["ChipRoutingProblem"]
