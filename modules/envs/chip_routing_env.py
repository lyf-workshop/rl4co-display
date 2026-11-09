"""
Grid-based chip global-routing environment.

The reinforcement-learning policy chooses the order in which two-pin nets are
routed. A deterministic A* router then materializes every path on the grid and
returns wirelength, bend, congestion-overflow, and unrouted-net metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


GridPoint = Tuple[int, int]
GridPath = List[GridPoint]


@dataclass(frozen=True)
class ChipRoutingInstance:
    """One global-routing instance with two-pin nets."""

    width: int
    height: int
    nets: np.ndarray
    obstacles: np.ndarray
    edge_capacity: int = 1

    @property
    def num_nets(self) -> int:
        return int(self.nets.shape[0])


@dataclass
class ChipRoutingSolution:
    """Result produced by routing one instance in a given net order."""

    order: List[int]
    paths: Dict[int, GridPath]
    wirelength: int
    bend_count: int
    overflow: int
    max_congestion: int
    unrouted_nets: List[int]
    objective: float
    horizontal_usage: np.ndarray
    vertical_usage: np.ndarray

    @property
    def routed_count(self) -> int:
        return len(self.paths)

    def cell_usage(self, instance: ChipRoutingInstance) -> np.ndarray:
        """Return a heatmap-friendly count of how many routes touch each cell."""
        usage = np.zeros((instance.height, instance.width), dtype=np.float32)
        for path in self.paths.values():
            for x, y in path:
                usage[y, x] += 1.0
        return usage


class ChipRoutingGenerator:
    """Generate reproducible synthetic global-routing instances."""

    def __init__(
        self,
        width: int = 18,
        height: int = 18,
        num_nets: int = 12,
        obstacle_ratio: float = 0.10,
        edge_capacity: int = 1,
        seed: int = 1234,
    ):
        if width < 4 or height < 4:
            raise ValueError("Grid width and height must both be at least 4")
        if num_nets < 2:
            raise ValueError("At least two nets are required")
        if not 0.0 <= obstacle_ratio <= 0.40:
            raise ValueError("obstacle_ratio must be in [0, 0.40]")
        if edge_capacity < 1:
            raise ValueError("edge_capacity must be positive")

        self.width = int(width)
        self.height = int(height)
        self.num_nets = int(num_nets)
        self.obstacle_ratio = float(obstacle_ratio)
        self.edge_capacity = int(edge_capacity)
        self._rng = np.random.default_rng(seed)

    def _generate_obstacles(self, rng: np.random.Generator) -> np.ndarray:
        obstacles = np.zeros((self.height, self.width), dtype=bool)
        target = int(round(self.width * self.height * self.obstacle_ratio))
        attempts = 0

        # Small rectangles look closer to blockages/macros than isolated noise.
        while int(obstacles.sum()) < target and attempts < target * 20 + 100:
            rect_w = int(rng.integers(1, min(4, self.width - 1)))
            rect_h = int(rng.integers(1, min(4, self.height - 1)))
            x = int(rng.integers(1, max(2, self.width - rect_w)))
            y = int(rng.integers(1, max(2, self.height - rect_h)))
            obstacles[y : y + rect_h, x : x + rect_w] = True
            attempts += 1

        if int(obstacles.sum()) > target:
            occupied = np.argwhere(obstacles)
            remove_count = int(obstacles.sum()) - target
            selected = rng.choice(
                len(occupied), remove_count, replace=False
            )
            for y, x in occupied[selected]:
                obstacles[y, x] = False
        return obstacles

    def _choose_endpoints(
        self,
        obstacles: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        free_yx = np.argwhere(~obstacles)
        required = self.num_nets * 2
        if len(free_yx) < required:
            raise ValueError(
                f"Grid has only {len(free_yx)} free cells, "
                f"but {required} pin cells are required"
            )

        chosen = free_yx[rng.choice(len(free_yx), required, replace=False)]
        points = [(int(x), int(y)) for y, x in chosen]
        rng.shuffle(points)

        # Pair short and long projections to avoid a batch dominated by tiny nets.
        points.sort(key=lambda point: point[0] + point[1])
        sources = points[: self.num_nets]
        targets = list(reversed(points[self.num_nets :]))
        nets = [
            [source[0], source[1], target[0], target[1]]
            for source, target in zip(sources, targets)
        ]
        return np.asarray(nets, dtype=np.int64)

    def generate(self, seed: Optional[int] = None) -> ChipRoutingInstance:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        obstacles = self._generate_obstacles(rng)
        nets = self._choose_endpoints(obstacles, rng)
        return ChipRoutingInstance(
            width=self.width,
            height=self.height,
            nets=nets,
            obstacles=obstacles,
            edge_capacity=self.edge_capacity,
        )

    def generate_batch(self, batch_size: int) -> List[ChipRoutingInstance]:
        return [self.generate() for _ in range(int(batch_size))]


class GridRoutingEnv:
    """A* based evaluator used by the net-ordering RL policy."""

    name = "chip_routing"

    _DIRECTIONS: Tuple[GridPoint, ...] = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    )

    def __init__(
        self,
        generator: ChipRoutingGenerator,
        congestion_weight: float = 2.0,
        overflow_weight: float = 12.0,
        bend_weight: float = 0.25,
        unrouted_penalty: float = 100.0,
    ):
        self.generator = generator
        self.congestion_weight = float(congestion_weight)
        self.overflow_weight = float(overflow_weight)
        self.bend_weight = float(bend_weight)
        self.unrouted_penalty = float(unrouted_penalty)

    def generate(self, seed: Optional[int] = None) -> ChipRoutingInstance:
        return self.generator.generate(seed=seed)

    def generate_batch(self, batch_size: int) -> List[ChipRoutingInstance]:
        return self.generator.generate_batch(batch_size)

    @staticmethod
    def _edge_usage(
        horizontal_usage: np.ndarray,
        vertical_usage: np.ndarray,
        current: GridPoint,
        nxt: GridPoint,
    ) -> int:
        x1, y1 = current
        x2, y2 = nxt
        if y1 == y2:
            return int(horizontal_usage[y1, min(x1, x2)])
        return int(vertical_usage[min(y1, y2), x1])

    @staticmethod
    def _increment_edge(
        horizontal_usage: np.ndarray,
        vertical_usage: np.ndarray,
        current: GridPoint,
        nxt: GridPoint,
    ) -> None:
        x1, y1 = current
        x2, y2 = nxt
        if y1 == y2:
            horizontal_usage[y1, min(x1, x2)] += 1
        else:
            vertical_usage[min(y1, y2), x1] += 1

    def _route_net(
        self,
        instance: ChipRoutingInstance,
        source: GridPoint,
        target: GridPoint,
        horizontal_usage: np.ndarray,
        vertical_usage: np.ndarray,
    ) -> Optional[GridPath]:
        """Route one net with A*, including direction in state for bend cost."""
        counter = itertools.count()
        start_state = (source[0], source[1], -1)
        frontier = [(0.0, next(counter), start_state)]
        costs = {start_state: 0.0}
        parents: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        final_state: Optional[Tuple[int, int, int]] = None

        while frontier:
            _, _, state = heapq.heappop(frontier)
            x, y, previous_direction = state
            current_cost = costs[state]
            if (x, y) == target:
                final_state = state
                break

            for direction, (dx, dy) in enumerate(self._DIRECTIONS):
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= instance.width or ny < 0 or ny >= instance.height:
                    continue
                if instance.obstacles[ny, nx] and (nx, ny) != target:
                    continue

                usage = self._edge_usage(
                    horizontal_usage,
                    vertical_usage,
                    (x, y),
                    (nx, ny),
                )
                projected_overflow = max(0, usage + 1 - instance.edge_capacity)
                bend_cost = (
                    self.bend_weight
                    if previous_direction >= 0 and previous_direction != direction
                    else 0.0
                )
                step_cost = (
                    1.0
                    + self.congestion_weight * (usage / instance.edge_capacity)
                    + self.overflow_weight * projected_overflow
                    + bend_cost
                )
                next_state = (nx, ny, direction)
                candidate_cost = current_cost + step_cost
                if candidate_cost >= costs.get(next_state, float("inf")):
                    continue

                costs[next_state] = candidate_cost
                parents[next_state] = state
                heuristic = abs(target[0] - nx) + abs(target[1] - ny)
                heapq.heappush(
                    frontier,
                    (candidate_cost + heuristic, next(counter), next_state),
                )

        if final_state is None:
            return None

        path: GridPath = []
        state = final_state
        while True:
            path.append((state[0], state[1]))
            if state == start_state:
                break
            state = parents[state]
        path.reverse()
        return path

    @staticmethod
    def _count_bends(path: Sequence[GridPoint]) -> int:
        bends = 0
        previous_direction = None
        for start, end in zip(path, path[1:]):
            direction = (end[0] - start[0], end[1] - start[1])
            if previous_direction is not None and direction != previous_direction:
                bends += 1
            previous_direction = direction
        return bends

    def route(
        self,
        instance: ChipRoutingInstance,
        order: Optional[Iterable[int]] = None,
    ) -> ChipRoutingSolution:
        """Route every net and evaluate the resulting solution."""
        route_order = (
            [int(net_id) for net_id in order]
            if order is not None
            else list(range(instance.num_nets))
        )
        if sorted(route_order) != list(range(instance.num_nets)):
            raise ValueError("order must contain every net exactly once")

        horizontal_usage = np.zeros(
            (instance.height, instance.width - 1),
            dtype=np.int16,
        )
        vertical_usage = np.zeros(
            (instance.height - 1, instance.width),
            dtype=np.int16,
        )
        paths: Dict[int, GridPath] = {}
        unrouted_nets: List[int] = []
        wirelength = 0
        bend_count = 0

        for net_id in route_order:
            sx, sy, tx, ty = [int(value) for value in instance.nets[net_id]]
            path = self._route_net(
                instance,
                (sx, sy),
                (tx, ty),
                horizontal_usage,
                vertical_usage,
            )
            if path is None:
                unrouted_nets.append(net_id)
                continue

            paths[net_id] = path
            wirelength += len(path) - 1
            bend_count += self._count_bends(path)
            for current, nxt in zip(path, path[1:]):
                self._increment_edge(horizontal_usage, vertical_usage, current, nxt)

        horizontal_overflow = np.maximum(
            horizontal_usage - instance.edge_capacity,
            0,
        )
        vertical_overflow = np.maximum(
            vertical_usage - instance.edge_capacity,
            0,
        )
        overflow = int(horizontal_overflow.sum() + vertical_overflow.sum())
        max_congestion = int(max(horizontal_usage.max(), vertical_usage.max()))
        objective = (
            float(wirelength)
            + self.bend_weight * bend_count
            + self.overflow_weight * overflow
            + self.unrouted_penalty * len(unrouted_nets)
        )

        return ChipRoutingSolution(
            order=route_order,
            paths=paths,
            wirelength=wirelength,
            bend_count=bend_count,
            overflow=overflow,
            max_congestion=max_congestion,
            unrouted_nets=unrouted_nets,
            objective=objective,
            horizontal_usage=horizontal_usage,
            vertical_usage=vertical_usage,
        )

