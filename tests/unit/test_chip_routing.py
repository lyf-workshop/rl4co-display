from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from app_training import _normalize_chip_routing_config
from modules.compatibility import (
    get_available_algorithms,
    get_available_policies,
    validate_combination,
)
from modules.envs.chip_routing_env import (
    ChipRoutingGenerator,
    GridRoutingEnv,
)
from modules.policies.chip_routing_policy import (
    NetOrderingPolicy,
    build_feature_batch,
)
from modules.problems.chip_routing import ChipRoutingProblem
from modules.rl_training.visualizations.chip_routing_viz import (
    create_chip_routing_animation,
    create_chip_routing_comparison,
)


def _make_env(seed=7, num_nets=6):
    generator = ChipRoutingGenerator(
        width=12,
        height=10,
        num_nets=num_nets,
        obstacle_ratio=0.10,
        edge_capacity=1,
        seed=seed,
    )
    return GridRoutingEnv(generator)


def test_generator_is_reproducible_and_keeps_pins_out_of_obstacles():
    first = _make_env().generate(seed=123)
    second = _make_env().generate(seed=123)

    assert np.array_equal(first.nets, second.nets)
    assert np.array_equal(first.obstacles, second.obstacles)
    assert first.nets.shape == (6, 4)

    endpoints = set()
    for sx, sy, tx, ty in first.nets:
        assert not first.obstacles[sy, sx]
        assert not first.obstacles[ty, tx]
        endpoints.add((int(sx), int(sy)))
        endpoints.add((int(tx), int(ty)))
    assert len(endpoints) == 12


def test_router_returns_connected_obstacle_free_paths_and_consistent_metrics():
    env = _make_env()
    instance = env.generate(seed=42)
    solution = env.route(instance, reversed(range(instance.num_nets)))

    assert solution.routed_count + len(solution.unrouted_nets) == instance.num_nets
    assert solution.objective >= solution.wirelength
    assert solution.overflow >= 0

    for net_id, path in solution.paths.items():
        sx, sy, tx, ty = instance.nets[net_id]
        assert path[0] == (sx, sy)
        assert path[-1] == (tx, ty)
        for point, next_point in zip(path, path[1:]):
            assert abs(point[0] - next_point[0]) + abs(point[1] - next_point[1]) == 1
        assert all(not instance.obstacles[y, x] for x, y in path)


def test_attention_policy_outputs_permutations_and_supports_reinforce_gradient():
    env = _make_env(seed=2)
    instances = env.generate_batch(4)
    features = build_feature_batch(instances, torch.device("cpu"))
    policy = NetOrderingPolicy(embed_dim=32, num_heads=4, num_layers=1)

    output = policy(features, decode_type="sampling")
    assert output["order"].shape == (4, 6)
    assert output["log_likelihood"].shape == (4,)
    assert output["entropy"].shape == (4,)
    for order in output["order"].tolist():
        assert sorted(order) == list(range(6))

    rewards = torch.tensor([-1.0, -0.8, -1.2, -0.6])
    advantages = rewards - rewards.mean()
    loss = -(advantages * output["log_likelihood"]).mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )


def test_chip_routing_problem_and_compatibility_contract():
    problem = ChipRoutingProblem(
        {
            "grid_width": 18,
            "grid_height": 16,
            "num_nets": 10,
            "obstacle_ratio": 0.12,
            "edge_capacity": 2,
        }
    )
    assert problem.validate_config() == (True, "")
    assert problem.get_problem_type() == "chip_routing"
    assert "attention" in get_available_policies("chip_routing")
    assert get_available_algorithms("chip_routing") == ["reinforce"]
    assert validate_combination(
        "chip_routing",
        "attention",
        "reinforce",
    )[0] is True
    assert validate_combination(
        "chip_routing",
        "attention",
        "ppo",
    )[0] is False


def test_chip_routing_request_normalization():
    normalized = _normalize_chip_routing_config(
        {
            "grid_width": "20",
            "grid_height": 14,
            "num_nets": "9",
            "obstacle_ratio": "0.16",
            "edge_capacity": 2,
        }
    )
    assert normalized == {
        "grid_width": 20,
        "grid_height": 14,
        "num_nets": 9,
        "edge_capacity": 2,
        "obstacle_ratio": 0.16,
        "num_loc": 9,
    }

    with pytest.raises(ValueError, match="网络数量"):
        _normalize_chip_routing_config({"num_nets": 41})


def test_chip_routing_visualizations_are_nonblank(tmp_path: Path):
    env = _make_env(seed=11, num_nets=5)
    instance = env.generate(seed=99)
    baseline = env.route(instance)
    trained = env.route(instance, reversed(range(instance.num_nets)))
    comparison_path = tmp_path / "comparison.png"
    animation_path = tmp_path / "routing.gif"

    create_chip_routing_comparison(
        instance,
        baseline,
        trained,
        comparison_path,
    )
    create_chip_routing_animation(
        instance,
        trained,
        animation_path,
        fps=4,
    )

    assert comparison_path.exists() and comparison_path.stat().st_size > 10_000
    assert animation_path.exists() and animation_path.stat().st_size > 10_000
    with Image.open(comparison_path) as image:
        assert image.width > 1000
        assert image.height > 300
        assert any(low != high for low, high in image.convert("RGB").getextrema())
    with Image.open(animation_path) as animation:
        assert animation.n_frames == instance.num_nets + 1

