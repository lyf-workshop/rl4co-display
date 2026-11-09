"""Regression tests for scalable TSP route animation generation."""

import json
import queue

import numpy as np
import torch
from PIL import Image

from modules.rl_training.tsp_trainer import (
    TSPTrainer,
    _get_animation_instance_limit,
)
from modules.rl_training.visualizations.tsp_viz import (
    _calculate_cumulative_route_costs,
    _select_animation_steps,
    create_tsp_route_animation,
)


def test_large_animation_steps_are_capped_and_include_endpoints():
    steps = _select_animation_steps(1000, max_frames=100)

    assert len(steps) == 100
    assert steps[0] == 0
    assert steps[-1] == 1000
    assert np.all(np.diff(steps) > 0)


def test_cumulative_route_cost_closes_route_once():
    locs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    actions = np.arange(4)

    route_locs, costs = _calculate_cumulative_route_costs(locs, actions)

    assert np.array_equal(route_locs, locs)
    assert np.allclose(costs, [0.0, 0.0, 1.0, 2.0, 4.0])


def test_large_problem_limits_only_animation_instances():
    assert _get_animation_instance_limit(499, 3) == 3
    assert _get_animation_instance_limit(500, 3) == 1
    assert _get_animation_instance_limit(1000, 3) == 1


def test_large_random_problem_keeps_three_plots_and_one_animation(monkeypatch):
    num_cities = 1000

    class FakeBatch:
        def __init__(self):
            self.locs = torch.rand(3, num_cities, 2)

        def __getitem__(self, key):
            if key == "locs":
                return self.locs
            raise KeyError(key)

        def __iter__(self):
            return iter({"locs": locs} for locs in self.locs)

        def __len__(self):
            return len(self.locs)

        def clone(self):
            return self

        def to(self, _device):
            return self

    class FakeEnv:
        def reset(self, batch_size):
            assert batch_size == [3]
            return FakeBatch()

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = FakePolicy()

    class FakeLightningTrainer:
        def __init__(self):
            self.saved_paths = []

        def save_checkpoint(self, path):
            self.saved_paths.append(path)

    actions = torch.arange(num_cities).unsqueeze(0).expand(3, -1)
    policy_output = {
        "actions": actions,
        "reward": torch.tensor([-1.0, -2.0, -3.0]),
    }
    plot_calls = []
    animation_calls = []

    monkeypatch.setattr(
        "modules.rl_training.tsp_trainer.create_tsp_comparison_plot",
        lambda *_args, **_kwargs: plot_calls.append(True),
    )
    monkeypatch.setattr(
        "modules.rl_training.tsp_trainer.create_tsp_route_animation",
        lambda *_args, **kwargs: animation_calls.append(kwargs),
    )

    trainer = TSPTrainer.__new__(TSPTrainer)
    trainer.custom_dataset_data = None
    trainer.session_id = "12345678-tsp"
    trainer.user_id = 7
    trainer.user_plots_dir = "unused"
    trainer.training_status = {trainer.session_id: {"plot_url": "/curve.png"}}
    trainer.queue = queue.SimpleQueue()
    trainer._save_file_record = lambda *_args: None
    trainer.create_untrained_policy_copy = lambda _model: _model.policy
    trainer._run_policy = lambda *_args, **_kwargs: policy_output

    lightning_trainer = FakeLightningTrainer()
    result = trainer.generate_visualizations(
        FakeEnv(),
        FakeModel(),
        lightning_trainer,
        checkpoint_path="unused.ckpt",
    )

    messages = []
    while not trainer.queue.empty():
        messages.append(json.loads(trainer.queue.get())["message"])

    assert len(plot_calls) == 3
    assert len(animation_calls) == 1
    assert animation_calls[0]["fps"] == 4
    assert len(result["plot_paths"]) == 3
    assert len(result["animation_paths"]) == 1
    assert any("动态路线图仅生成第1组" in message for message in messages)
    assert lightning_trainer.saved_paths == ["unused.ckpt"]


def test_1000_node_animation_renders_sampled_frames(tmp_path):
    num_cities = 1000
    angles = np.linspace(0, 2 * np.pi, num_cities, endpoint=False)
    locs = np.column_stack(
        (
            0.5 + 0.45 * np.cos(angles),
            0.5 + 0.45 * np.sin(angles),
        )
    ).astype(np.float32)
    progress_updates = []
    save_path = tmp_path / "tsp-1000.gif"

    create_tsp_route_animation(
        {"locs": locs},
        np.arange(num_cities),
        save_path,
        max_frames=6,
        progress_callback=lambda percent, step, total: progress_updates.append(
            (percent, step, total)
        ),
    )

    with Image.open(save_path) as animation:
        assert animation.n_frames == 6
        assert animation.size == (800, 600)

    assert progress_updates[0] == (0, 0, num_cities)
    assert progress_updates[-1] == (100, num_cities, num_cities)
