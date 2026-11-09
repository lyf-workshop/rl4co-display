"""Regression tests for scalable CVRP and VRPTW route animations."""

import json
import queue

import numpy as np
import torch
from PIL import Image

from modules.rl_training.cvrp_trainer import (
    CVRPTrainer,
    _get_cvrp_animation_instance_limit,
)
from modules.rl_training.visualizations.common import (
    build_closed_depot_route,
    select_animation_steps,
)
from modules.rl_training.visualizations.cvrp_viz import (
    _prepare_cvrp_route,
    create_cvrp_route_animation,
)
from modules.rl_training.visualizations.vrptw_viz import (
    _prepare_vrptw_route,
    create_vrptw_route_animation,
)


def test_vrp_animation_steps_are_capped_and_include_endpoints():
    steps = select_animation_steps(1000, max_frames=100)

    assert len(steps) == 100
    assert steps[0] == 0
    assert steps[-1] == 1000
    assert np.all(np.diff(steps) > 0)


def test_vrp_route_is_closed_and_consecutive_depot_padding_is_removed():
    route = build_closed_depot_route(
        np.array([1, 0, 0, 2, 0, 0]),
        num_nodes=3,
    )

    assert np.array_equal(route, [0, 1, 0, 2, 0])


def test_cvrp_metrics_start_at_depot_and_reset_load_on_return():
    locs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    _, route, demands, _, distances, loads = _prepare_cvrp_route(
        locs,
        actions=np.array([1, 0, 2]),
        demands=np.array([0.2, 0.3]),
    )

    assert np.array_equal(route, [0, 1, 0, 2, 0])
    assert np.allclose(demands, [0.0, 0.2, 0.3])
    assert np.allclose(distances, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.allclose(loads, [0.0, 0.2, 0.0, 0.3, 0.0])


def test_vrptw_metrics_include_waiting_service_and_depot_returns():
    locs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    result = _prepare_vrptw_route(
        locs,
        actions=np.array([1, 0, 2]),
        demands=np.array([0.2, 0.3]),
        time_windows=np.array([[2.0, 5.0], [8.0, 12.0]]),
        service_times=np.array([1.0, 2.0]),
    )
    route = result[1]
    distances = result[5]
    loads = result[6]
    elapsed_times = result[7]
    visits = result[8]

    assert np.array_equal(route, [0, 1, 0, 2, 0])
    assert np.allclose(distances, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.allclose(loads, [0.0, 0.2, 0.0, 0.3, 0.0])
    assert np.allclose(elapsed_times, [0.0, 3.0, 4.0, 10.0, 11.0])
    assert visits == [
        (1, 0, 1.0, 2.0, 5.0),
        (3, 1, 5.0, 8.0, 12.0),
    ]


def test_large_cvrp_limits_only_animation_instances():
    assert _get_cvrp_animation_instance_limit(99, 3) == 3
    assert _get_cvrp_animation_instance_limit(100, 3) == 1
    assert _get_cvrp_animation_instance_limit(1000, 3) == 1


def test_large_random_cvrp_keeps_three_plots_and_one_animation(monkeypatch):
    num_customers = 100

    class FakeBatch:
        def __init__(self):
            self.locs = torch.rand(3, num_customers + 1, 2)
            self.demands = torch.full((3, num_customers), 0.01)

        def __getitem__(self, key):
            if key == "locs":
                return self.locs
            if key == "demand":
                return self.demands
            raise KeyError(key)

        def __iter__(self):
            return iter(
                {"locs": locs, "demand": demands}
                for locs, demands in zip(self.locs, self.demands)
            )

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

    actions = torch.arange(1, num_customers + 1).unsqueeze(0).expand(3, -1)
    policy_output = {
        "actions": actions,
        "reward": torch.tensor([-1.0, -2.0, -3.0]),
    }
    plot_calls = []
    animation_calls = []

    monkeypatch.setattr(
        "modules.rl_training.cvrp_trainer.create_cvrp_comparison_plot",
        lambda *_args, **_kwargs: plot_calls.append(True),
    )
    monkeypatch.setattr(
        "modules.rl_training.cvrp_trainer.create_cvrp_route_animation",
        lambda *_args, **kwargs: animation_calls.append(kwargs),
    )

    trainer = CVRPTrainer.__new__(CVRPTrainer)
    trainer.custom_dataset_data = None
    trainer.session_id = "12345678-cvrp"
    trainer.user_id = 7
    trainer.user_plots_dir = "unused"
    trainer.training_status = {trainer.session_id: {"plot_url": "/curve.png"}}
    trainer.queue = queue.SimpleQueue()
    trainer._save_file_record = lambda *_args: None
    trainer.create_untrained_policy_copy = lambda model: model.policy
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


def test_large_cvrp_animation_renders_sampled_frames(tmp_path):
    num_customers = 120
    angles = np.linspace(0, 2 * np.pi, num_customers, endpoint=False)
    customers = np.column_stack(
        (
            0.5 + 0.45 * np.cos(angles),
            0.5 + 0.45 * np.sin(angles),
        )
    ).astype(np.float32)
    locs = np.vstack((np.array([[0.5, 0.5]], dtype=np.float32), customers))
    progress_updates = []
    save_path = tmp_path / "cvrp-120.gif"

    create_cvrp_route_animation(
        {"locs": locs, "demand": np.full(num_customers, 0.01)},
        np.arange(1, num_customers + 1),
        save_path,
        max_frames=6,
        progress_callback=lambda percent, step, total: progress_updates.append(
            (percent, step, total)
        ),
    )

    with Image.open(save_path) as animation:
        assert animation.n_frames == 6
        assert animation.size == (800, 600)
        first_frame = np.asarray(animation.convert("RGB"))
        animation.seek(animation.n_frames - 1)
        last_frame = np.asarray(animation.convert("RGB"))
        assert first_frame.std() > 5
        assert not np.array_equal(first_frame, last_frame)

    assert progress_updates[0] == (0, 0, num_customers + 1)
    assert progress_updates[-1] == (100, num_customers + 1, num_customers + 1)


def test_vrptw_animation_renders_sampled_frames_without_temp_pngs(tmp_path):
    num_customers = 40
    angles = np.linspace(0, 2 * np.pi, num_customers, endpoint=False)
    customers = np.column_stack(
        (
            0.5 + 0.45 * np.cos(angles),
            0.5 + 0.45 * np.sin(angles),
        )
    ).astype(np.float32)
    locs = np.vstack((np.array([[0.5, 0.5]], dtype=np.float32), customers))
    starts = np.linspace(0.0, 20.0, num_customers)
    time_windows = np.column_stack((starts, starts + 30.0))
    progress_updates = []
    save_path = tmp_path / "vrptw-40.gif"

    create_vrptw_route_animation(
        {
            "locs": locs,
            "demand": np.full(num_customers, 0.01),
            "time_windows": time_windows,
            "service_time": np.full(num_customers, 0.1),
        },
        np.arange(1, num_customers + 1),
        save_path,
        max_frames=6,
        progress_callback=lambda percent, step, total: progress_updates.append(
            (percent, step, total)
        ),
    )

    with Image.open(save_path) as animation:
        assert animation.n_frames == 6
        assert animation.size == (1200, 600)
        first_frame = np.asarray(animation.convert("RGB"))
        animation.seek(animation.n_frames - 1)
        last_frame = np.asarray(animation.convert("RGB"))
        assert first_frame.std() > 5
        assert not np.array_equal(first_frame, last_frame)

    assert list(tmp_path.glob("vrptw-40_frame_*.png")) == []
    assert progress_updates[0] == (0, 0, num_customers + 1)
    assert progress_updates[-1] == (100, num_customers + 1, num_customers + 1)
