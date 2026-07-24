import queue
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("rl4co")

from rl4co.envs import MTSPEnv
from rl4co.envs.routing import PCTSPEnv
from rl4co.models import AttentionModelPolicy

from modules.rl_training.atsp_trainer import ATSPTrainer
from modules.rl_training.base_trainer import BaseTrainer
from modules.rl_training.mtsp_trainer import MTSPTrainer
from modules.rl_training.pctsp_trainer import PCTSPTrainer
from modules.rl_training.visualizations.mtsp_viz import (
    create_mtsp_comparison_plot,
    extract_agent_routes,
)


def test_mtsp_custom_dataset_keeps_every_customer_and_supports_inference():
    customer_coords = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    env = MTSPEnv(
        generator_params={
            "num_loc": len(customer_coords) + 1,
            "min_num_agents": 2,
            "max_num_agents": 2,
        },
        cost_type="minmax",
    )
    td = env.reset(batch_size=[2])

    trainer = MTSPTrainer.__new__(MTSPTrainer)
    trainer.device = torch.device("cpu")
    trainer.custom_dataset_data = {
        "coordinates": customer_coords,
        "depot": [0.9, 0.8],
    }

    td = trainer._inject_custom_data(td)

    assert td["locs"].shape == (2, 4, 2)
    expected = torch.tensor([[0.9, 0.8], *customer_coords])
    assert torch.allclose(td["locs"][0], expected)
    assert torch.allclose(td["locs"][1], expected)

    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
    )
    policy.eval()
    with torch.no_grad():
        out = policy(
            td,
            phase="test",
            decode_type="greedy",
            return_actions=True,
        )

    assert out["actions"].shape[0] == 2
    assert int(out["actions"].max()) <= len(customer_coords)


def test_mtsp_visualization_reports_configured_and_inactive_agents():
    routes = extract_agent_routes([1, 2, 0, 3, 0], num_agents=4)
    assert routes == [[1, 2], [3], [], []]

    locs = torch.tensor(
        [
            [0.5, 0.5],
            [0.1, 0.2],
            [0.3, 0.8],
            [0.9, 0.4],
        ],
        dtype=torch.float32,
    )
    output_dir = Path(__file__).with_name("_visualization_output")
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "mtsp.png"
    create_mtsp_comparison_plot(
        td={"locs": locs},
        actions=np.array([1, 2, 0, 3, 0]),
        save_path=output,
        cost=1.25,
        num_agents=4,
    )

    assert output.exists()
    with Image.open(output) as image:
        assert image.width > 500
        assert image.height > 500
        assert any(lo != hi for lo, hi in image.convert("RGB").getextrema())
    output.unlink()
    output_dir.rmdir()


def test_pctsp_custom_dataset_updates_policy_and_reward_fields():
    env = PCTSPEnv(
        generator_params={
            "num_loc": 3,
            "penalty_factor": 3.0,
            "prize_required": 1.0,
        }
    )
    td = env.reset(batch_size=[2])

    trainer = PCTSPTrainer.__new__(PCTSPTrainer)
    trainer.custom_dataset_data = {
        "coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "depot": [0.9, 0.8],
        "prizes": [0.2, 0.4, 0.6],
        "penalties": [0.1, 0.3, 0.5],
    }

    td = trainer._inject_custom_data(td)

    expected_prizes = torch.tensor([[0.2, 0.4, 0.6]]).expand(2, -1)
    expected_full_prizes = torch.tensor([[0.0, 0.2, 0.4, 0.6]]).expand(2, -1)
    expected_penalties = torch.tensor([[0.0, 0.1, 0.3, 0.5]]).expand(2, -1)
    assert torch.allclose(td["expected_prize"], expected_prizes)
    assert torch.allclose(td["real_prize"], expected_full_prizes)
    assert torch.allclose(td["penalty"], expected_penalties)
    assert torch.allclose(td["cur_total_penalty"], torch.tensor([0.9, 0.9]))
    assert torch.allclose(
        td["locs"][:, 0],
        torch.tensor([[0.9, 0.8], [0.9, 0.8]]),
    )
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
    )
    policy.eval()
    with torch.no_grad():
        out = policy(
            td,
            phase="test",
            decode_type="greedy",
            return_actions=True,
        )

    assert out["actions"].shape[0] == 2
    assert torch.isfinite(out["reward"]).all()


def test_untrained_policy_copy_follows_current_model_device():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Linear(2, 2)

    model = DummyModel()
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.device = torch.device("meta")
    trainer.initial_policy_state_dict = {
        key: value.clone() for key, value in model.policy.state_dict().items()
    }

    copied_policy = trainer.create_untrained_policy_copy(model)

    assert next(copied_policy.parameters()).device == next(model.policy.parameters()).device

def test_atsp_random_visualization_uses_three_distinct_instances(monkeypatch):
    class FakeTensorDict(dict):
        def to(self, device):
            return FakeTensorDict(
                {key: value.to(device) for key, value in self.items()}
            )

        def clone(self):
            return FakeTensorDict(
                {key: value.clone() for key, value in self.items()}
            )

    class FakeEnv:
        def __init__(self):
            self.reset_calls = []

        def reset(self, batch_size):
            self.reset_calls.append(batch_size)
            base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            matrices = torch.stack([base + 100 * i for i in range(3)])
            return FakeTensorDict({"cost_matrix": matrices})

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, td, **kwargs):
            batch_size = td["cost_matrix"].shape[0]
            actions = torch.arange(4).unsqueeze(0).expand(batch_size, -1)
            return {"actions": actions}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = FakePolicy()

    class FakeLightningTrainer:
        def __init__(self):
            self.saved_paths = []

        def save_checkpoint(self, path):
            self.saved_paths.append(path)

    comparison_matrices = []
    animation_matrices = []

    def fake_comparison(cost_matrix, _before, _after, _path, **_kwargs):
        comparison_matrices.append(cost_matrix.clone())
        return {"cost_random": 10.0, "cost_trained": 8.0, "improvement": 20.0}

    def fake_animation(cost_matrix, _actions, _path, **_kwargs):
        animation_matrices.append(cost_matrix.clone())

    monkeypatch.setattr(
        "modules.rl_training.atsp_trainer.create_atsp_comparison_plot",
        fake_comparison,
    )
    monkeypatch.setattr(
        "modules.rl_training.atsp_trainer.create_atsp_route_animation",
        fake_animation,
    )

    env = FakeEnv()
    model = FakeModel()
    lightning_trainer = FakeLightningTrainer()
    trainer = ATSPTrainer.__new__(ATSPTrainer)
    trainer.device = torch.device("cpu")
    trainer.initial_policy_state_dict = {
        key: value.clone() for key, value in model.policy.state_dict().items()
    }
    trainer.session_id = "12345678-atsp"
    trainer.user_id = 7
    trainer.user_plots_dir = "unused"
    trainer.num_loc = 4
    trainer.training_status = {trainer.session_id: {"plot_url": "/curve.png"}}
    trainer.queue = queue.SimpleQueue()
    trainer.bg_file_manager = None

    result = trainer.generate_visualizations(
        env,
        model,
        lightning_trainer,
        checkpoint_path="unused.ckpt",
    )

    assert env.reset_calls == [[3]]
    assert len(comparison_matrices) == 3
    assert len(animation_matrices) == 3
    assert [float(matrix[0, 0]) for matrix in comparison_matrices] == [0.0, 100.0, 200.0]
    assert len(result["plot_paths"]) == 3
    assert len(result["animation_paths"]) == 3
    assert all(f"inst{i}" in result["plot_paths"][i - 1] for i in range(1, 4))
    assert all(f"inst{i}" in result["animation_paths"][i - 1] for i in range(1, 4))
    assert lightning_trainer.saved_paths == ["unused.ckpt"]
