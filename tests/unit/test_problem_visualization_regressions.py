import json
import queue
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("rl4co")

from rl4co.envs import CVRPEnv, MTSPEnv, SDVRPEnv
from rl4co.envs.routing import OPEnv, PCTSPEnv, PDPEnv
from rl4co.models import AttentionModelPolicy

from modules.envs.vrptw_env_wrapper import CVRPEnvWithTimeWindows
from modules.rl_training.atsp_trainer import ATSPTrainer
from modules.rl_training.base_trainer import BaseTrainer
from modules.rl_training.cvrp_trainer import CVRPTrainer
from modules.rl_training.mtsp_trainer import MTSPTrainer
from modules.rl_training.op_trainer import OPTrainer
from modules.rl_training.pctsp_trainer import PCTSPTrainer
from modules.rl_training.pdp_trainer import PDPTrainer
from modules.rl_training.sdvrp_trainer import SDVRPTrainer
from modules.rl_training.vrptw_trainer import VRPTWTrainer
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


def test_pctsp_uploaded_dataset_overrides_form_environment_size(tmp_path, monkeypatch):
    dataset_id = "pctsp-15"
    coordinates = [
        [index / 20.0, (index + 1) / 20.0]
        for index in range(15)
    ]
    dataset_dir = tmp_path / "datasets" / "user_7"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / f"{dataset_id}.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "filename": "pctsp_15nodes.json",
                "problem_type": "pctsp",
                "coordinates": coordinates,
                "depot": [0.5, 0.5],
                "prizes": [0.2] * 15,
                "penalties": [0.3] * 15,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    trainer = PCTSPTrainer(
        config={
            "problem": "pctsp",
            "num_loc": 20,
            "dataset_mode": "upload",
            "dataset_id": dataset_id,
            "gpu_id": None,
        },
        session_id="pctsp-upload-size",
        user_id=7,
        queue=queue.SimpleQueue(),
        training_status={},
        get_background_db_func=lambda: None,
    )

    assert trainer.num_loc == 15
    assert trainer.pctsp_num_loc == 15

    env = trainer.initialize_environment()
    td = env.reset(batch_size=[1])
    assert td["locs"].shape == (1, 16, 2)

    injected = trainer._inject_custom_data(td)
    assert injected["locs"].shape == (1, 16, 2)
    assert torch.allclose(
        injected["locs"][0, 1:],
        torch.tensor(coordinates, dtype=torch.float32),
    )

def test_op_uploaded_dataset_overrides_form_environment_size(tmp_path, monkeypatch):
    dataset_id = "op-15"
    coordinates = [[index / 20.0, (index + 1) / 20.0] for index in range(15)]
    dataset_dir = tmp_path / "datasets" / "user_7"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / f"{dataset_id}.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "filename": "op_15nodes.json",
                "problem_type": "op",
                "coordinates": coordinates,
                "depot": [0.5, 0.5],
                "prizes": [0.2] * 15,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    trainer = OPTrainer(
        config={
            "problem": "op",
            "num_loc": 20,
            "dataset_mode": "upload",
            "dataset_id": dataset_id,
            "gpu_id": None,
        },
        session_id="op-upload-size",
        user_id=7,
        queue=queue.SimpleQueue(),
        training_status={},
        get_background_db_func=lambda: None,
    )

    assert trainer.num_loc == 15
    assert trainer.op_num_loc == 15
    env = trainer.initialize_environment()
    td = trainer._inject_custom_data(env.reset(batch_size=[1]), env=env)
    assert td["locs"].shape == (1, 16, 2)
    assert td["max_length"].shape == (1, 16)


def test_vrptw_uploaded_dataset_overrides_form_environment_size(tmp_path, monkeypatch):
    dataset_id = "vrptw-3"
    dataset_dir = tmp_path / "datasets" / "user_7"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / f"{dataset_id}.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "filename": "vrptw_3nodes.json",
                "problem_type": "vrptw",
                "coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                "depot": [0.5, 0.5],
                "demands": [0.2, 0.3, 0.4],
                "time_windows": [[10.0, 60.0], [20.0, 80.0], [30.0, 90.0]],
                "service_times": [5.0, 6.0, 7.0],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    trainer = VRPTWTrainer(
        config={
            "problem": "vrptw",
            "num_loc": 20,
            "dataset_mode": "upload",
            "dataset_id": dataset_id,
            "gpu_id": None,
        },
        session_id="vrptw-upload-size",
        user_id=7,
        queue=queue.SimpleQueue(),
        training_status={},
        get_background_db_func=lambda: None,
    )

    assert trainer.num_loc == 3
    env = trainer.initialize_environment()
    td = trainer._inject_custom_data(env.reset(batch_size=[1]), env=env)
    assert td["locs"].shape == (1, 4, 2)
    assert td["time_windows"].shape == (1, 4, 2)


def _inference_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _assert_tensordict_device(td, device):
    assert td.device == device
    for value in td.values():
        if isinstance(value, torch.Tensor):
            assert value.device == device


def _run_attention_inference(env, td, device):
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
    ).to(device)
    policy.eval()
    with torch.no_grad():
        return policy(
            td.clone(),
            phase="test",
            decode_type="greedy",
            return_actions=True,
        )


def test_op_custom_dataset_inference_stays_on_model_device():
    device = _inference_device()
    env = OPEnv(generator_params={"num_loc": 3, "max_length": 2.0})
    td = env.reset(batch_size=[2]).to(device)

    trainer = OPTrainer.__new__(OPTrainer)
    trainer.max_length = 2.0
    trainer.custom_dataset_data = {
        "coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "depot": [0.9, 0.8],
        "prizes": [0.2, 0.4, 0.6],
    }
    td = trainer._inject_custom_data(td, env=env)

    _assert_tensordict_device(td, device)
    assert td["locs"].shape == (2, 4, 2)
    assert td["prize"].shape == (2, 4)
    assert td["max_length"].shape == (2, 4)
    out = _run_attention_inference(env, td, device)
    assert out["actions"].device == device
    assert torch.isfinite(out["reward"]).all()


def test_cvrp_custom_dataset_inference_stays_on_model_device():
    device = _inference_device()
    env = CVRPEnv(generator_params={"num_loc": 3, "vehicle_capacity": 1.0})
    td = env.reset(batch_size=[2]).to(device)

    trainer = CVRPTrainer.__new__(CVRPTrainer)
    trainer.custom_dataset_data = {
        "coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "depot": [0.9, 0.8],
        "demands": [0.2, 0.3, 0.4],
    }
    td = trainer._inject_custom_data(td, env=env)

    _assert_tensordict_device(td, device)
    assert torch.allclose(td["demand"][0], torch.tensor([0.2, 0.3, 0.4], device=device))
    out = _run_attention_inference(env, td, device)
    assert out["actions"].device == device
    assert torch.isfinite(out["reward"]).all()


def test_sdvrp_custom_dataset_updates_demand_with_depot():
    device = _inference_device()
    env = SDVRPEnv(generator_params={"num_loc": 3, "vehicle_capacity": 1.0})
    td = env.reset(batch_size=[2]).to(device)

    trainer = SDVRPTrainer.__new__(SDVRPTrainer)
    trainer.custom_dataset_data = {
        "coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "depot": [0.9, 0.8],
        "demands": [0.2, 0.3, 0.4],
    }
    td = trainer._inject_custom_data(td, env=env)

    _assert_tensordict_device(td, device)
    expected = torch.tensor([0.0, 0.2, 0.3, 0.4], device=device)
    assert torch.allclose(td["demand_with_depot"][0], expected)
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
    ).to(device)
    policy.eval()
    with torch.no_grad():
        out = trainer._run_visualization_policy(
            policy,
            td.clone(),
            env,
            phase="test",
            decode_type="greedy",
        )
    assert out["actions"].device == device
    assert torch.equal(out["actions"][..., -1], torch.zeros(2, dtype=torch.long, device=device))
    assert torch.isfinite(out["reward"]).all()


def test_pdp_custom_dataset_keeps_depot_and_runs_inference():
    device = _inference_device()
    env = PDPEnv(generator_params={"num_loc": 4})
    td = env.reset(batch_size=[2]).to(device)

    trainer = PDPTrainer.__new__(PDPTrainer)
    trainer.custom_dataset_data = {
        "coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
        "depot": [0.9, 0.8],
    }
    td = trainer._inject_custom_data(td)

    _assert_tensordict_device(td, device)
    assert td["locs"].shape == (2, 5, 2)
    assert torch.allclose(td["locs"][:, 0], torch.tensor([[0.9, 0.8], [0.9, 0.8]], device=device))
    out = _run_attention_inference(env, td, device)
    assert out["actions"].device == device
    assert torch.isfinite(out["reward"]).all()


def test_vrptw_custom_dataset_updates_all_uploaded_fields():
    device = _inference_device()
    env = CVRPEnvWithTimeWindows(
        {"num_loc": 3, "vehicle_capacity": 1.0},
        {
            "time_window_width": 50.0,
            "service_time": 10.0,
            "max_time": 480.0,
            "hard_time_windows": True,
        },
    )
    td = env.reset(batch_size=[2]).to(device)

    trainer = VRPTWTrainer.__new__(VRPTWTrainer)
    trainer.max_time = 480.0
    trainer.custom_dataset_data = {
        "coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "depot": [0.9, 0.8],
        "demands": [0.2, 0.3, 0.4],
        "time_windows": [[10.0, 60.0], [20.0, 80.0], [30.0, 90.0]],
        "service_times": [5.0, 6.0, 7.0],
    }
    td = trainer._inject_custom_data(td, env=env)

    _assert_tensordict_device(td, device)
    assert td["locs"].shape == (2, 4, 2)
    assert td["demand"].shape == (2, 3)
    assert td["time_windows"].shape == (2, 4, 2)
    assert td["service_time"].shape == (2, 4)
    assert torch.allclose(td["service_time"][0], torch.tensor([0.0, 5.0, 6.0, 7.0], device=device))


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
