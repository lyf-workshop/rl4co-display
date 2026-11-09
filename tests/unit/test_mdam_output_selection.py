import pytest
import torch

from modules.policies.mdam_policy import MDAMPolicyWrapper
from modules.policies.mdam_utils import select_best_mdam_path


def test_select_best_mdam_path_keeps_reward_and_actions_aligned():
    rewards = torch.tensor(
        [
            [-4.0, -2.0, -3.0],
            [-1.0, -6.0, -2.0],
        ]
    )
    log_likelihood = torch.tensor(
        [
            [-40.0, -20.0, -30.0],
            [-10.0, -60.0, -20.0],
        ]
    )
    actions_by_path = [
        torch.tensor([[0, 1, 2], [0, 2, 1]]),
        torch.tensor([[0, 2, 1], [0, 1, 2]]),
        torch.tensor([[0, 3, 1], [0, 3, 2]]),
    ]

    best_reward, best_ll, best_actions, best_indices = select_best_mdam_path(
        rewards, log_likelihood, actions_by_path
    )

    assert best_reward.tolist() == [-2.0, -1.0]
    assert best_ll.tolist() == [-20.0, -10.0]
    assert best_indices.tolist() == [1, 0]
    assert best_actions.tolist() == [[0, 2, 1], [0, 2, 1]]
    assert best_reward[0].item() == -2.0


def test_select_best_mdam_path_pads_variable_length_routes():
    rewards = torch.tensor([[-3.0, -1.0], [-1.0, -2.0]])
    log_likelihood = torch.tensor([[-30.0, -10.0], [-10.0, -20.0]])
    actions_by_path = [
        torch.tensor([[0, 1], [0, 2]]),
        torch.tensor([[0, 3, 2, 0], [0, 1, 3, 0]]),
    ]

    _, _, best_actions, _ = select_best_mdam_path(
        rewards, log_likelihood, actions_by_path
    )

    assert best_actions.tolist() == [[0, 3, 2, 0], [0, 2, 0, 0]]


@pytest.mark.parametrize(
    ("config_key", "value"),
    [("num_paths", 0), ("eg_step_gap", 0)],
)
def test_mdam_rejects_non_positive_runtime_parameters(config_key, value):
    config = {
        "embed_dim": 128,
        "num_encoder_layers": 3,
        "num_heads": 8,
        config_key: value,
    }

    valid, message = MDAMPolicyWrapper(config).validate_config()

    assert valid is False
    assert config_key in message


def test_mdam_exposes_runtime_parameters():
    wrapper = MDAMPolicyWrapper(
        {
            "embed_dim": 64,
            "num_encoder_layers": 2,
            "num_heads": 4,
            "num_paths": 3,
            "eg_step_gap": 100,
        }
    )

    assert wrapper.get_policy_params()["num_paths"] == 3
    assert wrapper.get_policy_params()["eg_step_gap"] == 100
