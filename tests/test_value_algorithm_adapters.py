"""DQN 与 Q-Learning 训练适配器测试。"""

import pytest

from modules.algorithms import (
    DQNAlgorithm,
    QLearningAlgorithm,
    get_algorithm_class,
    list_available_algorithms,
)
from modules.compatibility import validate_combination


@pytest.mark.parametrize(
    ("name", "expected_class"),
    [
        ("dqn", DQNAlgorithm),
        ("qlearning", QLearningAlgorithm),
    ],
)
def test_value_algorithm_registered(name, expected_class):
    assert get_algorithm_class(name) is expected_class


def test_value_algorithms_listed_as_available():
    names = {name for name, _status, _cn_name in list_available_algorithms()}
    assert {"dqn", "qlearning"} <= names


@pytest.mark.parametrize("name", ["dqn", "qlearning"])
def test_value_algorithm_accepts_tsp_attention(name):
    valid, _message, level = validate_combination("tsp", "attention", name)
    assert valid is True
    assert level == "success"


@pytest.mark.parametrize("name", ["dqn", "qlearning"])
def test_value_algorithm_rejects_other_combinations(name):
    assert validate_combination("cvrp", "attention", name)[0] is False
    assert validate_combination("tsp", "pomo", name)[0] is False


@pytest.mark.parametrize(
    ("name", "expected_name"),
    [
        ("dqn", "dqn"),
        ("qlearning", "qlearning"),
    ],
)
def test_value_algorithm_creates_trainable_model(name, expected_name):
    from rl4co.envs import TSPEnv
    from rl4co.models import AttentionModelPolicy, REINFORCE

    env = TSPEnv(generator_params={"num_loc": 5})
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
    )
    algorithm = get_algorithm_class(name)(
        {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "train_data_size": 8,
            "val_data_size": 4,
        }
    )

    model = algorithm.create_model(env, policy)

    assert algorithm.algorithm_name == expected_name
    assert isinstance(model, REINFORCE)
