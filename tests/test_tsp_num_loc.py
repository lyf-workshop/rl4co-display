"""TSP 节点数量配置链路测试。"""

from queue import Queue

import pytest

from app_training import _normalize_tsp_num_loc
from modules.rl_training.tsp_trainer import TSPTrainer


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (5, 5),
        ("100", 100),
        (1000, 1000),
    ],
)
def test_normalize_tsp_num_loc_accepts_supported_range(raw_value, expected):
    assert _normalize_tsp_num_loc({"num_loc": raw_value}) == expected


@pytest.mark.parametrize("raw_value", [None, "", "abc", True, 5.5, 4, 1001])
def test_normalize_tsp_num_loc_rejects_invalid_values(raw_value):
    with pytest.raises(ValueError):
        _normalize_tsp_num_loc({"num_loc": raw_value})


def test_tsp_trainer_uses_configured_num_loc(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "modules.rl_training.base_trainer.get_user_plot_dir",
        lambda _user_id: str(tmp_path / "plots"),
    )
    monkeypatch.setattr(
        "modules.rl_training.base_trainer.get_user_checkpoint_dir",
        lambda _user_id: str(tmp_path / "checkpoints"),
    )

    trainer = TSPTrainer(
        config={
            "problem": "tsp",
            "model": "attention",
            "algorithm": "reinforce",
            "num_loc": 100,
            "batch_size": 4,
        },
        session_id="test-num-loc",
        user_id=1,
        queue=Queue(),
        training_status={},
        get_background_db_func=lambda: None,
    )

    env = trainer.initialize_environment()

    assert trainer.num_loc == 100
    assert env.generator.num_loc == 100
