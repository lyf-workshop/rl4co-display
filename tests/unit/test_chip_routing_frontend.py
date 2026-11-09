from pathlib import Path
from queue import Queue

import modules.rl_training as training_module
from modules.compatibility import (
    ALL_INTEGRATED_PROBLEMS,
    get_recommended_combination,
    validate_combination,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_chip_routing_controls_and_request_fields_stay_in_sync():
    html = _read("templates/index.html")
    javascript = _read("static/js/training-progress.js")
    page_javascript = _read("static/js/index.js")

    assert 'value="chip_routing"' in html
    for control_id in (
        "chip-routing-params",
        "chip-grid-width",
        "chip-grid-height",
        "chip-num-nets",
        "chip-obstacle-ratio",
        "chip-edge-capacity",
    ):
        assert f'id="{control_id}"' in html

    assert "'chip-routing-params'" in page_javascript
    for control_id in (
        "chip-grid-width",
        "chip-grid-height",
        "chip-num-nets",
        "chip-obstacle-ratio",
        "chip-edge-capacity",
    ):
        assert f"'{control_id}'" in javascript
    for config_field in (
        "config.grid_width",
        "config.grid_height",
        "config.num_nets",
        "config.obstacle_ratio",
        "config.edge_capacity",
    ):
        assert config_field in javascript

    assert "config.model = 'attention'" in javascript
    assert "config.algorithm = 'reinforce'" in javascript
    assert "results.routing_metrics" in javascript


def test_problem_change_automatically_applies_best_recommendation():
    javascript = _read("static/js/index.js")
    problem_handler_start = javascript.index(
        "document.getElementById('problem-select').addEventListener"
    )
    problem_handler_end = javascript.index(
        "document.getElementById('model-select').addEventListener",
        problem_handler_start,
    )
    problem_handler = javascript[problem_handler_start:problem_handler_end]

    assert "async (e)" in problem_handler
    assert "await this.applyRecommendedConfig" in problem_handler
    assert "'best'" in problem_handler
    assert "{ automatic: true }" in problem_handler
    assert "requestedProblem === this.currentProblem" in javascript
    assert "this.currentProblem === selectedProblem" in problem_handler


def test_all_integrated_problems_have_valid_best_recommendations():
    for problem in ALL_INTEGRATED_PROBLEMS:
        recommended = get_recommended_combination(problem, "best")
        valid, message, _ = validate_combination(
            problem,
            recommended["policy"],
            recommended["algorithm"],
        )
        assert valid, f"{problem}: {message}"


def test_chip_routing_dispatch_reaches_the_dedicated_trainer(monkeypatch):
    calls = []

    def fake_train(*args):
        calls.append(args)

    monkeypatch.setattr(
        training_module,
        "_RL4CO_TRAINERS_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        training_module,
        "train_chip_routing",
        fake_train,
    )

    config = {"problem": "chip_routing"}
    queue = Queue()
    status = {}
    training_module.real_rl4co_training(
        config,
        "session-1",
        7,
        queue,
        status,
        lambda: None,
    )

    assert len(calls) == 1
    assert calls[0][0] is config
    assert calls[0][1:3] == ("session-1", 7)

