from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_manager_actions_do_not_submit_training_form():
    script = (ROOT / "static/js/index.js").read_text(encoding="utf-8")
    template = (ROOT / "templates/index.html").read_text(encoding="utf-8")

    assert '<button type="button" onclick="selectDataset(' in script
    assert '<button type="button" onclick="deleteDataset(' in script
    assert '<button type="button" onclick="closeDatasetManager()"' in template
