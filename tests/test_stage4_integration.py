"""Opt-in actual-data integration, blocked until original v1 is supplied.

Uses a temporary SQLite copy and temporary outputs, preserving the repository's
registry and reports. Set RUN_STAGE4_INTEGRATION=1 and configure the original
artifact directory before running these tests.
"""

import copy
import os
import sqlite3
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml

from src.monitoring import group_mae, load_v1_model
from src.retraining import finalize, train_candidate
from src.train import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_STAGE4_INTEGRATION") != "1",
        reason="Blocked: original registered v1 artifact missing; opt in after recovery with RUN_STAGE4_INTEGRATION=1",
    ),
]


@pytest.fixture
def actual_config(tmp_path):
    config = copy.deepcopy(yaml.safe_load((ROOT / "params.yaml").read_text())["monitoring"])
    with sqlite3.connect(f"file:{(ROOT / 'mlflow.db').as_posix()}?mode=ro", uri=True) as source:
        with sqlite3.connect(tmp_path / "mlflow.db") as destination:
            source.backup(destination)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    for key in ("train_path", "later_path", "holdout_path"):
        config[key] = str(ROOT / config[key])
    config["model"]["tracking_uri"] = uri
    findings = tmp_path / "findings.yaml"
    findings.write_text((ROOT / config["findings_path"]).read_text(), encoding="utf-8")
    config["findings_path"] = str(findings)
    config["predictions_path"] = str(tmp_path / "later_predictions.csv")
    settings = config["retraining"]
    settings["tracking_uri"] = uri
    settings["experiment_name"] = "stage4-actual-integration"
    settings["registered_model_name"] = "stage4-actual-integration"
    settings["artifact_root"] = str(tmp_path / "artifacts")
    for key in (
        "candidate_path",
        "candidate_report",
        "final_report",
        "champion_path",
        "holdout_predictions_path",
    ):
        settings[key] = str(tmp_path / Path(settings[key]).name)
    old_tracking, old_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    yield config
    mlflow.end_run()
    mlflow.set_tracking_uri(old_tracking)
    mlflow.set_registry_uri(old_registry)


def test_actual_original_v1_area_mae(actual_config):
    model = load_v1_model(actual_config["model"])
    later = pd.read_csv(actual_config["later_path"])
    predictions = model.predict(later[FEATURE_COLUMNS])
    maes = group_mae(later.energy_kwh, predictions, later.area)
    assert set(maes) == {"Old City", "New Colony", "Industrial"}
    assert len(predictions) == len(later)


def test_actual_full_stage4_and_holdout(actual_config):
    load_v1_model(actual_config["model"])
    candidate = train_candidate(actual_config)
    assert candidate["train_rows"] == 1000
    assert candidate["later_fit_rows"] == 210
    assert candidate["later_check_rows"] == 90
    results = finalize(actual_config)
    predictions = pd.read_csv(actual_config["retraining"]["holdout_predictions_path"])
    holdout = pd.read_csv(actual_config["holdout_path"])
    assert predictions.columns.tolist() == ["record_id", "prediction"]
    assert predictions.record_id.tolist() == holdout.record_id.tolist()
    expected = (
        "promote"
        if results["mae_v1_on_check"] > 0
        and results["mae_v2_on_check"] <= 0.88 * results["mae_v1_on_check"]
        else "keep"
    )
    assert results["decision"] == expected
