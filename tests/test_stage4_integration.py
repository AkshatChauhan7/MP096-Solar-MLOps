"""Opt-in actual-data integration using the recovered original v1 artifact.

Uses a temporary SQLite copy and temporary outputs, preserving the repository's
registry and reports. Set RUN_STAGE4_INTEGRATION=1 and configure the original
artifact directory before running these tests.
"""

import copy
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
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
        reason="Opt in to actual-data tests with restored v1 artifacts: RUN_STAGE4_INTEGRATION=1",
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


def test_actual_v1_and_export_work_from_another_checkout(actual_config, tmp_path):
    """Registry v1 and exported champion must work after moving the checkout."""
    with sqlite3.connect(tmp_path / "mlflow.db") as database:
        location = database.execute(
            "SELECT storage_location FROM model_versions WHERE name=? AND version=1",
            (actual_config["model"]["name"],),
        ).fetchone()[0]
    assert not Path(location).is_absolute() and not location.startswith("file:")
    shutil.copytree(ROOT / location, tmp_path / location)
    (tmp_path / "models").mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "models/model.joblib", tmp_path / "models/model.joblib")
    holdout = pd.read_csv(actual_config["holdout_path"])
    holdout.to_csv(tmp_path / "holdout.csv", index=False)
    original = load_v1_model(actual_config["model"])
    np.savez(
        tmp_path / "expected.npz",
        v1=original.predict(holdout[FEATURE_COLUMNS]),
        champion=pd.read_csv(ROOT / "predictions/holdout_predictions.csv").prediction.to_numpy(),
    )
    program = """
import sys, joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_registry_uri('sqlite:///mlflow.db')
v1=mlflow.sklearn.load_model('models:/'+sys.argv[1]+'/1')
champion=joblib.load('models/model.joblib')
x=pd.read_csv('holdout.csv')[list(v1.feature_names_in_)]
expected=np.load('expected.npz')
np.testing.assert_allclose(v1.predict(x),expected['v1'])
np.testing.assert_allclose(champion.predict(x),expected['champion'])
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, actual_config["model"]["name"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
