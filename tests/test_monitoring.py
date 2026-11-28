"""Monitoring tests use synthetic data and never depend on private CSVs."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.monitoring import (
    AREAS,
    group_mae,
    load_v1_model,
    population_stability_index,
    psi_by_feature,
    run_monitoring,
    update_findings,
)
from src.train import FEATURE_COLUMNS, NUMERICAL_FEATURES


def test_psi_identical_distribution():
    values = np.arange(100)
    assert population_stability_index(values, values[::-1]) == 0


def test_psi_known_ten_bin_result():
    # Exactly 10 equally populated reference bins; actual populates the first 5.
    reference = np.arange(10)
    actual = np.arange(5)
    expected = 5 * (0.2 - 0.1) * np.log(2)
    expected += 5 * (0.0001 - 0.1) * np.log(0.0001 / 0.1)
    assert population_stability_index(reference, actual) == pytest.approx(expected)


def test_psi_out_of_range_is_counted_in_outer_bins():
    assert population_stability_index(np.arange(10), [-100, 100]) == pytest.approx(
        population_stability_index(np.arange(10), [0, 9])
    )
    assert population_stability_index(np.arange(10), [-100, 100]) > 0


def test_psi_missing_bucket_includes_missingness_drift():
    assert population_stability_index([1, None, 2], [1, None, 2]) == 0
    assert population_stability_index([1, 2], [None, None]) > 0
    assert population_stability_index([1, 2], [1, pd.NA]) > 0


def test_psi_constant_reference_detects_changes_in_both_directions():
    assert population_stability_index([5] * 10, [5] * 3) == 0
    for actual in ([4] * 3, [6] * 3, [4, 5, 6]):
        result = population_stability_index([5] * 10, actual)
        assert np.isfinite(result) and result > 0


@pytest.mark.parametrize(
    "reference,actual", [([], [1]), ([1], []), ([None], [1]), ([np.inf], [1]), ([1], [-np.inf])]
)
def test_psi_rejects_unusable_samples(reference, actual):
    with pytest.raises(ValueError):
        population_stability_index(reference, actual)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bins": 0},
        {"bins": 2.5},
        {"bins": True},
        {"epsilon": 0},
        {"epsilon": 1},
        {"epsilon": np.nan},
    ],
)
def test_psi_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        population_stability_index([1, 2], [2, 3], **kwargs)


def test_psi_monitors_only_prediction_features():
    train = pd.DataFrame({name: [1, 2, 3] for name in NUMERICAL_FEATURES})
    train["energy_kwh"] = [1, 2, 3]
    later = train.copy()
    later["dust_days"] = [100, 100, 100]
    result = psi_by_feature(train, later)
    assert set(result) == set(NUMERICAL_FEATURES)
    assert max(result, key=result.get) == "dust_days"


def test_group_mae_is_positional_and_uses_group_denominators():
    target = pd.Series([10, 20, 30, 40], index=[9, 4, 2, 0])
    predictions = pd.Series([11, 18, 36, 43], index=[0, 1, 2, 3])
    result = group_mae(target, predictions, ["Old City", "Old City", "Industrial", "New Colony"])
    assert result == {"Old City": 1.5, "New Colony": 3.0, "Industrial": 6.0}


@pytest.mark.parametrize(
    "target,predictions,groups",
    [
        ([], [], []),
        ([1], [1, 2], ["Old City"]),
        ([1, 2, np.nan], [1, 2, 3], AREAS),
        ([1, 2, 3], [1, np.inf, 3], AREAS),
        ([1, 2, 3], [1, 2, 3], ["Old City", "Old City", "Industrial"]),
        ([1, 2, 3], [1, 2, 3], ["Old City", None, "Industrial"]),
        ([[1], [2], [3]], [1, 2, 3], AREAS),
    ],
)
def test_group_mae_rejects_invalid_input(target, predictions, groups):
    with pytest.raises(ValueError):
        group_mae(target, predictions, groups)


def test_loads_registry_version_one_even_when_champion_changes():
    with (
        patch("src.monitoring.mlflow.set_tracking_uri") as tracking,
        patch("src.monitoring.mlflow.sklearn.load_model") as load,
    ):
        assert load_v1_model() is load.return_value
    tracking.assert_called_once_with("sqlite:///mlflow.db")
    load.assert_called_once_with("models:/solar-power-predictor/1")


@pytest.mark.parametrize("run_id,valid", [("original-v1", True), ("different-run", False)])
def test_relocated_artifact_must_match_registered_v1(tmp_path, monkeypatch, run_id, valid):
    monkeypatch.delenv("STAGE4_V1_ARTIFACT_DIR", raising=False)
    (tmp_path / "MLmodel").write_text(yaml.safe_dump({"run_id": run_id}))
    with (
        patch("src.monitoring.MlflowClient") as client,
        patch("src.monitoring.mlflow.sklearn.load_model") as load,
        patch("src.monitoring.mlflow.set_tracking_uri"),
    ):
        client.return_value.get_model_version.return_value.run_id = "original-v1"
        config = {"v1_artifact_dir": str(tmp_path)}
        if valid:
            assert load_v1_model(config) is load.return_value
            load.assert_called_once_with(str(tmp_path))
        else:
            with pytest.raises(ValueError, match="does not belong"):
                load_v1_model(config)
            load.assert_not_called()


def test_v1_loader_uses_configured_registry(monkeypatch):
    monkeypatch.delenv("STAGE4_V1_ARTIFACT_DIR", raising=False)
    with (
        patch("src.monitoring.mlflow.set_tracking_uri") as tracking,
        patch("src.monitoring.mlflow.sklearn.load_model") as load,
    ):
        load_v1_model({"tracking_uri": "sqlite:///fixture.db", "name": "fixture-model"})
    tracking.assert_called_once_with("sqlite:///fixture.db")
    load.assert_called_once_with("models:/fixture-model/1")


def test_update_findings_preserves_other_stages_and_comments(tmp_path):
    findings = tmp_path / "findings.yaml"
    original = Path("reports/findings.yaml").read_text(encoding="utf-8")
    findings.write_text(original, encoding="utf-8")
    before = yaml.safe_load(original)
    update_findings(
        findings,
        {
            "psi": dict.fromkeys(NUMERICAL_FEATURES, 0.5),
            "most_drifted_feature": "dust_days",
            "mae_by_group": dict.fromkeys(AREAS, 1.5),
            "worst_served_group": "Old City",
        },
    )
    after = yaml.safe_load(findings.read_text())
    assert after["tracking"] == before["tracking"]
    assert after["data_issues"] == before["data_issues"]
    assert after["monitoring"]["decision"] == before["monitoring"]["decision"]
    assert after["monitoring"]["psi"]["dust_days"] == 0.5
    assert after["monitoring"]["mae_by_group"]["Old City"] == 1.5
    assert "# v1 on later.csv" in findings.read_text()


def monitoring_config(tmp_path):
    findings = tmp_path / "findings.yaml"
    findings.write_text(Path("reports/findings.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "train_path": "train.csv",
        "later_path": "later.csv",
        "bins": 10,
        "epsilon": 0.0001,
        "findings_path": str(findings),
        "predictions_path": str(tmp_path / "predictions.csv"),
    }


def synthetic_later():
    frame = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in NUMERICAL_FEATURES})
    frame["area"] = list(AREAS)
    frame["roof_type"] = "flat"
    frame["record_id"] = ["a", "b", "c"]
    # Training target typo correction must NOT run on clean later data.
    frame["energy_kwh"] = [110.0, 120.0, 130.0]
    return frame


def test_full_monitoring_predicts_all_later_rows_without_cleaning_later(tmp_path):
    config = monitoring_config(tmp_path)
    later = synthetic_later()
    model = Mock()
    model.predict.return_value = [109.0, 118.0, 127.0]
    with (
        patch("src.monitoring.load_data", side_effect=[later, later]),
        patch("src.monitoring.load_v1_model", return_value=model),
    ):
        result = run_monitoring(config)
    assert result["mae_by_group"] == dict(zip(AREAS, [1.0, 2.0, 3.0], strict=True))
    assert result["worst_served_group"] == "Industrial"
    pd.testing.assert_frame_equal(model.predict.call_args.args[0], later[FEATURE_COLUMNS])
    predictions = pd.read_csv(config["predictions_path"])
    assert predictions["record_id"].tolist() == ["a", "b", "c"]
    assert predictions["prediction"].tolist() == [109.0, 118.0, 127.0]


def test_missing_original_model_never_writes_fake_results(tmp_path):
    config = monitoring_config(tmp_path)
    before = Path(config["findings_path"]).read_bytes()
    later = synthetic_later()
    with (
        patch("src.monitoring.load_data", side_effect=[later, later]),
        patch("src.monitoring.load_v1_model", side_effect=FileNotFoundError("original v1 missing")),
    ):
        with pytest.raises(FileNotFoundError):
            run_monitoring(config)
    assert Path(config["findings_path"]).read_bytes() == before
    assert not Path(config["predictions_path"]).exists()


def test_explicit_psi_only_does_not_load_model_or_change_mae(tmp_path):
    config = monitoring_config(tmp_path)
    before = yaml.safe_load(Path(config["findings_path"]).read_text())["monitoring"]
    later = synthetic_later()
    with (
        patch("src.monitoring.load_data", side_effect=[later, later]),
        patch("src.monitoring.load_v1_model") as load,
    ):
        result = run_monitoring(config, psi_only=True)
    load.assert_not_called()
    assert "mae_by_group" not in result
    after = yaml.safe_load(Path(config["findings_path"]).read_text())["monitoring"]
    assert after["mae_by_group"] == before["mae_by_group"]
