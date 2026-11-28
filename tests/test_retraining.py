"""Synthetic verification of temporal isolation, registry safety and final output."""

from pathlib import Path
from unittest.mock import Mock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from src.retraining import (
    chronological_split,
    finalize,
    fit_candidate,
    load_candidate,
    log_pipeline,
    predict_checked,
    promotion_decision,
    register_pair,
    train_candidate,
    update_tracking,
    write_holdout_predictions,
)
from src.train import FEATURE_COLUMNS, create_model_pipeline


def fixture_frame(n=30, prefix="later"):
    index = np.arange(n)
    return pd.DataFrame(
        {
            "record_id": [f"{prefix}-{i}" for i in index],
            "date": pd.date_range("2026-07-01", periods=n).strftime("%Y-%m-%d"),
            "panel_area_m2": 10.0 + index,
            "sunshine_hours": 4.0 + index % 4,
            "temp_c": 25.0 + index % 8,
            "dust_days": index % 6,
            "panel_age_yr": 2.0 + index % 5,
            "cloud_cover_pct": 10.0 + index % 40,
            "roof_type": np.resize(["flat", "tin", "sloped"], n),
            "area": np.resize(["Old City", "New Colony", "Industrial"], n),
            "energy_kwh": 5.0 + 0.5 * index,
        }
    )


def test_chronological_split_sorts_stably_and_has_no_overlap():
    later = fixture_frame(10).iloc[[9, 0, 8, 1, 7, 2, 6, 3, 5, 4]]
    early, check = chronological_split(later)
    assert len(early) == 7 and len(check) == 3
    assert early["date"].max() <= check["date"].min()
    assert set(early.record_id).isdisjoint(check.record_id)
    assert check.record_id.tolist() == ["later-7", "later-8", "later-9"]


def test_chronological_split_ties_keep_original_row_order():
    later = fixture_frame(10)
    later["date"] = "2026-07-01"
    early, check = chronological_split(later)
    assert early.record_id.tolist() == later.record_id.iloc[:7].tolist()
    assert check.record_id.tolist() == later.record_id.iloc[7:].tolist()


@pytest.mark.parametrize("date", [None, "not-a-date"])
def test_invalid_dates_rejected(date):
    later = fixture_frame()
    later.loc[0, "date"] = date
    with pytest.raises(ValueError):
        chronological_split(later)


def test_short_sample_rejected():
    with pytest.raises(ValueError):
        chronological_split(fixture_frame(1))


@pytest.mark.parametrize(
    "v1,v2,expected",
    [
        (10.0, 8.8, "promote"),
        (10.0, 8.8001, "keep"),
        (10.0, 7.0, "promote"),
        (10.0, 11.0, "keep"),
        (0.0, 0.0, "keep"),
        (10.0, 0.0, "promote"),
    ],
)
def test_exact_twelve_percent_rule(v1, v2, expected):
    assert promotion_decision(v1, v2) == expected


@pytest.mark.parametrize("v1,v2", [(-1, 0), (1, -1), (np.nan, 1), (1, np.inf)])
def test_invalid_mae_rejected(v1, v2):
    with pytest.raises(ValueError):
        promotion_decision(v1, v2)


def test_check_rows_never_reach_fit_or_preprocessor():
    train, later = fixture_frame(20, "train"), fixture_frame(10)
    changed = later.copy()
    changed.loc[7:, "energy_kwh"] = 1e6
    changed.loc[7:, "panel_area_m2"] = 1e8
    config = {"type": "ridge", "params": {"alpha": 1.0}}
    a, _, _, _ = fit_candidate(train, later, config)
    b, _, _, _ = fit_candidate(train, changed, config)
    np.testing.assert_allclose(a.predict(train[FEATURE_COLUMNS]), b.predict(train[FEATURE_COLUMNS]))
    np.testing.assert_allclose(
        a["preprocessor"].named_transformers_["num"].mean_,
        b["preprocessor"].named_transformers_["num"].mean_,
    )


@pytest.mark.parametrize("values", [[np.nan, 1], [1], [[1], [2]]])
def test_predictions_must_be_finite_and_match_rows(values):
    model = Mock()
    model.predict.return_value = values
    with pytest.raises(ValueError):
        predict_checked(model, fixture_frame(2)[FEATURE_COLUMNS])


def test_holdout_output_has_exact_schema_and_preserves_ids(tmp_path):
    frame = fixture_frame(5).drop(columns="energy_kwh").iloc[::-1]
    model = Mock()
    model.predict.return_value = np.arange(5, dtype=float)
    output = tmp_path / "predictions" / "holdout_predictions.csv"
    result = write_holdout_predictions(model, frame, output)
    assert list(result.columns) == ["record_id", "prediction"]
    assert result.record_id.tolist() == frame.record_id.tolist()
    assert len(pd.read_csv(output)) == len(frame)
    assert list(model.predict.call_args.args[0].columns) == FEATURE_COLUMNS


@pytest.mark.parametrize("issue", ["duplicates", "missing_ids", "missing_feature"])
def test_invalid_holdout_does_not_write_file(tmp_path, issue):
    frame = fixture_frame(5)
    if issue == "duplicates":
        frame.loc[1, "record_id"] = frame.loc[0, "record_id"]
    elif issue == "missing_ids":
        frame.loc[0, "record_id"] = None
    else:
        frame = frame.drop(columns="sunshine_hours")
    output = tmp_path / "holdout.csv"
    with pytest.raises((ValueError, KeyError)):
        write_holdout_predictions(Mock(), frame, output)
    assert not output.exists()


def test_update_tracking_preserves_data_and_monitoring(tmp_path):
    path = tmp_path / "findings.yaml"
    original = Path("reports/findings.yaml").read_text(encoding="utf-8")
    path.write_text(original, encoding="utf-8")
    before = yaml.safe_load(original)
    update_tracking(path, {"registered_model_name": "stage4", "champion_version": 1})
    after = yaml.safe_load(path.read_text())
    assert after["monitoring"] == before["monitoring"]
    assert after["data_issues"] == before["data_issues"]
    assert after["tracking"]["registered_model_name"] == "stage4"


@pytest.fixture
def config(tmp_path):
    train = fixture_frame(40, "train")
    train["date"] = "2026-01-01"
    train.to_csv(tmp_path / "train.csv", index=False)
    fixture_frame().to_csv(tmp_path / "later.csv", index=False)
    fixture_frame(8, "holdout").drop(columns="energy_kwh").to_csv(
        tmp_path / "holdout.csv", index=False
    )
    findings = tmp_path / "findings.yaml"
    findings.write_text(Path("reports/findings.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    tracking = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    return {
        "train_path": str(tmp_path / "train.csv"),
        "later_path": str(tmp_path / "later.csv"),
        "holdout_path": str(tmp_path / "holdout.csv"),
        "findings_path": str(findings),
        "predictions_path": str(tmp_path / "later_predictions.csv"),
        "bins": 10,
        "epsilon": 0.0001,
        "model": {"tracking_uri": tracking, "name": "original-model"},
        "retraining": {
            "tracking_uri": tracking,
            "experiment_name": "stage4-test",
            "registered_model_name": "stage4-model",
            "artifact_root": str(tmp_path / "artifacts"),
            "model": {"type": "ridge", "params": {"alpha": 1.0}},
            "candidate_path": str(tmp_path / "candidate.joblib"),
            "candidate_report": str(tmp_path / "candidate.yaml"),
            "final_report": str(tmp_path / "final.yaml"),
            "champion_path": str(tmp_path / "model.joblib"),
            "holdout_predictions_path": str(tmp_path / "holdout_predictions.csv"),
        },
    }


def test_missing_original_blocks_finalization_before_any_writes(config):
    findings = Path(config["findings_path"])
    before = findings.read_bytes()
    with (
        patch("src.retraining.load_v1_model", side_effect=FileNotFoundError("original v1 missing")),
        patch("src.retraining.register_pair") as register,
    ):
        with pytest.raises(FileNotFoundError):
            finalize(config)
    register.assert_not_called()
    assert findings.read_bytes() == before
    assert not Path(config["retraining"]["holdout_predictions_path"]).exists()


def test_registration_rejects_source_namespace(config):
    config["retraining"]["registered_model_name"] = config["model"]["name"]
    with pytest.raises(ValueError, match="must differ"):
        register_pair(Mock(), {}, pd.DataFrame(), {}, config)


def test_registration_rejects_conflicting_existing_versions(config):
    with (
        patch("src.retraining.MlflowClient") as client,
        patch("src.retraining.mlflow.register_model") as register,
    ):
        client.return_value.search_model_versions.return_value = [Mock(version="2", run_id="old")]
        with pytest.raises(ValueError, match="different/incomplete"):
            register_pair(Mock(), {"run_id": "new"}, pd.DataFrame(), {}, config)
    register.assert_not_called()


@pytest.mark.parametrize("expected_decision", ["promote", "keep"])
def test_synthetic_pipeline_preserves_original_versions_and_publishes_champion(
    config, expected_decision
):
    """Actual MLflow/SQLite and sklearn roundtrip on entirely synthetic data."""
    old_tracking = mlflow.get_tracking_uri()
    old_registry = mlflow.get_registry_uri()
    try:
        frame = pd.read_csv(config["train_path"])
        original = create_model_pipeline("ridge", alpha=1.0)
        estimator = (
            DummyRegressor(strategy="constant", constant=0.0)
            if expected_decision == "promote"
            else LinearRegression()
        )
        original.set_params(model=estimator)
        original.fit(frame[FEATURE_COLUMNS], frame.energy_kwh)
        settings = config["retraining"]
        baseline = log_pipeline(
            original,
            frame[FEATURE_COLUMNS],
            {"mae": 0.1},
            {"data_sha256": "synthetic"},
            settings,
            "fixture",
        )
        for _ in range(2):
            mlflow.register_model(baseline["model_uri"], config["model"]["name"])
        client = MlflowClient(tracking_uri=settings["tracking_uri"])
        client.set_registered_model_alias(config["model"]["name"], "champion", "2")
        before = [vars(v).copy() for v in client.search_model_versions("name='original-model'")]
        report = train_candidate(config)
        assert report["later_fit_rows"] == 21 and report["later_check_rows"] == 9
        assert not client.search_model_versions("name='stage4-model'")
        with patch("src.retraining.load_v1_model", return_value=original):
            results = finalize(config)
            repeated = finalize(config)
        assert repeated == results
        assert results["decision"] == expected_decision
        assert len(client.search_model_versions("name='stage4-model'")) == 2
        assert [
            vars(v).copy() for v in client.search_model_versions("name='original-model'")
        ] == before
        assert int(client.get_model_version_by_alias("original-model", "champion").version) == 2
        expected_version = 2 if results["decision"] == "promote" else 1
        assert (
            int(client.get_model_version_by_alias("stage4-model", "champion").version)
            == expected_version
        )
        assert len(pd.read_csv(settings["holdout_predictions_path"])) == 8
        findings = yaml.safe_load(Path(config["findings_path"]).read_text())
        assert findings["monitoring"]["decision"] == results["decision"]
        assert findings["tracking"]["registered_model_name"] == "stage4-model"
        # Data changes invalidate a candidate before a comparison can proceed.
        Path(config["later_path"]).write_text("changed")
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            load_candidate(config, pd.read_csv(config["train_path"]))
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(old_tracking)
        mlflow.set_registry_uri(old_registry)
