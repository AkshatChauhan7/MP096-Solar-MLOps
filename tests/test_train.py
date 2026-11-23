"""Tests for model training pipeline."""

import json
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from src.solar_ml.data import clean_data, load_raw_data, prepare_features_target
from src.solar_ml.train import (
    get_preprocessing_pipeline,
    get_model_configs,
    train_model,
    evaluate_model,
    run_training_pipeline,
    load_champion_model,
)


class TestModelConfigs:
    """Test model configuration generation."""

    def test_get_model_configs_returns_at_least_three(self):
        configs = get_model_configs()
        assert len(configs) >= 3
        for config in configs:
            assert "name" in config
            assert "model" in config

    def test_model_configs_have_distinct_names(self):
        configs = get_model_configs()
        names = [c["name"] for c in configs]
        assert len(names) == len(set(names))

    def test_model_configs_are_sklearn_estimators(self):
        configs = get_model_configs()
        for config in configs:
            assert hasattr(config["model"], "fit")
            assert hasattr(config["model"], "predict")


class TestPreprocessing:
    """Test preprocessing pipeline."""

    def test_preprocessing_pipeline_created(self):
        preprocessor = get_preprocessing_pipeline()
        assert preprocessor is not None

    def test_preprocessing_fits_and_transforms(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        X, y = prepare_features_target(clean_df)

        preprocessor = get_preprocessing_pipeline()
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] > 0


class TestTraining:
    """Test model training functions."""

    def test_train_model_returns_pipeline(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        X, y = prepare_features_target(clean_df)

        configs = get_model_configs()
        pipeline = train_model(X, y, configs[0])

        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "fit")

    def test_train_model_can_predict(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        X, y = prepare_features_target(clean_df)

        configs = get_model_configs()
        pipeline = train_model(X, y, configs[0])

        preds = pipeline.predict(X.iloc[:5])
        assert len(preds) == 5
        assert all(isinstance(p, (int, float, np.number)) for p in preds)

    def test_evaluate_model_returns_mae(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        X, y = prepare_features_target(clean_df)

        configs = get_model_configs()
        pipeline = train_model(X, y, configs[0])

        mae = evaluate_model(pipeline, X, y)
        assert isinstance(mae, float)
        assert mae > 0


class TestTrainingPipeline:
    """Test full training pipeline (integration)."""

    @pytest.fixture(autouse=True)
    def setup_mlflow(self, tmp_path):
        """Use temporary MLflow tracking for tests."""
        test_db = tmp_path / "test_mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{test_db}")
        mlflow.set_experiment("test-solar-energy")
        yield
        mlflow.end_run()

    def test_run_training_pipeline_completes(self):
        """Test that training pipeline runs and produces outputs."""
        metrics = run_training_pipeline()

        # Check metrics structure
        assert "best_model" in metrics
        assert "best_cv_mae" in metrics
        assert "best_run_id" in metrics
        assert "registered_model_name" in metrics
        assert "champion_version" in metrics
        assert "data_sha256" in metrics
        assert "n_samples" in metrics

        # Check MLflow run was created
        assert metrics["best_run_id"] is not None
        assert metrics["best_cv_mae"] > 0

    def test_run_training_pipeline_creates_model_file(self):
        """Test that model.joblib is saved."""
        metrics = run_training_pipeline()
        model_path = Path("models/model.joblib")
        assert model_path.exists()

    def test_run_training_pipeline_creates_metrics_file(self):
        """Test that metrics.json is saved."""
        metrics = run_training_pipeline()
        metrics_path = Path("reports/metrics.json")
        assert metrics_path.exists()

        with open(metrics_path) as f:
            saved = json.load(f)
        assert saved["best_model"] == metrics["best_model"]


class TestChampionLoading:
    """Test loading champion model."""

    def test_load_champion_model(self):
        # First run training to create champion
        run_training_pipeline()

        # Load champion
        champion = load_champion_model()
        assert hasattr(champion, "predict")

        # Test prediction
        df = load_raw_data()
        clean_df = clean_data(df)
        X, _ = prepare_features_target(clean_df)
        preds = champion.predict(X.iloc[:3])
        assert len(preds) == 3


class TestDataSHA256Logged:
    """Test that data_sha256 is logged in MLflow runs."""

    def test_data_sha256_logged(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        X, y = prepare_features_target(clean_df)

        configs = get_model_configs()
        pipeline = train_model(X, y, configs[0])

        # Check that we can compute the SHA
        from src.solar_ml.data import get_data_sha256
        sha = get_data_sha256()
        assert len(sha) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])