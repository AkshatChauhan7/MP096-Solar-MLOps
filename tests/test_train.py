"""Unit tests for training module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline

from src.train import (
    create_preprocessor,
    create_model_pipeline,
    prepare_data,
    evaluate_model,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    original_training_data_sha256,
)


class TestCreatePreprocessor:
    """Tests for create_preprocessor function."""

    def test_returns_column_transformer(self):
        preprocessor = create_preprocessor()
        assert isinstance(preprocessor, Pipeline) or hasattr(preprocessor, "transformers")

    def test_has_correct_transformers(self):
        preprocessor = create_preprocessor()
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in transformer_names
        assert "cat" in transformer_names

    def test_numerical_features_scaled(self):
        preprocessor = create_preprocessor()
        # Get transformer from transformers list (before fitting)
        num_transformer = None
        for name, trans, cols in preprocessor.transformers:
            if name == "num":
                num_transformer = trans
                break
        from sklearn.preprocessing import StandardScaler
        assert isinstance(num_transformer, StandardScaler)

    def test_categorical_features_encoded(self):
        preprocessor = create_preprocessor()
        # Get transformer from transformers list (before fitting)
        cat_transformer = None
        for name, trans, cols in preprocessor.transformers:
            if name == "cat":
                cat_transformer = trans
                break
        from sklearn.preprocessing import OneHotEncoder
        assert isinstance(cat_transformer, OneHotEncoder)
        assert cat_transformer.handle_unknown == "ignore"


class TestCreateModelPipeline:
    """Tests for create_model_pipeline function."""

    def test_creates_random_forest_pipeline(self):
        pipeline = create_model_pipeline("random_forest", n_estimators=10)
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(pipeline.named_steps["model"], RandomForestRegressor)
        assert pipeline.named_steps["model"].n_estimators == 10

    def test_creates_gradient_boosting_pipeline(self):
        pipeline = create_model_pipeline("gradient_boosting", n_estimators=20)
        from sklearn.ensemble import GradientBoostingRegressor
        assert isinstance(pipeline.named_steps["model"], GradientBoostingRegressor)

    def test_creates_ridge_pipeline(self):
        pipeline = create_model_pipeline("ridge", alpha=0.5)
        from sklearn.linear_model import Ridge
        assert isinstance(pipeline.named_steps["model"], Ridge)
        assert pipeline.named_steps["model"].alpha == 0.5

    def test_raises_on_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model_pipeline("unknown_model")


class TestPrepareData:
    """Tests for prepare_data function."""

    def test_splits_features_and_target(self):
        df = pd.DataFrame({
            "panel_area_m2": [20, 30],
            "sunshine_hours": [5, 6],
            "temp_c": [25, 26],
            "dust_days": [1, 2],
            "panel_age_yr": [5, 10],
            "cloud_cover_pct": [10, 20],
            "roof_type": ["tin", "flat"],
            "area": ["Old City", "New Colony"],
            "meter_reading_kwh": [500, 700],  # leakage; must not be a feature
            "energy_kwh": [100, 200],  # target
        })
        X, y = prepare_data(df)
        assert list(X.columns) == FEATURE_COLUMNS
        assert list(y) == [100, 200]
        assert "meter_reading_kwh" not in X.columns

    def test_raises_if_target_missing(self):
        df = pd.DataFrame({
            "panel_area_m2": [20, 30],
            "sunshine_hours": [5, 6],
            # missing target
        })
        with pytest.raises(ValueError, match="Target column"):
            prepare_data(df)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_all_metrics(self):
        # Create a simple pipeline that predicts mean
        class MeanPredictor:
            def __init__(self, mean_val):
                self.mean_val = mean_val
            def predict(self, X):
                return np.full(len(X), self.mean_val)

        pipeline = MeanPredictor(mean_val=150)
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([100, 150, 200])

        metrics = evaluate_model(pipeline, X, y)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] == pytest.approx(33.33, rel=0.01)
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


class TestFeatureColumns:
    """Tests that feature column definitions are consistent."""

    def test_feature_columns_complete(self):
        expected = [
            "panel_area_m2",
            "sunshine_hours",
            "temp_c",
            "dust_days",
            "panel_age_yr",
            "cloud_cover_pct",
            "roof_type",
            "area",
        ]
        assert FEATURE_COLUMNS == expected

    def test_categorical_features_subset(self):
        assert set(CATEGORICAL_FEATURES).issubset(set(FEATURE_COLUMNS))

    def test_numerical_features_subset(self):
        assert set(NUMERICAL_FEATURES).issubset(set(FEATURE_COLUMNS))

    def test_no_overlap(self):
        assert set(CATEGORICAL_FEATURES).isdisjoint(set(NUMERICAL_FEATURES))

    def test_all_features_covered(self):
        covered = set(CATEGORICAL_FEATURES) | set(NUMERICAL_FEATURES)
        assert covered == set(FEATURE_COLUMNS)


class TestTrainIntegration:
    """Integration tests for training (mocked MLflow)."""

    @patch("src.train.MLflowTracker")
    def test_train_model_calls_mlflow(self, mock_tracker_class, tmp_path):
        from src.train import train_model

        # Setup mock
        mock_tracker = MagicMock()
        mock_tracker.__enter__ = MagicMock(return_value=mock_tracker)
        mock_tracker.__exit__ = MagicMock(return_value=None)
        mock_tracker_class.return_value = mock_tracker

        # Create enough data for 5-fold CV (need at least 10 samples for 5 folds with 0.2 test_size)
        np.random.seed(42)
        n_samples = 20
        df = pd.DataFrame({
            "panel_area_m2": np.random.uniform(10, 60, n_samples),
            "sunshine_hours": np.random.uniform(2, 11, n_samples),
            "temp_c": np.random.uniform(15, 45, n_samples),
            "dust_days": np.random.randint(0, 30, n_samples),
            "panel_age_yr": np.random.uniform(5, 25, n_samples),
            "cloud_cover_pct": np.random.randint(0, 100, n_samples),
            "roof_type": np.random.choice(["tin", "flat", "sloped"], n_samples),
            "area": np.random.choice(["Old City", "New Colony", "Industrial"], n_samples),
            "meter_reading_kwh": np.random.uniform(50, 500, n_samples),
            "energy_kwh": np.random.uniform(10, 50, n_samples),
        })

        # Create temp directory for test data
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        train_path = processed_dir / "train_clean.csv"
        df.to_csv(train_path, index=False)

        pipeline, model_uri, metrics = train_model(
            train_path=str(train_path),
            model_type="random_forest",
            model_params={"n_estimators": 5, "max_depth": 3},
        )

        # Verify MLflow was called
        mock_tracker.log_params.assert_called()
        assert any(
            call.args[0].get("data_sha256") == original_training_data_sha256()
            for call in mock_tracker.log_params.call_args_list
        )
        mock_tracker.log_metrics.assert_called()
        mock_tracker.log_model.assert_called()

        assert isinstance(pipeline, Pipeline)
        assert "mae" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
