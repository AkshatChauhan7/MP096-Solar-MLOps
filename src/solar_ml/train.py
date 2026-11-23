"""Model training with MLflow tracking for MP096."""

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    load_cleaned_data,
    save_cleaned_data,
    clean_data,
    load_raw_data,
    prepare_features_target,
    get_feature_names,
    get_data_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MLFLOW_DB = ROOT / "mlflow.db"


def get_preprocessing_pipeline() -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_features, categorical_features = get_feature_names()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_model_configs() -> list[dict[str, Any]]:
    """Return list of model configurations to try."""
    return [
        {
            "name": "RandomForest",
            "model": RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        },
        {
            "name": "GradientBoosting",
            "model": GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            ),
        },
        {
            "name": "Ridge",
            "model": Ridge(alpha=1.0, random_state=42),
        },
        {
            "name": "RandomForest_Deep",
            "model": RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        },
    ]


def train_model(X: pd.DataFrame, y: pd.Series, model_config: dict[str, Any]) -> Pipeline:
    """Train a single model with preprocessing pipeline."""
    preprocessor = get_preprocessing_pipeline()
    model = model_config["model"]

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model),
    ])

    pipeline.fit(X, y)
    return pipeline


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """Evaluate model using cross-validation MAE."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X, y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return -scores.mean()


def run_training_pipeline() -> dict[str, Any]:
    """
    Run complete training pipeline:
    1. Load and clean data
    2. Train 3+ model configurations with MLflow tracking
    3. Select best model by CV MAE
    4. Register best model as v1 with champion alias
    5. Save model to models/model.joblib
    """
    # Setup MLflow
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("solar-energy-mp096")

    client = MlflowClient()

    # Load and clean data
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    save_cleaned_data(clean_df)

    X, y = prepare_features_target(clean_df)
    data_sha256 = get_data_sha256()

    # Get model configs
    configs = get_model_configs()

    best_mae = float("inf")
    best_model = None
    best_config = None
    best_run_id = None

    print(f"Training {len(configs)} model configurations...")
    print(f"Data SHA256: {data_sha256}")
    print(f"Training samples: {len(X)}")
    print(f"Features: {list(X.columns)}")

    for config in configs:
        with mlflow.start_run(run_name=config["name"]) as run:
            run_id = run.info.run_id

            # Train model
            pipeline = train_model(X, y, config)

            # Evaluate with cross-validation
            mae = evaluate_model(pipeline, X, y)

            # Log parameters
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", config["name"])
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", X.shape[1])

            # Log model hyperparameters
            model_params = config["model"].get_params()
            for k, v in model_params.items():
                mlflow.log_param(f"model_{k}", v)

            # Log metric
            mlflow.log_metric("cv_mae", mae)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(f"  {config['name']}: CV MAE = {mae:.4f}")

            # Track best
            if mae < best_mae:
                best_mae = mae
                best_model = pipeline
                best_config = config
                best_run_id = run_id

    print(f"\nBest model: {best_config['name']} with CV MAE = {best_mae:.4f}")
    print(f"Best run_id: {best_run_id}")

    # Register best model
    model_name = "solar-energy-model"
    model_uri = f"runs:/{best_run_id}/model"

    try:
        registered_model = mlflow.register_model(model_uri, model_name)
        version = registered_model.version
        print(f"Registered model version: {version}")

        # Set champion alias
        client.set_registered_model_alias(model_name, "champion", version)
        print(f"Set @champion alias to version {version}")

    except Exception as e:
        print(f"Model registration error: {e}")
        # Try to get existing version
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            version = max(v.version for v in versions)
            client.set_registered_model_alias(model_name, "champion", version)
            print(f"Set @champion alias to existing version {version}")
        else:
            version = 1

    # Save best model for API
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(best_model, MODEL_DIR / "model.joblib")
    print(f"Saved model to {MODEL_DIR / 'model.joblib'}")

    # Save metrics
    metrics = {
        "best_model": best_config["name"],
        "best_cv_mae": float(best_mae),
        "best_run_id": best_run_id,
        "registered_model_name": model_name,
        "champion_version": version,
        "data_sha256": data_sha256,
        "n_samples": len(X),
    }

    REPORTS_DIR = ROOT / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_champion_model() -> Pipeline:
    """Load the champion model from MLflow."""
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    model_uri = "models:/solar-energy-model@champion"
    return mlflow.sklearn.load_model(model_uri)


if __name__ == "__main__":
    metrics = run_training_pipeline()
    print("\nTraining complete!")
    print(json.dumps(metrics, indent=2))