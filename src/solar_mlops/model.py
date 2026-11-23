"""Model utilities for solar energy prediction MLOps pipeline."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import setup_logging

logger = setup_logging(__name__)

# Column definitions
TARGET_COLUMN = "energy_kwh"
NUMERIC_FEATURES = [
    "panel_area_m2",
    "sunshine_hours",
    "temp_c",
    "dust_days",
    "panel_age_yr",
    "cloud_cover_pct",
]
CATEGORICAL_FEATURES = ["roof_type", "area"]
EXCLUDE_COLUMNS = ["record_id", "date", "meter_reading_kwh", TARGET_COLUMN]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature column names from DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        List of feature column names (excluding target and metadata columns)
    """
    return [col for col in df.columns if col not in EXCLUDE_COLUMNS]


def get_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing pipeline for numeric and categorical features.

    Returns:
        ColumnTransformer for preprocessing
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def create_model_pipeline(model: BaseEstimator, preprocessor: ColumnTransformer) -> Pipeline:
    """
    Create a full pipeline with preprocessing and model.

    Args:
        model: Scikit-learn estimator
        preprocessor: ColumnTransformer for preprocessing

    Returns:
        Complete Pipeline
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def get_models_config() -> Dict[str, Dict[str, Any]]:
    """
    Get configuration of models to train with their hyperparameters.

    Returns:
        Dictionary mapping model names to configurations
    """
    return {
        "RandomForest": {
            "model": RandomForestRegressor,
            "params": {
                "n_estimators": 300,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor,
            "params": {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.1,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        },
        "LinearRegression": {
            "model": LinearRegression,
            "params": {},
        },
        "RandomForest_Deep": {
            "model": RandomForestRegressor,
            "params": {
                "n_estimators": 500,
                "max_depth": 15,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "GradientBoosting_LR": {
            "model": GradientBoostingRegressor,
            "params": {
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        },
    }


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from DataFrame.

    Args:
        df: Input DataFrame with features and target

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    X = df[get_feature_columns(df)].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def train_and_evaluate(
    model_name: str,
    model_config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    preprocessor: ColumnTransformer,
) -> Dict[str, Any]:
    """
    Train a model and evaluate on validation set.

    Args:
        model_name: Name of the model
        model_config: Model configuration dictionary
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        preprocessor: Preprocessing ColumnTransformer

    Returns:
        Dictionary with model, pipeline, and metrics
    """
    logger.info(f"Training {model_name}...")

    model_class = model_config["model"]
    model_params = model_config["params"]
    model = model_class(**model_params)

    pipeline = create_model_pipeline(model, preprocessor)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"{model_name} validation MAE: {mae:.4f}")

    return {
        "model_name": model_name,
        "model": model,
        "pipeline": pipeline,
        "mae": mae,
        "params": model_params,
    }


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> Dict[str, float]:
    """
    Perform cross-validation on a pipeline.

    Args:
        pipeline: Scikit-learn Pipeline
        X: Features
        y: Target
        cv: Number of CV folds

    Returns:
        Dictionary with CV scores
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    mae_scores = -scores

    return {
        "cv_mae_mean": float(mae_scores.mean()),
        "cv_mae_std": float(mae_scores.std()),
        "cv_mae_scores": mae_scores.tolist(),
    }


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    """
    Save model pipeline to disk using joblib.

    Args:
        pipeline: Trained Pipeline
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str | Path) -> Pipeline:
    """
    Load model pipeline from disk.

    Args:
        path: Model file path

    Returns:
        Loaded Pipeline
    """
    pipeline = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return pipeline


def predict(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the pipeline.

    Args:
        pipeline: Trained Pipeline
        X: Features DataFrame

    Returns:
        Array of predictions
    """
    return pipeline.predict(X)