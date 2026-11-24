"""Training pipeline for Solar MLOps project with MLflow tracking."""

import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .cleaning import load_data
from .mlflow_utils import (
    MLflowTracker,
    log_dataset_info,
    create_model_signature,
    REGISTERED_MODEL_NAME,
    register_model,
    set_model_alias,
)


# Feature and target configuration
TARGET_COLUMN = "energy_kwh"
FEATURE_COLUMNS = [
    "panel_area_m2",
    "sunshine_hours",
    "temp_c",
    "dust_days",
    "panel_age_yr",
    "cloud_cover_pct",
    "roof_type",
    "area",
]
CATEGORICAL_FEATURES = ["roof_type", "area"]
NUMERICAL_FEATURES = [
    "panel_area_m2",
    "sunshine_hours",
    "temp_c",
    "dust_days",
    "panel_age_yr",
    "cloud_cover_pct",
]


def create_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline for features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def create_model_pipeline(model_type: str = "random_forest", **model_params) -> Pipeline:
    """Create a full pipeline with preprocessing and model."""
    preprocessor = create_preprocessor()

    if model_type == "random_forest":
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **model_params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=42, **model_params)
    elif model_type == "ridge":
        model = Ridge(random_state=42, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare features and target from cleaned dataframe."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data.")
    missing_features = sorted(set(FEATURE_COLUMNS) - set(df.columns))
    if missing_features:
        raise ValueError(f"Required feature columns are missing: {missing_features}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def original_training_data_sha256() -> str:
    """Return the SHA-256 of the original DVC-tracked training CSV."""
    original_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.csv"
    return hashlib.sha256(original_path.read_bytes()).hexdigest()


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    return {
        "mae": mean_absolute_error(y, y_pred),
        "mse": mean_squared_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "r2": r2_score(y, y_pred),
    }


def train_model(
    train_path: str = "data/processed/train_clean.csv",
    model_type: str = "random_forest",
    model_params: dict = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Train a model with MLflow tracking."""
    model_params = model_params or {}

    # Load and prepare data
    df = load_data(train_path)
    X, y = prepare_data(df)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create and train pipeline
    pipeline = create_model_pipeline(model_type, **model_params)

    with MLflowTracker(run_name=f"{model_type}_training", tags={"model_type": model_type}) as tracker:
        # Log dataset info
        log_dataset_info(
            train_path=train_path,
            test_path=train_path,
            train_shape=X_train.shape,
            test_shape=X_val.shape,
            features=FEATURE_COLUMNS,
            target=TARGET_COLUMN,
        )

        # Log model parameters
        tracker.log_params({
            "model_type": model_type,
            **model_params,
            "test_size": test_size,
            "random_state": random_state,
            "data_sha256": original_training_data_sha256(),
        })

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate on validation set
        val_metrics = evaluate_model(pipeline, X_val, y_val)
        tracker.log_metrics({**val_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}})

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        tracker.log_metrics({
            "cv_mae_mean": -cv_scores.mean(),
            "cv_mae_std": cv_scores.std(),
        })

        # Create signature and log model
        signature = create_model_signature(X_train, pipeline.predict(X_train))
        input_example = X_train.iloc[:5]

        model_uri = tracker.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        print(f"Model trained: {model_type}")
        print(f"Validation MAE: {val_metrics['mae']:.4f}")
        print(f"CV MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Model logged to: {model_uri}")

        return pipeline, model_uri, val_metrics


def train_and_compare_models(
    train_path: str = "data/processed/train_clean.csv",
    model_configs: list = None,
) -> dict:
    """Train multiple models and compare them."""
    model_configs = model_configs or [
        {"model_type": "random_forest", "params": {"n_estimators": 100, "max_depth": 10}},
        {"model_type": "gradient_boosting", "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}},
        {"model_type": "ridge", "params": {"alpha": 1.0}},
    ]

    results = {}
    best_model = None
    best_model_type = None
    best_mae = float("inf")
    best_uri = None

    for config in model_configs:
        model_type = config["model_type"]
        params = config["params"]
        pipeline, model_uri, metrics = train_model(
            train_path=train_path,
            model_type=model_type,
            model_params=params,
        )
        results[model_type] = {
            "pipeline": pipeline,
            "model_uri": model_uri,
            "metrics": metrics,
        }

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_model = pipeline
            best_model_type = model_type
            best_uri = model_uri

    # Register only the selected v1 model, then give it the required champion alias.
    if best_model is not None and best_uri is not None:
        registered = register_model(best_uri, REGISTERED_MODEL_NAME)
        version = int(registered.version)
        set_model_alias(REGISTERED_MODEL_NAME, "champion", version)
        for result in results.values():
            result["is_best"] = result["model_uri"] == best_uri
        results[best_model_type]["registered_version"] = version
        print(f"\nBest model: {best_uri} (MAE: {best_mae:.4f})")
        print(f"Registered {REGISTERED_MODEL_NAME} version {version} as champion")

    return results


def main():
    """Main training entry point."""
    # Ensure processed data exists
    train_path = Path("data/processed/train_clean.csv")
    if not train_path.exists():
        print("Processed data not found. Run cleaning first:")
        print("  python -m src.cleaning")
        return

    # Train and compare models
    results = train_and_compare_models(str(train_path))

    # Print summary
    print("\n=== Training Summary ===")
    for model_type, result in results.items():
        print(f"{model_type}: MAE={result['metrics']['mae']:.4f}, R2={result['metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()
