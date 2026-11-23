"""Training script for solar energy prediction model.

This script:
1. Sets MLflow tracking to sqlite:///mlflow.db (in repo root)
2. Loads cleaned train.csv (from data/processed/train_clean.csv)
3. Computes data_sha256 of original train.csv (from data/raw/train.csv)
4. Trains 3+ different models with different hyperparameters
5. For each run: logs data_sha256 param, MAE metric, model with log_model (serialization_format="cloudpickle")
6. Registers best model as "solar-energy-model" version 1
7. Sets @champion alias to version 1
8. Saves best model to models/model.joblib for the API
"""

import hashlib
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from .model import (
    create_model_pipeline,
    cross_validate_model,
    get_models_config,
    get_preprocessor,
    prepare_data,
    save_model,
    train_and_evaluate,
)
from .utils import setup_logging

logger = setup_logging(__name__)

# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent
MLFLOW_DB_PATH = REPO_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MODEL_REGISTRY_NAME = "solar-energy-model"
CHAMPION_ALIAS = "champion"
MODEL_OUTPUT_PATH = REPO_ROOT / "models" / "model.joblib"
TRAIN_CLEAN_PATH = REPO_ROOT / "data" / "processed" / "train_clean.csv"
TRAIN_RAW_PATH = REPO_ROOT / "data" / "raw" / "train.csv"
PARAMS_PATH = REPO_ROOT / "params.yaml"


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA256 hash as hex string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_params() -> dict:
    """Load parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def setup_mlflow() -> None:
    """Configure MLflow tracking."""
    logger.info(f"Setting MLflow tracking URI to {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("solar-energy-training")


def train() -> str:
    """
    Main training function.

    Returns:
        Best run_id
    """
    # Setup MLflow
    setup_mlflow()

    # Load parameters
    params = load_params()
    target_column = params["training"]["target_column"]
    random_state = params["model_settings"]["random_state"]
    test_size = params["model_settings"]["test_size"]

    # Compute data SHA256 of original train.csv
    logger.info(f"Computing SHA256 of {TRAIN_RAW_PATH}")
    data_sha256 = compute_sha256(TRAIN_RAW_PATH)
    logger.info(f"Data SHA256: {data_sha256}")

    # Load cleaned training data
    logger.info(f"Loading cleaned data from {TRAIN_CLEAN_PATH}")
    df = pd.read_csv(TRAIN_CLEAN_PATH)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Prepare features and target
    X, y = prepare_data(df)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # Get preprocessor
    preprocessor = get_preprocessor()

    # Get models to train
    models_config = get_models_config()

    # Track best model
    best_mae = float("inf")
    best_run_id = None
    best_model_name = None
    best_pipeline = None

    # Train each model
    for model_name, model_config in models_config.items():
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id} for {model_name}")

            # Log parameters
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("test_size", test_size)
            for param_name, param_value in model_config["params"].items():
                mlflow.log_param(f"model_{param_name}", param_value)

            # Train and evaluate
            result = train_and_evaluate(
                model_name=model_name,
                model_config=model_config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                preprocessor=preprocessor,
            )

            # Log metric
            mlflow.log_metric("val_mae", result["mae"])

            # Cross-validation
            cv_results = cross_validate_model(result["pipeline"], X_train, y_train, cv=5)
            mlflow.log_metric("cv_mae_mean", cv_results["cv_mae_mean"])
            mlflow.log_metric("cv_mae_std", cv_results["cv_mae_std"])

            # Log model with cloudpickle serialization
            mlflow.sklearn.log_model(
                sk_model=result["pipeline"],
                artifact_path="model",
                serialization_format="cloudpickle",
                registered_model_name=MODEL_REGISTRY_NAME,
            )

            logger.info(f"{model_name} - Val MAE: {result['mae']:.4f}, CV MAE: {cv_results['cv_mae_mean']:.4f} +/- {cv_results['cv_mae_std']:.4f}")

            # Track best model (by validation MAE)
            if result["mae"] < best_mae:
                best_mae = result["mae"]
                best_run_id = run_id
                best_model_name = model_name
                best_pipeline = result["pipeline"]
                logger.info(f"New best model: {model_name} with MAE: {best_mae:.4f}")

    logger.info(f"Best model: {best_model_name} (run_id: {best_run_id}) with MAE: {best_mae:.4f}")

    # Register best model as version 1 and set champion alias
    if best_run_id:
        # Get the model URI from the best run
        model_uri = f"runs:/{best_run_id}/model"

        # Register the model (this creates version 1 if not exists)
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_REGISTRY_NAME,
        )
        version = registered_model.version
        logger.info(f"Registered model {MODEL_REGISTRY_NAME} version {version}")

        # Set champion alias
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=MODEL_REGISTRY_NAME,
            alias=CHAMPION_ALIAS,
            version=version,
        )
        logger.info(f"Set @{CHAMPION_ALIAS} alias to version {version}")

    # Save best model to models/model.joblib for API
    if best_pipeline is not None:
        save_model(best_pipeline, MODEL_OUTPUT_PATH)

    return best_run_id


def main() -> int:
    """Main entry point."""
    try:
        run_id = train()
        if run_id:
            print(f"Best run_id: {run_id}")
            return 0
        else:
            logger.error("No best model found")
            return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


# Import here to avoid circular import
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    sys.exit(main())