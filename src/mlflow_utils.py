"""MLflow tracking and model registry utilities for Solar MLOps project."""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json


# Configuration
EXPERIMENT_NAME = "solar-mlops-mp096"
REGISTERED_MODEL_NAME = "solar-power-predictor"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Local SQLite for development


def setup_mlflow(tracking_uri: str = MLFLOW_TRACKING_URI) -> None:
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def get_or_create_experiment(experiment_name: str = EXPERIMENT_NAME) -> str:
    """Get experiment ID, create if doesn't exist."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    return client.create_experiment(experiment_name)


def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to MLflow."""
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to MLflow."""
    mlflow.log_metrics(metrics, step=step)


def log_model(
    model: Any,
    artifact_path: str = "model",
    signature: Optional[Any] = None,
    input_example: Optional[pd.DataFrame] = None,
    registered_model_name: Optional[str] = None,
) -> str:
    """Log model to MLflow and optionally register it."""
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        serialization_format="cloudpickle",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
    )
    return model_info.model_uri


def register_model(model_uri: str, model_name: str = REGISTERED_MODEL_NAME) -> Any:
    """Register a model in the MLflow Model Registry."""
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    return mv


def set_model_alias(
    model_name: str = REGISTERED_MODEL_NAME,
    alias: str = "champion",
    version: int | str = 1,
) -> None:
    """Assign a registry alias to a model version."""
    MlflowClient().set_registered_model_alias(model_name, alias, str(version))


def transition_model_stage(
    model_name: str,
    version: int,
    stage: str,  # "Staging", "Production", "Archived"
    archive_existing_versions: bool = True,
) -> None:
    """Transition model version to a stage."""
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )


def get_latest_model_version(model_name: str = REGISTERED_MODEL_NAME) -> Optional[int]:
    """Get the latest version of a registered model."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    return max(v.version for v in versions)


def get_model_by_stage(model_name: str = REGISTERED_MODEL_NAME, stage: str = "Production") -> Optional[Any]:
    """Load a model from the registry by stage."""
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return None
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.sklearn.load_model(model_uri)


def log_artifacts(local_dir: str, artifact_path: Optional[str] = None) -> None:
    """Log artifacts (files, plots, etc.) to MLflow."""
    mlflow.log_artifacts(local_dir, artifact_path)


def log_figure(figure: Any, artifact_file: str) -> None:
    """Log a matplotlib figure to MLflow."""
    mlflow.log_figure(figure, artifact_file)


def get_run_info(run_id: str) -> Dict[str, Any]:
    """Get run information including params, metrics, artifacts."""
    client = MlflowClient()
    run = client.get_run(run_id)
    return {
        "run_id": run_id,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "tags": run.data.tags,
        "artifact_uri": run.info.artifact_uri,
    }


def search_runs(
    filter_string: str = "",
    order_by: list = None,
    max_results: int = 100,
) -> list:
    """Search for runs in the experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        return []
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=order_by or ["metrics.mae ASC"],
        max_results=max_results,
    )
    return runs


def log_dataset_info(
    train_path: str,
    test_path: str,
    train_shape: tuple,
    test_shape: tuple,
    features: list,
    target: str,
) -> None:
    """Log dataset information as parameters."""
    mlflow.log_params({
        "train_path": train_path,
        "test_path": test_path,
        "train_rows": train_shape[0],
        "train_cols": train_shape[1],
        "test_rows": test_shape[0],
        "test_cols": test_shape[1],
        "n_features": len(features),
        "target": target,
        "features": json.dumps(features),
    })


def create_model_signature(X_train: pd.DataFrame, y_pred: np.ndarray) -> Any:
    """Create MLflow model signature from training data and predictions."""
    return infer_signature(X_train, y_pred)


def log_confusion_matrix(cm: np.ndarray, labels: list, artifact_file: str = "confusion_matrix.json") -> None:
    """Log confusion matrix as JSON artifact."""
    cm_dict = {
        "matrix": cm.tolist(),
        "labels": labels,
    }
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cm_dict, f)
        mlflow.log_artifact(f.name, artifact_file)


class MLflowTracker:
    """Context manager for MLflow runs with automatic cleanup."""

    def __init__(
        self,
        run_name: Optional[str] = None,
        experiment_name: str = EXPERIMENT_NAME,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.tags = tags or {}
        self.run = None

    def __enter__(self):
        setup_mlflow()
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run:
            mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

    def log_params(self, params: Dict[str, Any]) -> None:
        log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        log_metrics(metrics, step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        signature: Optional[Any] = None,
        input_example: Optional[pd.DataFrame] = None,
    ) -> str:
        # Candidate runs are tracked first; only the selected model is registered.
        return log_model(model, artifact_path, signature, input_example)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        log_artifacts(local_dir, artifact_path)


def promote_model_to_production(model_name: str = REGISTERED_MODEL_NAME) -> bool:
    """Promote the best model (lowest MAE in Staging) to Production."""
    client = MlflowClient()

    # Get all versions in Staging
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        print("No model in Staging to promote")
        return False

    # Find the one with best (lowest) MAE
    best_version = None
    best_mae = float("inf")

    for version in staging_versions:
        run = client.get_run(version.run_id)
        mae = run.data.metrics.get("mae", float("inf"))
        if mae < best_mae:
            best_mae = mae
            best_version = version.version

    if best_version:
        transition_model_stage(model_name, best_version, "Production")
        print(f"Promoted model version {best_version} to Production (MAE: {best_mae})")
        return True

    return False


def archive_old_production_models(model_name: str = REGISTERED_MODEL_NAME, keep: int = 2) -> None:
    """Archive old production models, keeping only the latest `keep` versions."""
    client = MlflowClient()
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])

    if len(prod_versions) > keep:
        # Sort by version number descending
        prod_versions.sort(key=lambda v: v.version, reverse=True)
        for version in prod_versions[keep:]:
            transition_model_stage(model_name, version.version, "Archived")
            print(f"Archived model version {version.version}")


if __name__ == "__main__":
    # Test MLflow setup
    setup_mlflow()
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {EXPERIMENT_NAME}")

    exp_id = get_or_create_experiment()
    print(f"Experiment ID: {exp_id}")
