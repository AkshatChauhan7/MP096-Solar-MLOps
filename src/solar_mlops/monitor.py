"""Monitoring module for solar energy prediction MLOps pipeline.

This module implements:
1. PSI (Population Stability Index) computation for drift detection
2. MAE evaluation per area group
3. Model retraining with new data
4. Champion model promotion based on MAE improvement threshold
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from .cleaning import clean_data, run_cleaning_pipeline
from .model import get_preprocessor, get_models_config, prepare_data, train_and_evaluate
from .utils import setup_logging

logger = setup_logging(__name__)

# Configuration constants
REPO_ROOT = Path(__file__).parent.parent.parent
MLFLOW_DB_PATH = REPO_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MODEL_REGISTRY_NAME = "solar-energy-model"
CHAMPION_ALIAS = "champion"
MODEL_OUTPUT_PATH = REPO_ROOT / "models" / "model.joblib"
TRAIN_CLEAN_PATH = REPO_ROOT / "data" / "processed" / "train_clean.csv"
LATER_RAW_PATH = REPO_ROOT / "data" / "raw" / "later.csv"
PARAMS_PATH = REPO_ROOT / "params.yaml"

NUMERIC_FEATURES = [
    "panel_area_m2",
    "sunshine_hours",
    "temp_c",
    "dust_days",
    "panel_age_yr",
    "cloud_cover_pct",
]
TARGET_COLUMN = "energy_kwh"
GROUP_COLUMN = "area"
DATE_COLUMN = "date"


def load_params() -> dict:
    """Load parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def setup_mlflow() -> None:
    """Configure MLflow tracking."""
    logger.info(f"Setting MLflow tracking URI to {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("solar-energy-monitoring")


def compute_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.

    PSI = sum((actual_pct - expected_pct) * log(actual_pct / expected_pct))

    Args:
        expected: Reference distribution (training data)
        actual: Current distribution (new data)
        bins: Number of bins for histogram (default 10)

    Returns:
        PSI value (0 = no drift, >0.2 = significant drift)
    """
    # Remove NaN values
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Create bins based on expected distribution quantiles
    try:
        _, bin_edges = pd.qcut(expected, q=bins, retbins=True, duplicates="drop")
    except ValueError:
        # If qcut fails (e.g., constant values), use uniform bins
        bin_edges = np.linspace(expected.min(), expected.max(), bins + 1)

    # Ensure bin edges cover the full range of both distributions
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Compute histograms
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to percentages (avoid division by zero)
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    # Replace zeros with small epsilon to avoid log(0)
    epsilon = 1e-10
    expected_pct = np.maximum(expected_pct, epsilon)
    actual_pct = np.maximum(actual_pct, epsilon)

    # Compute PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def compute_all_psi(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    numeric_cols: List[str],
    bins: int = 10,
) -> Dict[str, float]:
    """
    Compute PSI for all numeric columns between train and later data.

    Args:
        train_df: Cleaned training DataFrame
        later_df: Cleaned later DataFrame
        numeric_cols: List of numeric column names
        bins: Number of bins for PSI computation

    Returns:
        Dictionary mapping column name to PSI value
    """
    psi_results = {}
    for col in numeric_cols:
        if col in train_df.columns and col in later_df.columns:
            psi_val = compute_psi(train_df[col], later_df[col], bins=bins)
            psi_results[col] = psi_val
            logger.info(f"PSI for {col}: {psi_val:.4f}")
        else:
            logger.warning(f"Column {col} not found in both DataFrames")
            psi_results[col] = 0.0

    return psi_results


def load_champion_model() -> Tuple[Any, str]:
    """
    Load the champion model from MLflow model registry.

    Returns:
        Tuple of (model_pipeline, model_version)
    """
    setup_mlflow()
    client = mlflow.MlflowClient()

    # Try to get model by champion alias first
    try:
        model_version = client.get_model_version_by_alias(
            name=MODEL_REGISTRY_NAME, alias=CHAMPION_ALIAS
        )
        version = model_version.version
        logger.info(f"Found champion model version {version}")
    except Exception:
        # Fallback to version 1
        version = "1"
        logger.info("No champion alias found, using version 1")

    model_uri = f"models:/{MODEL_REGISTRY_NAME}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Loaded champion model from {model_uri}")

    return model, str(version)


def evaluate_model_per_group(
    model: Any,
    df: pd.DataFrame,
    group_col: str = GROUP_COLUMN,
) -> Dict[str, float]:
    """
    Evaluate model MAE per group (area).

    Args:
        model: Trained pipeline
        df: DataFrame with features and target
        group_col: Column to group by

    Returns:
        Dictionary mapping group name to MAE
    """
    X, y = prepare_data(df)
    y_pred = model.predict(X)

    # Add predictions and group info for per-group evaluation
    results_df = df[[group_col]].copy()
    results_df["y_true"] = y.values
    results_df["y_pred"] = y_pred

    mae_per_group = {}
    for group_val in results_df[group_col].unique():
        group_data = results_df[results_df[group_col] == group_val]
        mae = mean_absolute_error(group_data["y_true"], group_data["y_pred"])
        mae_per_group[group_val] = float(mae)
        logger.info(f"MAE for {group_col}={group_val}: {mae:.4f}")

    return mae_per_group


def retrain_model(
    train_df: pd.DataFrame,
    later_train_df: pd.DataFrame,
    model_name: str = "RandomForest",
) -> Tuple[Any, float]:
    """
    Retrain a model on combined training data.

    Args:
        train_df: Cleaned original training data
        later_train_df: First 70% of later data (by date)
        model_name: Name of model to train

    Returns:
        Tuple of (trained_pipeline, validation_mae)
    """
    # Combine datasets
    combined_df = pd.concat([train_df, later_train_df], ignore_index=True)
    logger.info(f"Combined training data: {len(combined_df)} rows "
                f"({len(train_df)} train + {len(later_train_df)} later)")

    # Prepare data
    X, y = prepare_data(combined_df)

    # Split for validation
    params = load_params()
    test_size = params["model_settings"]["test_size"]
    random_state = params["model_settings"]["random_state"]

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Get model config
    models_config = get_models_config()
    if model_name not in models_config:
        model_name = "RandomForest"
        logger.warning(f"Model {model_name} not found, using RandomForest")

    model_config = models_config[model_name]
    preprocessor = get_preprocessor()

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

    logger.info(f"Retrained {model_name} validation MAE: {result['mae']:.4f}")
    return result["pipeline"], result["mae"]


def evaluate_on_test_set(
    model: Any,
    test_df: pd.DataFrame,
) -> float:
    """
    Evaluate model on test set.

    Args:
        model: Trained pipeline
        test_df: Test DataFrame with features and target

    Returns:
        MAE score
    """
    X, y = prepare_data(test_df)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    return float(mae)


def register_new_champion(
    model: Any,
    run_name: str = "monitoring-retrain",
) -> str:
    """
    Register a new model version and set as champion.

    Args:
        model: Trained pipeline to register
        run_name: Name for the MLflow run

    Returns:
        New model version string
    """
    setup_mlflow()
    client = mlflow.MlflowClient()

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id} for {run_name}")

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

    # Register the model (creates new version)
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_REGISTRY_NAME,
    )
    version = registered_model.version
    logger.info(f"Registered model {MODEL_REGISTRY_NAME} version {version}")

    # Set champion alias to new version
    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias=CHAMPION_ALIAS,
        version=version,
    )
    logger.info(f"Set @{CHAMPION_ALIAS} alias to version {version}")

    return str(version)


def save_champion_model(model: Any) -> None:
    """Save champion model to models/model.joblib for API."""
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logger.info(f"Champion model saved to {MODEL_OUTPUT_PATH}")


def run_monitoring_pipeline() -> Dict[str, Any]:
    """
    Run the complete monitoring pipeline.

    Returns:
        Dictionary with all computed metrics for findings.yaml
    """
    logger.info("Starting monitoring pipeline")

    # 1. Load cleaned train.csv and later.csv
    logger.info("Loading cleaned training data")
    train_df = pd.read_csv(TRAIN_CLEAN_PATH)

    logger.info("Loading and cleaning later data")
    later_raw_df = pd.read_csv(LATER_RAW_PATH)
    later_df = clean_data(later_raw_df)

    # 2. Compute PSI (10 bins) for all 6 numeric columns
    logger.info("Computing PSI for numeric features")
    params = load_params()
    psi_bins = params["monitoring"]["psi_bins"]
    psi_results = compute_all_psi(train_df, later_df, NUMERIC_FEATURES, bins=psi_bins)

    # 3. Identify most_drifted_feature (highest PSI)
    most_drifted_feature = max(psi_results, key=psi_results.get)
    max_psi = psi_results[most_drifted_feature]
    logger.info(f"Most drifted feature: {most_drifted_feature} (PSI={max_psi:.4f})")

    # 4. Load v1 model (champion) and compute MAE on later.csv per area group
    logger.info("Loading champion model (v1)")
    v1_model, v1_version = load_champion_model()

    logger.info("Evaluating v1 model per area group on later.csv")
    v1_mae_per_group = evaluate_model_per_group(v1_model, later_df)

    # 5. Identify worst_served_group (highest MAE)
    worst_served_group = max(v1_mae_per_group, key=v1_mae_per_group.get)
    worst_group_mae = v1_mae_per_group[worst_served_group]
    logger.info(f"Worst served group: {worst_served_group} (MAE={worst_group_mae:.4f})")

    # 6. Sort later.csv by date, take first 70% for retraining
    logger.info("Splitting later.csv by date (70/30)")
    later_df_sorted = later_df.sort_values(DATE_COLUMN).reset_index(drop=True)
    split_idx = int(len(later_df_sorted) * params["monitoring"]["retrain_split"])
    later_train_df = later_df_sorted.iloc[:split_idx].copy()
    later_test_df = later_df_sorted.iloc[split_idx:].copy()
    logger.info(f"Later train: {len(later_train_df)} rows, Later test: {len(later_test_df)} rows")

    # 7. Retrain v2 on cleaned train.csv + first 70% of later.csv
    logger.info("Retraining v2 model")
    v2_model, v2_val_mae = retrain_model(train_df, later_train_df, model_name="RandomForest")

    # 8. Compute MAE of v1 and v2 on last 30% of later.csv
    logger.info("Evaluating v1 and v2 on test set (last 30% of later.csv)")
    v1_test_mae = evaluate_on_test_set(v1_model, later_test_df)
    v2_test_mae = evaluate_on_test_set(v2_model, later_test_df)
    logger.info(f"v1 test MAE: {v1_test_mae:.4f}, v2 test MAE: {v2_test_mae:.4f}")

    # 9. Apply promotion rule: promote v2 only if MAE_v2 <= MAE_v1 * 0.88 (12% lower)
    promotion_threshold = params["monitoring"]["promotion_threshold"]
    improvement_ratio = v2_test_mae / v1_test_mae if v1_test_mae > 0 else 1.0
    promote = v2_test_mae <= v1_test_mae * (1 - promotion_threshold)
    logger.info(f"Promotion check: v2 MAE ({v2_test_mae:.4f}) <= v1 MAE ({v1_test_mae:.4f}) * "
                f"(1 - {promotion_threshold}) = {v1_test_mae * (1 - promotion_threshold):.4f}")
    logger.info(f"Improvement ratio: {improvement_ratio:.4f}, Promote: {promote}")

    # 10. If promote: register v2 as new version, set @champion alias to version 2
    champion_version = v1_version
    champion_model = v1_model
    champion_test_mae = v1_test_mae

    if promote:
        logger.info("Promoting v2 to champion")
        new_version = register_new_champion(v2_model, run_name="monitoring-retrain-v2")
        champion_version = new_version
        champion_model = v2_model
        champion_test_mae = v2_test_mae

        # 11. Update models/model.joblib with champion model
        save_champion_model(champion_model)
    else:
        logger.info("Keeping v1 as champion")
        # Ensure v1 is still saved as champion
        save_champion_model(v1_model)

    # 12. Return all metrics for findings.yaml
    results = {
        "psi": psi_results,
        "most_drifted_feature": most_drifted_feature,
        "max_psi": max_psi,
        "v1_version": v1_version,
        "v1_mae_per_group": v1_mae_per_group,
        "v1_test_mae": v1_test_mae,
        "worst_served_group": worst_served_group,
        "worst_group_mae": worst_group_mae,
        "v2_val_mae": v2_val_mae,
        "v2_test_mae": v2_test_mae,
        "improvement_ratio": improvement_ratio,
        "promotion_threshold": promotion_threshold,
        "promoted": promote,
        "champion_version": champion_version,
        "champion_test_mae": champion_test_mae,
        "later_train_size": len(later_train_df),
        "later_test_size": len(later_test_df),
        "train_size": len(train_df),
    }

    logger.info("Monitoring pipeline completed")
    return results


def save_findings(results: Dict[str, Any], output_path: Optional[Path] = None) -> None:
    """Save monitoring results to findings.yaml."""
    if output_path is None:
        output_path = REPO_ROOT / "reports" / "findings.yaml"

    # Load existing findings if any
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = yaml.safe_load(f) or {}

    # Update with monitoring results
    existing["monitoring"] = {
        "psi": {k: round(v, 4) for k, v in results["psi"].items()},
        "most_drifted_feature": results["most_drifted_feature"],
        "max_psi": round(results["max_psi"], 4),
        "v1_version": results["v1_version"],
        "v1_mae_per_group": {k: round(v, 4) for k, v in results["v1_mae_per_group"].items()},
        "v1_test_mae": round(results["v1_test_mae"], 4),
        "worst_served_group": results["worst_served_group"],
        "worst_group_mae": round(results["worst_group_mae"], 4),
        "v2_val_mae": round(results["v2_val_mae"], 4),
        "v2_test_mae": round(results["v2_test_mae"], 4),
        "improvement_ratio": round(results["improvement_ratio"], 4),
        "promotion_threshold": results["promotion_threshold"],
        "promoted": results["promoted"],
        "champion_version": results["champion_version"],
        "champion_test_mae": round(results["champion_test_mae"], 4),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Findings saved to {output_path}")


if __name__ == "__main__":
    import sys

    # Run monitoring pipeline
    try:
        results = run_monitoring_pipeline()
        save_findings(results)

        # Print summary
        print("\n" + "=" * 60)
        print("MONITORING PIPELINE RESULTS")
        print("=" * 60)
        print(f"Most drifted feature: {results['most_drifted_feature']} (PSI={results['max_psi']:.4f})")
        print(f"Worst served group: {results['worst_served_group']} (MAE={results['worst_group_mae']:.4f})")
        print(f"v1 test MAE: {results['v1_test_mae']:.4f}")
        print(f"v2 test MAE: {results['v2_test_mae']:.4f}")
        print(f"Improvement ratio: {results['improvement_ratio']:.4f}")
        print(f"Promoted: {results['promoted']}")
        print(f"Champion version: {results['champion_version']}")
        print(f"Champion test MAE: {results['champion_test_mae']:.4f}")

        sys.exit(0)
    except Exception as e:
        logger.exception(f"Monitoring pipeline failed: {e}")
        sys.exit(1)