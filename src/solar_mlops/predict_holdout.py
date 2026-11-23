"""Holdout prediction script for solar energy prediction MLOps pipeline.

This script:
1. Loads champion model (from models/model.joblib or MLflow @champion)
2. Loads data/raw/holdout_features.csv
3. Applies same preprocessing as training (handle sentinel 9999 in dust_days,
   spelling variants in roof_type, etc.)
4. Predicts energy_kwh for all 250 rows
5. Saves predictions/predictions/holdout_predictions.csv with exactly two columns:
   record_id,prediction
"""

import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

# Add src to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_mlops.cleaning import (
    SENTINEL_COLUMN,
    SENTINEL_VALUE,
    SPELLING_COLUMN,
    TRUE_ROOF_TYPES,
    detect_spelling_variants,
)
from solar_mlops.model import get_preprocessor
from solar_mlops.utils import setup_logging

logger = setup_logging(__name__)

# Configuration
MODEL_REGISTRY_NAME = "solar-energy-model"
CHAMPION_ALIAS = "champion"
MODEL_JOBLIB_PATH = REPO_ROOT / "models" / "model.joblib"
HOLDOUT_FEATURES_PATH = REPO_ROOT / "data" / "raw" / "holdout_features.csv"
PREDICTIONS_OUTPUT_PATH = REPO_ROOT / "predictions" / "holdout_predictions.csv"
MLFLOW_DB_PATH = REPO_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

# Columns
NUMERIC_FEATURES = [
    "panel_area_m2",
    "sunshine_hours",
    "temp_c",
    "dust_days",
    "panel_age_yr",
    "cloud_cover_pct",
]
CATEGORICAL_FEATURES = ["roof_type", "area"]
TARGET_COLUMN = "energy_kwh"
ID_COLUMN = "record_id"
DATE_COLUMN = "date"
EXCLUDE_COLUMNS = [ID_COLUMN, DATE_COLUMN, "meter_reading_kwh", TARGET_COLUMN]


def load_holdout_features(path: Path = HOLDOUT_FEATURES_PATH) -> pd.DataFrame:
    """
    Load holdout features data from CSV.

    Args:
        path: Path to holdout_features.csv

    Returns:
        DataFrame with holdout features
    """
    logger.info(f"Loading holdout features from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def preprocess_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply same preprocessing as training to holdout data.

    Steps:
    1. Handle sentinel value 9999 in dust_days -> replace with NaN
    2. Standardize spelling variants in roof_type
    3. Ensure all required feature columns exist

    Args:
        df: Raw holdout DataFrame

    Returns:
        Preprocessed DataFrame ready for prediction
    """
    logger.info("Preprocessing holdout data")
    df_clean = df.copy()

    # 1. Handle sentinel value in dust_days
    if SENTINEL_COLUMN in df_clean.columns:
        sentinel_count = (df_clean[SENTINEL_COLUMN] == SENTINEL_VALUE).sum()
        if sentinel_count > 0:
            logger.info(f"Replacing {sentinel_count} sentinel values ({SENTINEL_VALUE}) with NaN in {SENTINEL_COLUMN}")
            df_clean.loc[df_clean[SENTINEL_COLUMN] == SENTINEL_VALUE, SENTINEL_COLUMN] = np.nan

    # 2. Standardize spelling variants in roof_type
    if SPELLING_COLUMN in df_clean.columns:
        spell_result = detect_spelling_variants(df_clean)
        if spell_result.get("present", False):
            logger.info(f"Standardizing spelling variants in {SPELLING_COLUMN}")
            variant_map = spell_result["variant_mapping"]
            df_clean[SPELLING_COLUMN] = df_clean[SPELLING_COLUMN].map(variant_map).fillna(df_clean[SPELLING_COLUMN])

    # 3. Ensure all required feature columns are present and in correct order
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_cols = [col for col in feature_columns if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    # Select only feature columns (in the order expected by the model pipeline)
    X = df_clean[feature_columns].copy()
    logger.info(f"Preprocessed holdout features shape: {X.shape}")
    return X


def load_champion_model() -> tuple:
    """
    Load the champion model.

    Tries to load from models/model.joblib first (faster, used by API).
    Falls back to MLflow model registry @champion alias if file not found.

    Returns:
        Tuple of (model_pipeline, source_description)
    """
    # Try local joblib file first (used by FastAPI)
    if MODEL_JOBLIB_PATH.exists():
        logger.info(f"Loading champion model from {MODEL_JOBLIB_PATH}")
        model = joblib.load(MODEL_JOBLIB_PATH)
        return model, "models/model.joblib"

    # Fallback to MLflow model registry
    logger.info("Local model not found, loading from MLflow @champion alias")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    try:
        model_version = client.get_model_version_by_alias(
            name=MODEL_REGISTRY_NAME, alias=CHAMPION_ALIAS
        )
        version = model_version.version
        logger.info(f"Found champion model version {version}")
    except Exception:
        version = "1"
        logger.warning("No champion alias found, using version 1")

    model_uri = f"models:/{MODEL_REGISTRY_NAME}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Loaded champion model from {model_uri}")
    return model, model_uri


def predict_holdout() -> pd.DataFrame:
    """
    Main prediction function.

    Returns:
        DataFrame with record_id and prediction columns
    """
    logger.info("=" * 60)
    logger.info("STARTING HOLDOUT PREDICTION")
    logger.info("=" * 60)

    # 1. Load holdout features
    holdout_df = load_holdout_features()

    # 2. Keep record_id for output
    record_ids = holdout_df[ID_COLUMN].copy()

    # 3. Preprocess features (same as training)
    X_holdout = preprocess_holdout(holdout_df)

    # 4. Load champion model
    model, model_source = load_champion_model()
    logger.info(f"Model loaded from: {model_source}")

    # 5. Make predictions
    logger.info(f"Making predictions for {len(X_holdout)} rows")
    predictions = model.predict(X_holdout)

    # 6. Create output DataFrame with exactly two columns: record_id, prediction
    output_df = pd.DataFrame({
        "record_id": record_ids,
        "prediction": predictions
    })

    logger.info(f"Predictions complete. Range: {predictions.min():.4f} - {predictions.max():.4f}")
    logger.info(f"Mean prediction: {predictions.mean():.4f}")

    return output_df


def save_predictions(df: pd.DataFrame, path: Path = PREDICTIONS_OUTPUT_PATH) -> None:
    """
    Save predictions to CSV.

    Args:
        df: DataFrame with record_id and prediction columns
        path: Output path
    """
    logger.info(f"Saving predictions to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} predictions to {path}")


def main() -> int:
    """Main entry point."""
    try:
        predictions_df = predict_holdout()
        save_predictions(predictions_df)

        # Print summary
        print("\n" + "=" * 60)
        print("HOLDOUT PREDICTION COMPLETE")
        print("=" * 60)
        print(f"Rows predicted: {len(predictions_df)}")
        print(f"Output file: {PREDICTIONS_OUTPUT_PATH}")
        print(f"Prediction range: {predictions_df['prediction'].min():.4f} - {predictions_df['prediction'].max():.4f}")
        print(f"Mean prediction: {predictions_df['prediction'].mean():.4f}")
        print(f"\nFirst 5 predictions:")
        print(predictions_df.head().to_string(index=False))

        return 0
    except Exception as e:
        logger.exception(f"Holdout prediction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())