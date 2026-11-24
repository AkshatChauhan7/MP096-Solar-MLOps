"""Data cleaning module for Solar MLOps project.

Handles issues identified in findings.yaml:
- Leaking column: meter_reading_kwh (target leakage)
- Sentinel value: dust_days = 9999 (43 occurrences)
- Stuck sensor: temp_c from 2026-05-02 to 2026-05-18
- Target typos: energy_kwh values entered 10 times too large
- Spelling variants: roof_type has 3 true levels (typos like 'tin' vs 'Tin')
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Constants from findings.yaml
SENTINEL_VALUE = 9999
SENTINEL_COLUMN = "dust_days"
SENTINEL_COUNT = 43

STUCK_SENSOR_COLUMN = "temp_c"
STUCK_START_DATE = "2026-05-02"
STUCK_END_DATE = "2026-05-18"

LEAKING_COLUMN = "meter_reading_kwh"
TARGET_TYPOS_COUNT = 10
TARGET_COLUMN = "energy_kwh"
TARGET_TYPO_THRESHOLD = 100
TARGET_TYPO_FACTOR = 10

ROOF_TYPE_COLUMN = "roof_type"
ROOF_TYPE_TRUE_LEVELS = 3
ROOF_TYPE_MAPPING = {
    "tin": "tin",
    "tin.": "tin",
    "flat": "flat",
    "fla.": "flat",
    "sloped": "sloped",
    "slo.": "sloped",
    "slopped": "sloped",
}


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load raw data from CSV."""
    return pd.read_csv(data_path)


def remove_leaking_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the leaking column (target leakage)."""
    if LEAKING_COLUMN in df.columns:
        df = df.drop(columns=[LEAKING_COLUMN])
    return df


def fix_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel values (9999) in dust_days with NaN for imputation."""
    if SENTINEL_COLUMN in df.columns:
        df[SENTINEL_COLUMN] = df[SENTINEL_COLUMN].replace(SENTINEL_VALUE, np.nan)
    return df


def fix_stuck_sensor(df: pd.DataFrame) -> pd.DataFrame:
    """Handle stuck sensor readings in temp_c for the date range 2026-05-02 to 2026-05-18."""
    if "date" in df.columns and STUCK_SENSOR_COLUMN in df.columns:
        # ISO YYYY-MM-DD dates sort chronologically as strings and avoid
        # changing the source date column's dtype during cleaning.
        dates = df["date"].astype("string").str.slice(0, 10)
        mask = (dates >= STUCK_START_DATE) & (dates <= STUCK_END_DATE)
        df.loc[mask, STUCK_SENSOR_COLUMN] = np.nan
    return df


def fix_target_typos(df: pd.DataFrame) -> pd.DataFrame:
    """Correct target values recorded with an extra zero."""
    if TARGET_COLUMN in df.columns:
        mask = df[TARGET_COLUMN] > TARGET_TYPO_THRESHOLD
        df.loc[mask, TARGET_COLUMN] = df.loc[mask, TARGET_COLUMN] / TARGET_TYPO_FACTOR
    return df


def fix_roof_type_spelling(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize roof_type spelling variants to 3 true levels."""
    if ROOF_TYPE_COLUMN in df.columns:
        normalized = df[ROOF_TYPE_COLUMN].astype("string").str.strip().str.lower()
        standardized = normalized.map(ROOF_TYPE_MAPPING)
        df[ROOF_TYPE_COLUMN] = standardized.where(standardized.notna(), df[ROOF_TYPE_COLUMN])
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values after cleaning.

    - Numerical columns: median imputation
    - Categorical columns: mode imputation
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numerical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main cleaning pipeline applying all fixes in order."""
    df = df.copy()

    # 1. Remove leaking column (target leakage)
    df = remove_leaking_column(df)

    # 2. Fix sentinel values
    df = fix_sentinel_values(df)

    # 3. Fix stuck sensor
    df = fix_stuck_sensor(df)

    # 4. Fix target typos (placeholder)
    df = fix_target_typos(df)

    # 5. Fix roof_type spelling variants
    df = fix_roof_type_spelling(df)

    # 6. Impute missing values
    df = impute_missing_values(df)

    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save cleaned data to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    """Main entry point for data cleaning."""
    raw_train_path = Path("data/raw/train.csv")
    raw_later_path = Path("data/raw/later.csv")
    holdout_path = Path("data/raw/holdout_features.csv")

    # Since DVC files are present, we need to pull data first
    # For now, assume data is available
    if not raw_train_path.exists():
        print(f"Data not found at {raw_train_path}. Run 'dvc pull' first.")
        return

    # Clean train data
    train_df = load_data(raw_train_path)
    train_clean = clean_data(train_df)
    save_cleaned_data(train_clean, "data/processed/train_clean.csv")
    print(f"Cleaned train data saved: {train_clean.shape}")

    # Clean later data
    if raw_later_path.exists():
        later_df = load_data(raw_later_path)
        later_clean = clean_data(later_df)
        save_cleaned_data(later_clean, "data/processed/later_clean.csv")
        print(f"Cleaned later data saved: {later_clean.shape}")

    # Clean holdout data (features only, no target)
    if holdout_path.exists():
        holdout_df = load_data(holdout_path)
        holdout_clean = clean_data(holdout_df)
        save_cleaned_data(holdout_clean, "data/processed/holdout_clean.csv")
        print(f"Cleaned holdout data saved: {holdout_clean.shape}")


if __name__ == "__main__":
    main()
