"""Data loading, auditing, and cleaning for MP096 solar energy prediction."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def load_raw_data() -> pd.DataFrame:
    """Load raw training data from CSV."""
    return pd.read_csv(RAW_DIR / "train.csv")


def load_later_data() -> pd.DataFrame:
    """Load later data from CSV."""
    return pd.read_csv(RAW_DIR / "later.csv")


def load_holdout_data() -> pd.DataFrame:
    """Load holdout features from CSV."""
    return pd.read_csv(RAW_DIR / "holdout_features.csv")


def get_data_sha256() -> str:
    """Compute SHA-256 of original train.csv bytes."""
    with open(RAW_DIR / "train.csv", "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def audit_data(df: pd.DataFrame) -> dict[str, Any]:
    """
    Audit the raw training data for the seven data problems.
    Returns a dict with findings matching findings.yaml structure.
    """
    findings = {}

    # 1. Leaking column: meter_reading_kwh only filled after event, missing from holdout
    holdout_df = load_holdout_data()
    leak_present = "meter_reading_kwh" not in holdout_df.columns
    findings["leaking_column"] = {
        "present": leak_present,
        "column": "meter_reading_kwh" if leak_present else None,
    }

    # 2. Unit mixup: check numeric columns per area group for 10x/100x/1000x differences
    numeric_cols = [
        "panel_area_m2", "sunshine_hours", "temp_c", "dust_days",
        "panel_age_yr", "cloud_cover_pct"
    ]
    unit_mixup_found = False
    unit_mixup_details = {}
    for col in numeric_cols:
        stats = df.groupby("area")[col].agg(["min", "mean", "max"])
        ratios = stats["max"] / stats["min"].replace(0, np.nan)
        if (ratios > 10).any() and not ratios.isna().all():
            # Check if it's exactly a unit conversion factor (10, 12, 60, 100, 1000)
            for idx, r in ratios.items():
                if r > 10:
                    for factor in [10, 12, 60, 100, 1000]:
                        if abs(r - factor) < 0.01 * factor:
                            unit_mixup_found = True
                            unit_mixup_details = {
                                "present": True,
                                "column": col,
                                "group_value": idx,
                                "factor": factor,
                            }
                            break
    if not unit_mixup_found:
        unit_mixup_details = {"present": False, "column": None, "group_value": None, "factor": None}
    findings["unit_mixup"] = unit_mixup_details

    # 3. Sentinel value: dust_days has 9999
    sentinel_count = (df["dust_days"] == 9999).sum()
    findings["sentinel_value"] = {
        "present": bool(sentinel_count > 0),
        "column": "dust_days" if sentinel_count > 0 else None,
        "value": 9999 if sentinel_count > 0 else None,
        "count": int(sentinel_count),
    }

    # 4. Duplicate rows
    dup_count = df.duplicated().sum()
    findings["duplicate_rows"] = {
        "present": bool(dup_count > 0),
        "count": int(dup_count),
    }

    # 5. Stuck sensor: temp_c constant for a long run (by date, not row)
    # Group by date and check if all rows for a date have same temp_c
    daily_temp = df.groupby("date")["temp_c"].agg(["nunique", "first"]).reset_index()
    daily_temp = daily_temp[daily_temp["nunique"] == 1].copy()
    daily_temp = daily_temp.sort_values("date").reset_index(drop=True)

    stuck_runs = []
    current_val = None
    run_start_idx = None
    for i, row in daily_temp.iterrows():
        if row["first"] == current_val:
            continue
        else:
            if current_val is not None and (i - run_start_idx) > 7:  # more than a week
                start_date = daily_temp.iloc[run_start_idx]["date"]
                end_date = daily_temp.iloc[i - 1]["date"]
                stuck_runs.append({
                    "column": "temp_c",
                    "value": current_val,
                    "start_date": start_date,
                    "end_date": end_date,
                    "length": i - run_start_idx,
                })
            current_val = row["first"]
            run_start_idx = i
    # Check last run
    if current_val is not None and (len(daily_temp) - run_start_idx) > 7:
        start_date = daily_temp.iloc[run_start_idx]["date"]
        end_date = daily_temp.iloc[-1]["date"]
        stuck_runs.append({
            "column": "temp_c",
            "value": current_val,
            "start_date": start_date,
            "end_date": end_date,
            "length": len(daily_temp) - run_start_idx,
        })

    if stuck_runs:
        longest = max(stuck_runs, key=lambda x: x["length"])
        findings["stuck_sensor"] = {
            "present": True,
            "column": "temp_c",
            "value": float(longest["value"]),
            "start_date": longest["start_date"],
            "end_date": longest["end_date"],
            "length": int(longest["length"]),
        }
    else:
        findings["stuck_sensor"] = {"present": False, "column": None}

    # 6. Target typos: energy_kwh values 10x too large
    # Sort target, look for values far above the rest with a gap
    # Normal range ~10-50, typos are > 100 (about 10x normal)
    typo_mask = df["energy_kwh"] > 100
    typo_count = typo_mask.sum()
    findings["target_typos"] = {
        "present": bool(typo_count > 0),
        "count": int(typo_count),
        "threshold_used": 100.0,
    }

    # 7. Spelling variants: roof_type
    roof_counts = df["roof_type"].value_counts()
    # later.csv has clean spellings - load to compare
    later_df = load_later_data()
    clean_levels = later_df["roof_type"].nunique()
    findings["spelling_variants"] = {
        "present": len(roof_counts) > clean_levels,
        "column": "roof_type",
        "raw_levels": int(len(roof_counts)),
        "true_levels": int(clean_levels),
        "raw_counts": roof_counts.to_dict(),
        "clean_levels": later_df["roof_type"].unique().tolist(),
    }

    return findings


def _get_clean_roof_mapping() -> dict[str, str]:
    """Get mapping from raw roof_type values to clean values using later.csv."""
    later_df = load_later_data()
    clean_levels = set(later_df["roof_type"].unique())
    raw_df = load_raw_data()
    raw_levels = raw_df["roof_type"].unique()

    mapping = {}
    for raw in raw_levels:
        if raw in clean_levels:
            mapping[raw] = raw
        else:
            # Try case-insensitive match
            for clean in clean_levels:
                if raw.lower().strip() == clean.lower().strip():
                    mapping[raw] = clean
                    break
            else:
                # Default to most common clean level
                mapping[raw] = later_df["roof_type"].mode()[0]
    return mapping


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw training data applying all verified fixes.
    Returns a cleaned DataFrame ready for training.
    """
    df = df.copy()

    # 1. Fix target typos: values > 100 are 10x too large (must be before dropping target)
    target_col = "energy_kwh"
    if target_col in df.columns:
        typo_mask = df[target_col] > 100
        df.loc[typo_mask, target_col] = df.loc[typo_mask, target_col] / 10

    # 2. Handle sentinel value in dust_days (9999 -> NaN)
    df.loc[df["dust_days"] == 9999, "dust_days"] = np.nan

    # 3. Handle stuck sensor in temp_c (2026-05-02 to 2026-05-18)
    # Mark those rows as NaN for imputation
    stuck_mask = (df["date"] >= "2026-05-02") & (df["date"] <= "2026-05-18")
    df.loc[stuck_mask, "temp_c"] = np.nan

    # 4. Normalize roof_type spelling variants
    roof_mapping = _get_clean_roof_mapping()
    df["roof_type"] = df["roof_type"].map(roof_mapping).fillna(df["roof_type"])

    # 5. Impute missing numeric values with median
    numeric_cols = [
        "panel_area_m2", "sunshine_hours", "temp_c",
        "dust_days", "panel_age_yr", "cloud_cover_pct"
    ]
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 6. Drop leak columns (meter_reading_kwh, record_id) - keep energy_kwh for training
    leak_cols = ["meter_reading_kwh", "record_id"]
    df = df.drop(columns=[c for c in leak_cols if c in df.columns])

    return df


def save_cleaned_data(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save cleaned data to processed directory."""
    if path is None:
        path = PROCESSED_DIR / "train_clean.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_cleaned_data(path: Path | None = None) -> pd.DataFrame:
    """Load cleaned training data."""
    if path is None:
        path = PROCESSED_DIR / "train_clean.csv"
    return pd.read_csv(path)


def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split cleaned data into features and target."""
    target_col = "energy_kwh"
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found. Data may already be cleaned.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def get_feature_names() -> tuple[list[str], list[str]]:
    """Get numeric and categorical feature names for modeling."""
    numeric = [
        "panel_area_m2", "sunshine_hours", "temp_c",
        "dust_days", "panel_age_yr", "cloud_cover_pct"
    ]
    categorical = ["roof_type", "area"]
    return numeric, categorical


def verify_findings() -> dict[str, Any]:
    """Run audit and verify against findings.yaml."""
    df = load_raw_data()
    audit_results = audit_data(df)

    # Load existing findings.yaml
    findings_path = REPORTS_DIR / "findings.yaml"
    with open(findings_path) as f:
        existing = yaml.safe_load(f)

    # Compare key findings
    comparison = {}
    for key in ["leaking_column", "unit_mixup", "sentinel_value", "duplicate_rows",
                "stuck_sensor", "target_typos", "spelling_variants"]:
        audit_val = audit_results[key].get("present")
        existing_val = existing.get("data_issues", {}).get(key, {}).get("present")
        comparison[key] = {
            "audit": audit_val,
            "existing": existing_val,
            "match": audit_val == existing_val,
        }

    return {"audit": audit_results, "comparison": comparison}