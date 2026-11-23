"""Tests for data auditing and cleaning."""

import numpy as np
import pandas as pd
import pytest

from src.solar_ml.data import (
    load_raw_data,
    load_later_data,
    audit_data,
    clean_data,
    _get_clean_roof_mapping,
    verify_findings,
    prepare_features_target,
    get_feature_names,
    get_data_sha256,
)


class TestDataLoading:
    """Test data loading functions."""

    def test_load_raw_data(self):
        df = load_raw_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "energy_kwh" in df.columns
        assert "meter_reading_kwh" in df.columns

    def test_load_later_data(self):
        df = load_later_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "energy_kwh" in df.columns
        assert "meter_reading_kwh" in df.columns  # later.csv has both

    def test_data_sha256_consistent(self):
        sha1 = get_data_sha256()
        sha2 = get_data_sha256()
        assert sha1 == sha2
        assert len(sha1) == 64  # SHA-256 hex length


class TestDataAudit:
    """Test data audit findings."""

    def test_audit_returns_all_seven_problems(self):
        df = load_raw_data()
        findings = audit_data(df)

        expected_keys = [
            "leaking_column", "unit_mixup", "sentinel_value",
            "duplicate_rows", "stuck_sensor", "target_typos", "spelling_variants"
        ]
        for key in expected_keys:
            assert key in findings, f"Missing audit key: {key}"
            assert "present" in findings[key]

    def test_leaking_column_detected(self):
        df = load_raw_data()
        findings = audit_data(df)
        leak = findings["leaking_column"]
        assert leak["present"] is True
        assert leak["column"] == "meter_reading_kwh"

    def test_unit_mixup_absent(self):
        df = load_raw_data()
        findings = audit_data(df)
        unit = findings["unit_mixup"]
        assert unit["present"] is False

    def test_sentinel_value_detected(self):
        df = load_raw_data()
        findings = audit_data(df)
        sentinel = findings["sentinel_value"]
        assert sentinel["present"] is True
        assert sentinel["column"] == "dust_days"
        assert sentinel["value"] == 9999
        assert sentinel["count"] == 43  # Per findings.yaml

    def test_duplicate_rows_absent(self):
        df = load_raw_data()
        findings = audit_data(df)
        dup = findings["duplicate_rows"]
        assert dup["present"] is False
        assert dup["count"] == 0

    def test_stuck_sensor_detected(self):
        df = load_raw_data()
        findings = audit_data(df)
        stuck = findings["stuck_sensor"]
        assert stuck["present"] is True
        assert stuck["column"] == "temp_c"
        assert stuck["start_date"] == "2026-05-02"
        assert stuck["end_date"] == "2026-05-18"

    def test_target_typos_detected(self):
        df = load_raw_data()
        findings = audit_data(df)
        typos = findings["target_typos"]
        assert typos["present"] is True
        assert typos["count"] == 10  # Per findings.yaml

    def test_spelling_variants_detected(self):
        df = load_raw_data()
        findings = audit_data(df)
        variants = findings["spelling_variants"]
        assert variants["present"] is True
        assert variants["column"] == "roof_type"
        assert variants["true_levels"] == 3
        assert variants["raw_levels"] > 3


class TestDataCleaning:
    """Test data cleaning transformations."""

    def test_clean_drops_leak_columns(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        # meter_reading_kwh and record_id are leaks; energy_kwh is target (kept for training)
        for col in ["meter_reading_kwh", "record_id"]:
            assert col not in clean_df.columns
        assert "energy_kwh" in clean_df.columns  # Target kept for training

    def test_clean_sentinel_to_nan(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        # Sentinel values should be NaN (then imputed)
        assert clean_df["dust_days"].isna().sum() == 0  # Imputed
        assert (clean_df["dust_days"] == 9999).sum() == 0

    def test_clean_stuck_sensor_to_nan(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        # Stuck sensor period should be imputed
        stuck_mask = (df["date"] >= "2026-05-02") & (df["date"] <= "2026-05-18")
        assert clean_df.loc[stuck_mask, "temp_c"].isna().sum() == 0  # Imputed

    def test_clean_fixes_target_typos(self):
        df = load_raw_data()
        # Find typo rows before cleaning
        p95 = df["energy_kwh"].quantile(0.95)
        typo_mask = df["energy_kwh"] > 10 * p95
        typo_indices = df[typo_mask].index

        clean_df = clean_data(df)
        # After cleaning, those values should be divided by 10
        for idx in typo_indices:
            if idx in clean_df.index:
                original = df.loc[idx, "energy_kwh"]
                cleaned = clean_df.loc[idx, "energy_kwh"]
                assert abs(cleaned - original / 10) < 0.01

    def test_clean_normalizes_roof_type(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        # All roof_type values should match later.csv clean levels
        later_df = load_later_data()
        clean_levels = set(later_df["roof_type"].unique())
        for val in clean_df["roof_type"].unique():
            assert val in clean_levels, f"Unclean roof_type: {val}"

    def test_clean_no_missing_values(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        numeric_cols = [
            "panel_area_m2", "sunshine_hours", "temp_c",
            "dust_days", "panel_age_yr", "cloud_cover_pct"
        ]
        for col in numeric_cols:
            assert clean_df[col].isna().sum() == 0, f"Missing values in {col}"

    def test_prepare_features_target(self):
        df = load_raw_data()
        clean_df = clean_data(df)
        X, y = prepare_features_target(clean_df)

        assert "energy_kwh" not in X.columns
        assert y.name == "energy_kwh"
        assert len(X) == len(y)
        assert len(X) == len(clean_df)

    def test_get_feature_names(self):
        numeric, categorical = get_feature_names()
        assert "panel_area_m2" in numeric
        assert "sunshine_hours" in numeric
        assert "temp_c" in numeric
        assert "dust_days" in numeric
        assert "panel_age_yr" in numeric
        assert "cloud_cover_pct" in numeric
        assert "roof_type" in categorical
        assert "area" in categorical
        assert len(numeric) == 6
        assert len(categorical) == 2


class TestVerifyFindings:
    """Test findings verification against findings.yaml."""

    def test_verify_findings_matches(self):
        result = verify_findings()
        assert "audit" in result
        assert "comparison" in result

        for key, comp in result["comparison"].items():
            assert comp["match"] is True, f"Finding mismatch for {key}: audit={comp['audit']}, existing={comp['existing']}"


class TestRoofMapping:
    """Test roof type normalization mapping."""

    def test_mapping_covers_all_raw_values(self):
        df = load_raw_data()
        mapping = _get_clean_roof_mapping()
        raw_values = set(df["roof_type"].unique())
        for val in raw_values:
            assert val in mapping, f"No mapping for {val}"
            assert mapping[val] in {"flat", "sloped", "tin"}  # Known clean levels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])