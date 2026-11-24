"""Unit tests for data cleaning module."""

import pytest
import pandas as pd
import numpy as np
from src.cleaning import (
    clean_data,
    remove_leaking_column,
    fix_sentinel_values,
    fix_stuck_sensor,
    fix_roof_type_spelling,
    fix_target_typos,
    impute_missing_values,
    SENTINEL_VALUE,
    SENTINEL_COLUMN,
    STUCK_SENSOR_COLUMN,
    STUCK_START_DATE,
    STUCK_END_DATE,
    LEAKING_COLUMN,
    ROOF_TYPE_COLUMN,
    ROOF_TYPE_MAPPING,
)


class TestRemoveLeakingColumn:
    """Tests for remove_leaking_column function."""

    def test_removes_leaking_column(self):
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "meter_reading_kwh": [10, 20, 30],  # leaking column
            "feature2": [4, 5, 6],
        })
        result = remove_leaking_column(df)
        assert LEAKING_COLUMN not in result.columns
        assert "feature1" in result.columns
        assert "feature2" in result.columns

    def test_noop_if_column_missing(self):
        df = pd.DataFrame({"feature1": [1, 2, 3]})
        result = remove_leaking_column(df)
        assert list(result.columns) == ["feature1"]


class TestFixSentinelValues:
    """Tests for fix_sentinel_values function."""

    def test_replaces_sentinel_with_nan(self):
        df = pd.DataFrame({
            "dust_days": [1, 2, SENTINEL_VALUE, 4, SENTINEL_VALUE],
            "other": [10, 20, 30, 40, 50],
        })
        result = fix_sentinel_values(df)
        assert result[SENTINEL_COLUMN].isna().sum() == 2
        assert result.loc[0, SENTINEL_COLUMN] == 1
        assert result.loc[1, SENTINEL_COLUMN] == 2
        assert result.loc[3, SENTINEL_COLUMN] == 4

    def test_noop_if_column_missing(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = fix_sentinel_values(df)
        assert list(result.columns) == ["other"]


class TestFixStuckSensor:
    """Tests for fix_stuck_sensor function."""

    def test_replaces_stuck_range_with_nan(self):
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2026-05-01", "2026-05-02", "2026-05-10",
                "2026-05-18", "2026-05-19"
            ]),
            "temp_c": [20, 25, 25, 25, 22],
        })
        result = fix_stuck_sensor(df)
        # Rows 1, 2, 3 (indices) should be NaN (dates 2026-05-02 to 2026-05-18)
        assert result.loc[1, STUCK_SENSOR_COLUMN] != 25 or pd.isna(result.loc[1, STUCK_SENSOR_COLUMN])
        assert pd.isna(result.loc[1, STUCK_SENSOR_COLUMN])
        assert pd.isna(result.loc[2, STUCK_SENSOR_COLUMN])
        assert pd.isna(result.loc[3, STUCK_SENSOR_COLUMN])
        # Outside range should remain
        assert result.loc[0, STUCK_SENSOR_COLUMN] == 20
        assert result.loc[4, STUCK_SENSOR_COLUMN] == 22

    def test_handles_string_dates(self):
        df = pd.DataFrame({
            "date": ["2026-05-01", "2026-05-05", "2026-05-20"],
            "temp_c": [20, 25, 22],
        })
        result = fix_stuck_sensor(df)
        assert pd.isna(result.loc[1, STUCK_SENSOR_COLUMN])
        assert result.loc[0, STUCK_SENSOR_COLUMN] == 20
        assert result.loc[2, STUCK_SENSOR_COLUMN] == 22


class TestFixRoofTypeSpelling:
    """Tests for fix_roof_type_spelling function."""

    def test_standardizes_known_variants(self):
        df = pd.DataFrame({
            "roof_type": ["tin", " TIN ", "flat", "fla.", "sloped", "slo.", "sloped "],
        })
        result = fix_roof_type_spelling(df)
        expected = ["tin", "tin", "flat", "flat", "sloped", "sloped", "sloped"]
        assert list(result[ROOF_TYPE_COLUMN]) == expected

    def test_preserves_unknown_values(self):
        df = pd.DataFrame({"roof_type": ["unknown_type"]})
        result = fix_roof_type_spelling(df)
        assert result.loc[0, ROOF_TYPE_COLUMN] == "unknown_type"

    def test_noop_if_column_missing(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = fix_roof_type_spelling(df)
        assert list(result.columns) == ["other"]


class TestImputeMissingValues:
    """Tests for impute_missing_values function."""

    def test_imputes_numerical_with_median(self):
        df = pd.DataFrame({
            "num_col": [1, 2, np.nan, 4, 5],
            "cat_col": ["a", "b", "a", "b", "a"],
        })
        result = impute_missing_values(df)
        assert result["num_col"].isna().sum() == 0
        assert result.loc[2, "num_col"] == 3  # median of [1,2,4,5] = 3

    def test_imputes_categorical_with_mode(self):
        df = pd.DataFrame({
            "num_col": [1, 2, 3, 4, 5],
            "cat_col": ["a", "b", np.nan, "b", "a"],
        })
        result = impute_missing_values(df)
        assert result["cat_col"].isna().sum() == 0
        assert result.loc[2, "cat_col"] == "a"  # mode is "a"

    def test_handles_all_nan_column(self):
        df = pd.DataFrame({"col": [np.nan, np.nan, np.nan]})
        result = impute_missing_values(df)
        # Should not crash, fills with something


class TestFixTargetTypos:
    def test_divides_tenfold_target_values(self):
        df = pd.DataFrame({"energy_kwh": [20.0, 50.76, 233.9, 409.5]})
        result = fix_target_typos(df)
        assert list(result["energy_kwh"]) == pytest.approx([20.0, 50.76, 23.39, 40.95])


class TestCleanDataIntegration:
    """Integration tests for the full clean_data pipeline."""

    def test_full_cleaning_pipeline(self):
        """Test the complete cleaning pipeline on synthetic data."""
        df = pd.DataFrame({
            "record_id": ["R1", "R2", "R3", "R4", "R5"],
            "date": ["2026-05-01", "2026-05-05", "2026-05-10", "2026-05-18", "2026-05-20"],
            "panel_area_m2": [20, 30, 40, 50, 60],
            "sunshine_hours": [5, 6, 7, 8, 9],
            "temp_c": [25, 30, 30, 30, 28],  # stuck at 30 for 3 days
            "dust_days": [1, 2, SENTINEL_VALUE, 4, 5],
            "panel_age_yr": [5, 10, 15, 20, 25],
            "cloud_cover_pct": [10, 20, 30, 40, 50],
            "roof_type": ["tin", "Tin", "flat", "slopped", "sloped"],
            "area": ["Old City", "New Colony", "Industrial", "Old City", "New Colony"],
            "meter_reading_kwh": [100, 200, 300, 400, 500],  # leaking column
            "energy_kwh": [10, 20, 30, 40, 233.9],
        })

        result = clean_data(df)

        # Check leaking column removed
        assert LEAKING_COLUMN not in result.columns
        assert "energy_kwh" in result.columns
        assert result.loc[4, "energy_kwh"] == pytest.approx(23.39)

        # Check sentinel values were handled (replaced with NaN then imputed)
        # After full pipeline, no NaN should remain
        assert result[SENTINEL_COLUMN].isna().sum() == 0
        # Verify sentinel value is gone (not in data)
        assert SENTINEL_VALUE not in result[SENTINEL_COLUMN].values

        # Check stuck sensor handled (replaced with NaN then imputed)
        stuck_mask = (pd.to_datetime(result["date"]) >= STUCK_START_DATE) & \
                     (pd.to_datetime(result["date"]) <= STUCK_END_DATE)
        # After imputation, no NaN should remain
        assert result.loc[stuck_mask, STUCK_SENSOR_COLUMN].isna().sum() == 0

        # Check roof_type standardized
        assert set(result[ROOF_TYPE_COLUMN].unique()).issubset({"tin", "flat", "sloped"})

        # Check no missing values remain
        assert result.isna().sum().sum() == 0

    def test_clean_data_preserves_record_id_and_date(self):
        df = pd.DataFrame({
            "record_id": ["R1", "R2"],
            "date": ["2026-01-01", "2026-01-02"],
            "panel_area_m2": [20, 30],
            "sunshine_hours": [5, 6],
            "temp_c": [25, 26],
            "dust_days": [1, 2],
            "panel_age_yr": [5, 10],
            "cloud_cover_pct": [10, 20],
            "roof_type": ["tin", "flat"],
            "area": ["Old City", "New Colony"],
            "meter_reading_kwh": [100, 200],
            "energy_kwh": [15, 25],
        })

        result = clean_data(df)
        assert "record_id" in result.columns
        assert "date" in result.columns
        assert list(result["record_id"]) == ["R1", "R2"]


class TestConstants:
    """Tests that constants match findings.yaml."""

    def test_sentinel_value_constant(self):
        assert SENTINEL_VALUE == 9999

    def test_sentinel_column_constant(self):
        assert SENTINEL_COLUMN == "dust_days"

    def test_stuck_sensor_dates(self):
        assert STUCK_START_DATE == "2026-05-02"
        assert STUCK_END_DATE == "2026-05-18"

    def test_leaking_column(self):
        assert LEAKING_COLUMN == "meter_reading_kwh"

    def test_roof_type_mapping_has_three_levels(self):
        unique_levels = set(ROOF_TYPE_MAPPING.values())
        assert len(unique_levels) == 3
        assert unique_levels == {"tin", "flat", "sloped"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
