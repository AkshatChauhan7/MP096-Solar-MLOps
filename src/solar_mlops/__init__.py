"""solar_mlops package for solar energy prediction MLOps pipeline."""

from .cleaning import (
    load_raw_data,
    detect_leaking_column,
    detect_unit_mixup,
    detect_sentinel_values,
    detect_duplicate_rows,
    detect_stuck_sensor,
    detect_target_typos,
    detect_spelling_variants,
    clean_data,
    save_cleaned_data,
)

from .utils import (
    setup_logging,
    get_numeric_columns,
    get_categorical_columns,
)

__all__ = [
    "load_raw_data",
    "detect_leaking_column",
    "detect_unit_mixup",
    "detect_sentinel_values",
    "detect_duplicate_rows",
    "detect_stuck_sensor",
    "detect_target_typos",
    "detect_spelling_variants",
    "clean_data",
    "save_cleaned_data",
    "setup_logging",
    "get_numeric_columns",
    "get_categorical_columns",
]

__version__ = "0.1.0"