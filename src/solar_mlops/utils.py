"""Shared utility functions for solar_mlops package."""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


def setup_logging(name: str = "solar_mlops", level: int = logging.INFO) -> logging.Logger:
    """
    Set up standardized logging configuration.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Get list of numeric column names from DataFrame.

    Args:
        df: Input DataFrame
        exclude: Columns to exclude from result

    Returns:
        List of numeric column names
    """
    exclude = exclude or []
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def get_categorical_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Get list of categorical/string column names from DataFrame.

    Args:
        df: Input DataFrame
        exclude: Columns to exclude from result

    Returns:
        List of categorical column names
    """
    exclude = exclude or []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [col for col in cat_cols if col not in exclude]


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists, create if needed.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Load CSV file with standard options.

    Args:
        path: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """
    Save DataFrame to CSV with standard options.

    Args:
        df: DataFrame to save
        path: Output path
        **kwargs: Additional arguments passed to df.to_csv
    """
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False, **kwargs)