"""
Utility helpers: structured logging, safe CSV export (CSV formula injection defense),
JSON serialization for numpy/dataclass objects, and path safety.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_logger(
    name: str = "customer_segmentation", log_file: Path | None = None
) -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def sanitize_csv_cell(value: Any) -> Any:
    """Mitigate CSV Formula Injection (DDE injection) for text values.

    If a text field begins with '=', '+', '-', '@', '	', or '' (even after leading whitespace),
    prepend a single quote so spreadsheet software (Excel, Calc) treats it as a text literal.
    """
    if isinstance(value, str) and value:
        stripped = value.lstrip(" 	
")
        if stripped and stripped[0] in ("=", "+", "-", "@", "	", ""):
            return f"'{value}"
    return value


def safe_export_csv(
    df: pd.DataFrame, target_path: str | Path, index: bool = False
) -> Path:
    """Sanitizes untrusted text columns and exports dataframe safely to CSV.

    Numeric columns (integers, floats) are preserved intact to avoid altering genuine numbers
    such as negative financial values or decimals.
    """
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sanitized_df = df.copy()
    for col in sanitized_df.columns:
        if pd.api.types.is_numeric_dtype(sanitized_df[col]):
            continue
        sanitized_df[col] = sanitized_df[col].apply(sanitize_csv_cell)

    sanitized_df.to_csv(path, index=index, encoding="utf-8")
    return path


class CustomJSONEncoder(json.JSONEncoder):
    """Encodes numpy data types and dataclasses into standard JSON."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def save_json_report(data: dict[str, Any], target_path: str | Path) -> Path:
    """Save structured dictionary to JSON file."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=CustomJSONEncoder)
    return path
