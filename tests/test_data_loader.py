"""
Unit tests for data_loader and synthetic_data modules.
"""

import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.synthetic_data import (
    generate_synthetic_rfm_customers,
    generate_synthetic_transactions,
)


def test_generate_synthetic_transactions():
    df = generate_synthetic_transactions(n_customers=50, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100
    expected_cols = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
    ]
    for col in expected_cols:
        assert col in df.columns
    # Check that synthetic demo label is present
    assert "DataClassification" in df.columns
    assert (df["DataClassification"] == "SYNTHETIC_DEMO").all()


def test_generate_synthetic_rfm_customers():
    df = generate_synthetic_rfm_customers(n_samples=40, random_state=42)
    assert len(df) == 40
    assert "recency" in df.columns
    assert "frequency" in df.columns
    assert "monetary" in df.columns
    assert (df["is_synthetic"] == True).all()


def test_load_dataset_fallback():
    df, is_synth = load_dataset(
        file_path="non_existent_file.csv", use_demo_fallback=True
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert is_synth is True


def test_load_dataset_no_fallback_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset(
            file_path="completely_invalid_path_xyz.csv", use_demo_fallback=False
        )
