"""
Unit tests for data cleaning and preprocessing module.
"""

import pandas as pd
import pytest

from src.preprocessor import DataCleaner


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        [
            # Valid transaction
            {
                "InvoiceNo": "1001",
                "StockCode": "A1",
                "Quantity": 2,
                "InvoiceDate": "2025-01-01 10:00:00",
                "UnitPrice": 10.0,
                "CustomerID": 100,
            },
            # Missing customer ID
            {
                "InvoiceNo": "1002",
                "StockCode": "A2",
                "Quantity": 1,
                "InvoiceDate": "2025-01-02 11:00:00",
                "UnitPrice": 5.0,
                "CustomerID": None,
            },
            # Duplicate row of 1001
            {
                "InvoiceNo": "1001",
                "StockCode": "A1",
                "Quantity": 2,
                "InvoiceDate": "2025-01-01 10:00:00",
                "UnitPrice": 10.0,
                "CustomerID": 100,
            },
            # Cancelled invoice
            {
                "InvoiceNo": "C1003",
                "StockCode": "A1",
                "Quantity": -2,
                "InvoiceDate": "2025-01-03 12:00:00",
                "UnitPrice": 10.0,
                "CustomerID": 100,
            },
            # Negative quantity without C
            {
                "InvoiceNo": "1004",
                "StockCode": "A3",
                "Quantity": -1,
                "InvoiceDate": "2025-01-04 10:00:00",
                "UnitPrice": 4.0,
                "CustomerID": 101,
            },
            # Zero price
            {
                "InvoiceNo": "1005",
                "StockCode": "A4",
                "Quantity": 1,
                "InvoiceDate": "2025-01-05 10:00:00",
                "UnitPrice": 0.0,
                "CustomerID": 102,
            },
            # Another valid row for customer 101
            {
                "InvoiceNo": "1006",
                "StockCode": "A5",
                "Quantity": 3,
                "InvoiceDate": "2025-01-06 14:00:00",
                "UnitPrice": 15.0,
                "CustomerID": 101,
            },
        ]
    )


def test_cleaner_removes_missing_customer(sample_raw_df):
    cleaner = DataCleaner()
    valid_df, report, _canc_df = cleaner.clean(sample_raw_df)
    assert report.missing_customer_ids == 1
    assert not valid_df["CustomerID"].isna().any()


def test_cleaner_removes_duplicates(sample_raw_df):
    cleaner = DataCleaner()
    _valid_df, report, _canc_df = cleaner.clean(sample_raw_df)
    assert report.duplicate_rows_removed == 1


def test_cleaner_segregates_cancellations(sample_raw_df):
    cleaner = DataCleaner()
    valid_df, report, canc_df = cleaner.clean(sample_raw_df)
    assert report.cancelled_transactions == 2
    assert len(canc_df) == 2
    assert (valid_df["Quantity"] > 0).all()


def test_cleaner_filters_invalid_prices(sample_raw_df):
    cleaner = DataCleaner()
    valid_df, report, _canc_df = cleaner.clean(sample_raw_df)
    assert report.zero_or_negative_price_rows == 1
    assert (valid_df["UnitPrice"] > 0).all()


def test_cleaner_calculates_total_price(sample_raw_df):
    cleaner = DataCleaner()
    valid_df, _report, _canc_df = cleaner.clean(sample_raw_df)
    assert "TotalPrice" in valid_df.columns
    # Check row 1001: 2 * 10.0 = 20.0
    row_1001 = valid_df[valid_df["InvoiceNo"] == "1001"].iloc[0]
    assert row_1001["TotalPrice"] == 20.0
