"""
Edge case and failure mode test suite.
Covers: empty datasets, 1 customer, very small datasets, all-null IDs,
zero/negative prices, extreme outliers, insufficient cluster samples.
"""

import numpy as np
import pandas as pd

from src.evaluation import evaluate_clustering
from src.feature_engineering import RFMFeatureExtractor
from src.preprocessor import DataCleaner


def test_empty_transaction_dataset():
    cleaner = DataCleaner()
    df_empty = pd.DataFrame(
        columns=[
            "InvoiceNo",
            "StockCode",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
        ]
    )
    valid_df, report, _canc_df = cleaner.clean(df_empty)
    assert len(valid_df) == 0
    assert report.total_raw_rows == 0

    extractor = RFMFeatureExtractor()
    rfm = extractor.extract_from_transactions(valid_df)
    assert len(rfm) == 0


def test_single_customer_dataset():
    df_single = pd.DataFrame(
        [
            {
                "CustomerID": "C001",
                "InvoiceNo": "INV1",
                "InvoiceDate": pd.Timestamp("2025-01-01"),
                "Quantity": 2,
                "UnitPrice": 10.0,
                "TotalPrice": 20.0,
                "StockCode": "P1",
            }
        ]
    )
    extractor = RFMFeatureExtractor()
    rfm = extractor.extract_from_transactions(df_single)
    assert len(rfm) == 1

    # Clustering evaluation on 1 sample should safely report invalidity
    X = np.array([[1.0, 1.0, 20.0]])
    labels = np.array([0])
    metrics = evaluate_clustering(X, labels)
    assert metrics.metrics_valid is False
    assert "Insufficient distinct clusters" in metrics.validity_notes


def test_all_missing_customer_ids():
    cleaner = DataCleaner()
    df_missing = pd.DataFrame(
        [
            {
                "InvoiceNo": "INV1",
                "StockCode": "P1",
                "Quantity": 2,
                "InvoiceDate": "2025-01-01",
                "UnitPrice": 10.0,
                "CustomerID": None,
            },
            {
                "InvoiceNo": "INV2",
                "StockCode": "P2",
                "Quantity": 1,
                "InvoiceDate": "2025-01-02",
                "UnitPrice": 5.0,
                "CustomerID": np.nan,
            },
        ]
    )
    valid_df, report, _ = cleaner.clean(df_missing)
    assert len(valid_df) == 0
    assert report.missing_customer_ids == 2


def test_invalid_numerical_values():
    cleaner = DataCleaner()
    df_invalid = pd.DataFrame(
        [
            {
                "InvoiceNo": "INV1",
                "StockCode": "P1",
                "Quantity": -5,
                "InvoiceDate": "2025-01-01",
                "UnitPrice": 10.0,
                "CustomerID": 101,
            },
            {
                "InvoiceNo": "INV2",
                "StockCode": "P2",
                "Quantity": 2,
                "InvoiceDate": "2025-01-02",
                "UnitPrice": -20.0,
                "CustomerID": 102,
            },
            {
                "InvoiceNo": "INV3",
                "StockCode": "P3",
                "Quantity": 0,
                "InvoiceDate": "2025-01-03",
                "UnitPrice": 0.0,
                "CustomerID": 103,
            },
        ]
    )
    valid_df, _report, _ = cleaner.clean(df_invalid)
    assert len(valid_df) == 0  # All rows invalid or cancelled


def test_extreme_outliers_flagging():
    df_rfm = pd.DataFrame(
        [
            {"customer_id": f"C{i}", "recency": 10, "frequency": 2, "monetary": 50.0}
            for i in range(50)
        ]
    )
    # Add one massive wholesale spender
    df_rfm = pd.concat(
        [
            df_rfm,
            pd.DataFrame(
                [
                    {
                        "customer_id": "C_WHALE",
                        "recency": 5,
                        "frequency": 500,
                        "monetary": 150000.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    flagged = RFMFeatureExtractor.flag_outliers(df_rfm, iqr_multiplier=3.0)
    assert "is_extreme_outlier" in flagged.columns
    whale_row = flagged[flagged["customer_id"] == "C_WHALE"].iloc[0]
    assert whale_row["is_extreme_outlier"] == True
