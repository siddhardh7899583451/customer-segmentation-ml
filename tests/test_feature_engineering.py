"""
Unit tests for feature engineering and RFM extraction.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import RFMFeatureExtractor


@pytest.fixture
def clean_transactions():
    return pd.DataFrame(
        [
            {
                "CustomerID": "C1",
                "InvoiceNo": "INV1",
                "InvoiceDate": pd.Timestamp("2025-10-01"),
                "Quantity": 2,
                "UnitPrice": 10.0,
                "TotalPrice": 20.0,
                "StockCode": "P1",
            },
            {
                "CustomerID": "C1",
                "InvoiceNo": "INV2",
                "InvoiceDate": pd.Timestamp("2025-10-05"),
                "Quantity": 1,
                "UnitPrice": 30.0,
                "TotalPrice": 30.0,
                "StockCode": "P2",
            },
            {
                "CustomerID": "C2",
                "InvoiceNo": "INV3",
                "InvoiceDate": pd.Timestamp("2025-09-01"),
                "Quantity": 4,
                "UnitPrice": 25.0,
                "TotalPrice": 100.0,
                "StockCode": "P1",
            },
        ]
    )


def test_extract_rfm_from_transactions(clean_transactions):
    ref_date = pd.Timestamp("2025-10-10")
    extractor = RFMFeatureExtractor(log_transform=False)
    rfm = extractor.extract_from_transactions(
        clean_transactions, reference_date=ref_date
    )

    assert len(rfm) == 2
    c1 = rfm[rfm["customer_id"] == "C1"].iloc[0]
    assert c1["frequency"] == 2
    assert c1["monetary"] == 50.0
    assert c1["recency"] == 5  # 2025-10-10 - 2025-10-05 = 5 days
    assert c1["avg_order_value"] == 25.0


def test_process_customer_rfm_direct():
    df_in = pd.DataFrame(
        [
            {"customer_id": "101", "recency": 10, "frequency": 4, "monetary": 400.0},
            {"customer_id": "102", "recency": 50, "frequency": 1, "monetary": 50.0},
        ]
    )
    extractor = RFMFeatureExtractor()
    out = extractor.process_customer_rfm(df_in)
    assert len(out) == 2
    assert "avg_order_value" in out.columns
    assert out.iloc[0]["avg_order_value"] == 100.0


def test_prepare_features_for_clustering(clean_transactions):
    extractor = RFMFeatureExtractor(log_transform=True, scaler_type="standard")
    rfm = extractor.extract_from_transactions(clean_transactions)
    X_scaled, features = extractor.prepare_features_for_clustering(rfm)

    assert X_scaled.shape == (2, 3)
    assert np.isfinite(X_scaled).all()
    assert features == ["recency", "frequency", "monetary"]
