"""
Parity validation tests ensuring Python and JavaScript calculations produce identical results.
Validates:
- Positive transaction filtering
- Cancellation exclusion
- Monetary spend formula (Quantity * UnitPrice)
- Invoice-level Frequency counting
- Recency elapsed day calculation
- Average Order Value (AOV) calculation
"""

from datetime import datetime

import pandas as pd

from src.feature_engineering import RFMFeatureExtractor
from src.preprocessor import DataCleaner


def simulate_js_client_aggregation(raw_rows, ref_date_str="2025-10-15T00:00:00"):
    """Replicates exact JavaScript calculation logic from dashboard/app.js parseCsv."""
    ref_date = datetime.fromisoformat(ref_date_str)
    cust_map = {}

    for r in raw_rows:
        cid = r.get("CustomerID") or r.get("customer_id")
        if cid is None or str(cid).strip() == "":
            continue
        inv_no = str(r.get("InvoiceNo", "")).strip()
        if inv_no.upper().startswith("C"):
            continue

        try:
            qty = float(r.get("Quantity", 0))
            price = float(r.get("UnitPrice", 0))
        except (ValueError, TypeError):
            continue

        if qty <= 0 or price <= 0:
            continue

        spend = qty * price
        inv_date = datetime.fromisoformat(str(r.get("InvoiceDate")))

        if cid not in cust_map:
            cust_map[cid] = {
                "customer_id": str(cid),
                "latest_date": inv_date,
                "invoices": set(),
                "monetary": 0.0,
            }

        cust_map[cid]["invoices"].add(inv_no)
        cust_map[cid]["monetary"] += spend
        cust_map[cid]["latest_date"] = max(cust_map[cid]["latest_date"], inv_date)

    results = {}
    for cid, data in cust_map.items():
        diff_days = max(
            0, int((ref_date - data["latest_date"]).total_seconds() // (24 * 3600))
        )
        freq = max(1, len(data["invoices"]))
        mon = round(data["monetary"], 2)
        aov = round(mon / freq, 2)
        results[cid] = {
            "customer_id": cid,
            "recency": diff_days,
            "frequency": freq,
            "monetary": mon,
            "avg_order_value": aov,
        }
    return results


def test_python_js_rfm_parity():
    # Test transactions
    test_rows = [
        {
            "InvoiceNo": "1001",
            "StockCode": "A",
            "Quantity": 3,
            "UnitPrice": 12.50,
            "CustomerID": 501,
            "InvoiceDate": "2025-10-10 10:00:00",
        },
        {
            "InvoiceNo": "1001",
            "StockCode": "B",
            "Quantity": 1,
            "UnitPrice": 5.00,
            "CustomerID": 501,
            "InvoiceDate": "2025-10-10 10:00:00",
        },
        {
            "InvoiceNo": "1002",
            "StockCode": "C",
            "Quantity": 2,
            "UnitPrice": 20.00,
            "CustomerID": 501,
            "InvoiceDate": "2025-10-12 14:00:00",
        },
        {
            "InvoiceNo": "C1003",
            "StockCode": "C",
            "Quantity": -2,
            "UnitPrice": 20.00,
            "CustomerID": 501,
            "InvoiceDate": "2025-10-13 10:00:00",
        },
        # Customer 502
        {
            "InvoiceNo": "1004",
            "StockCode": "D",
            "Quantity": 10,
            "UnitPrice": 3.00,
            "CustomerID": 502,
            "InvoiceDate": "2025-09-15 09:00:00",
        },
    ]

    ref_date = pd.Timestamp("2025-10-15 00:00:00")

    # 1. Run Python Cleaner + Extractor
    cleaner = DataCleaner()
    df_raw = pd.DataFrame(test_rows)
    valid_df, _, canc_df = cleaner.clean(df_raw)

    extractor = RFMFeatureExtractor(log_transform=False)
    py_rfm = extractor.extract_from_transactions(
        valid_df, canc_df, reference_date=ref_date
    )
    py_dict = py_rfm.set_index("customer_id").to_dict(orient="index")

    # 2. Run Simulated JS Engine
    js_dict = simulate_js_client_aggregation(
        test_rows, ref_date_str="2025-10-15T00:00:00"
    )

    # 3. Assert exact mathematical equivalence
    for cid_int in [501, 502]:
        cid_str = str(cid_int)
        assert cid_str in py_dict, f"Customer {cid_str} missing in Python output"
        assert cid_int in js_dict, f"Customer {cid_int} missing in JS output"

        py_cust = py_dict[cid_str]
        js_cust = js_dict[cid_int]

        # Check Frequency
        assert py_cust["frequency"] == js_cust["frequency"], (
            f"Frequency mismatch for {cid_str}"
        )
        # Check Monetary spend
        assert abs(py_cust["monetary"] - js_cust["monetary"]) < 0.01, (
            f"Monetary mismatch for {cid_str}"
        )
        # Check AOV
        assert abs(py_cust["avg_order_value"] - js_cust["avg_order_value"]) < 0.01, (
            f"AOV mismatch for {cid_str}"
        )
        # Check Recency
        assert py_cust["recency"] == js_cust["recency"], (
            f"Recency mismatch for {cid_str}"
        )
