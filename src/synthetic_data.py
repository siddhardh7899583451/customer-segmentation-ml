"""
Synthetic Transaction & RFM Customer Generator for Testing and Demo Environments.

Explicitly labels all generated records with metadata indicating DEMO / SYNTHETIC data.
Reflects realistic behavioral patterns (VIPs, Regulars, Bargain, Churned, New)
plus realistic anomalies (cancellations, guest checkouts, duplicates) for pipeline verification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("synthetic_data")

PRODUCT_CATALOG = [
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
    ("85099B", "JUMBO BAG RED RETROSPOT", 1.95),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("47566", "PARTY BUNTING", 4.95),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
    ("22197", "POPCORN HOLDER", 0.85),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
    ("22666", "RECIPE BOX PANTRY DESIGN", 7.95),
    ("22960", "JAM MAKING SET WITH JARS", 4.25),
    ("21931", "JUMBO STORAGE BAG SUKI", 2.08),
    ("84997D", "CHILDRENS CUTLERY POLKADOT PINK", 4.15),
    ("POST", "POSTAGE", 18.00),
]


def generate_synthetic_transactions(
    n_customers: int = 500,
    target_records: int = 5000,
    random_state: int = 42,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Generate transaction-level synthetic records matching the UCI Online Retail schema.

    Contains realistic customer archetypes:
    - High-Value VIPs: High frequency, large monetary spend, high AOV.
    - Loyal Regulars: Steady repeat purchases across the calendar year.
    - At-Risk / Churned: Transactions concentrated in early dates, none recent.
    - Low-Engagement / New: 1 or 2 small transactions recently.
    - Intentional data quirks for preprocessing testing:
      * Cancelled orders (InvoiceNo prefixed with 'C' and negative Quantity).
      * Guest checkouts (Null CustomerID).
      * Occasional duplicate transactions.
    """
    logger.info(
        f"Generating synthetic transactions for {n_customers} customer archetypes (seed={random_state})..."
    )
    rng = np.random.default_rng(random_state)

    # Customer assignment by archetype
    # 1. VIPs (10%)
    # 2. Loyal Regulars (30%)
    # 3. At-Risk / Churned (25%)
    # 4. Low-Engagement / Starters (35%)
    n_vip = int(n_customers * 0.10)
    n_loyal = int(n_customers * 0.30)
    n_at_risk = int(n_customers * 0.25)

    customer_ids = [10000 + i for i in range(n_customers)]
    vip_ids = customer_ids[:n_vip]
    loyal_ids = customer_ids[n_vip : n_vip + n_loyal]
    at_risk_ids = customer_ids[n_vip + n_loyal : n_vip + n_loyal + n_at_risk]
    low_ids = customer_ids[n_vip + n_loyal + n_at_risk :]

    base_date = pd.Timestamp("2025-12-09 12:00:00")
    records = []
    invoice_counter = 500000

    def create_invoice(
        cust_id: int | None,
        date_val: pd.Timestamp,
        n_items: int,
        is_cancel: bool = False,
    ):
        nonlocal invoice_counter
        invoice_counter += 1
        prefix = "C" if is_cancel else ""
        inv_no = f"{prefix}{invoice_counter}"

        # Pick items from catalog
        chosen_indices = rng.choice(len(PRODUCT_CATALOG), size=n_items, replace=True)
        for idx in chosen_indices:
            stock, desc, base_price = PRODUCT_CATALOG[idx]
            price = round(float(base_price * rng.uniform(0.9, 1.15)), 2)
            qty = (
                int(rng.integers(1, 12)) if not is_cancel else -int(rng.integers(1, 4))
            )
            records.append(
                {
                    "InvoiceNo": inv_no,
                    "StockCode": stock,
                    "Description": desc,
                    "Quantity": qty,
                    "InvoiceDate": date_val.strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice": price,
                    "CustomerID": cust_id,
                    "Country": "United Kingdom",
                    "DataClassification": "SYNTHETIC_DEMO",
                }
            )

    # 1. VIP transactions
    for cid in vip_ids:
        num_invoices = rng.integers(12, 28)
        for _ in range(num_invoices):
            # Recency: 1 to 45 days ago
            days_ago = rng.integers(1, 46)
            d = base_date - pd.Timedelta(
                days=int(days_ago), hours=int(rng.integers(0, 23))
            )
            items_count = rng.integers(3, 8)
            create_invoice(cid, d, items_count, is_cancel=False)
            if rng.random() < 0.04:  # Rare cancellation
                cancel_date = d + pd.Timedelta(days=1)
                create_invoice(cid, cancel_date, 1, is_cancel=True)

    # 2. Loyal Regulars
    for cid in loyal_ids:
        num_invoices = rng.integers(5, 14)
        for _ in range(num_invoices):
            days_ago = rng.integers(10, 120)
            d = base_date - pd.Timedelta(
                days=int(days_ago), hours=int(rng.integers(0, 23))
            )
            items_count = rng.integers(2, 5)
            create_invoice(cid, d, items_count, is_cancel=False)

    # 3. At-Risk / Churned
    for cid in at_risk_ids:
        num_invoices = rng.integers(2, 6)
        for _ in range(num_invoices):
            # Recency: 160 to 365 days ago
            days_ago = rng.integers(160, 365)
            d = base_date - pd.Timedelta(
                days=int(days_ago), hours=int(rng.integers(0, 23))
            )
            items_count = rng.integers(1, 4)
            create_invoice(cid, d, items_count, is_cancel=False)

    # 4. Low Engagement / Starters
    for cid in low_ids:
        num_invoices = rng.integers(1, 3)
        for _ in range(num_invoices):
            days_ago = rng.integers(5, 90)
            d = base_date - pd.Timedelta(
                days=int(days_ago), hours=int(rng.integers(0, 23))
            )
            items_count = rng.integers(1, 3)
            create_invoice(cid, d, items_count, is_cancel=False)

    # 5. Guest checkouts (Null CustomerID) for preprocessing robustness check
    n_guests = 30
    for _ in range(n_guests):
        days_ago = rng.integers(1, 200)
        d = base_date - pd.Timedelta(days=int(days_ago))
        create_invoice(None, d, 2, is_cancel=False)

    df = pd.DataFrame(records)

    # Inject a small handful of duplicate rows (e.g. 10 duplicates)
    dup_sample = df.sample(n=10, random_state=random_state)
    df = pd.concat([df, dup_sample], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info(f"Saved {len(df)} synthetic transactions to {out_p}")

    return df


def generate_synthetic_rfm_customers(
    n_samples: int = 500,
    random_state: int = 42,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Generate customer-level RFM metrics directly for dashboard demos and offline testing.

    Explicitly labeled as DEMO / SYNTHETIC.
    """
    rng = np.random.default_rng(random_state)

    n_vip = int(n_samples * 0.15)
    n_loyal = int(n_samples * 0.30)
    n_at_risk = int(n_samples * 0.25)
    n_new = int(n_samples * 0.15)
    n_low = n_samples - (n_vip + n_loyal + n_at_risk + n_new)

    records = []
    cid = 1000

    # 1. VIPs
    for _ in range(n_vip):
        cid += 1
        rec = rng.integers(1, 28)
        freq = rng.integers(12, 32)
        mon = round(float(rng.uniform(2800, 9500)), 2)
        aov = round(float(mon / freq), 2)
        records.append(
            {
                "customer_id": f"DEMO_CUST_{cid}",
                "recency": int(rec),
                "frequency": int(freq),
                "monetary": mon,
                "avg_order_value": aov,
                "return_rate": round(float(rng.uniform(0.01, 0.05)), 3),
                "unique_items": int(rng.integers(25, 70)),
                "true_segment_archetype": "High-Value Champions",
                "is_synthetic": True,
            }
        )

    # 2. Loyal Regulars
    for _ in range(n_loyal):
        cid += 1
        rec = rng.integers(15, 65)
        freq = rng.integers(5, 14)
        mon = round(float(rng.uniform(900, 2600)), 2)
        aov = round(float(mon / freq), 2)
        records.append(
            {
                "customer_id": f"DEMO_CUST_{cid}",
                "recency": int(rec),
                "frequency": int(freq),
                "monetary": mon,
                "avg_order_value": aov,
                "return_rate": round(float(rng.uniform(0.02, 0.08)), 3),
                "unique_items": int(rng.integers(12, 35)),
                "true_segment_archetype": "Loyal Customers",
                "is_synthetic": True,
            }
        )

    # 3. At-Risk Customers
    for _ in range(n_at_risk):
        cid += 1
        rec = rng.integers(140, 360)
        freq = rng.integers(2, 7)
        mon = round(float(rng.uniform(300, 1400)), 2)
        aov = round(float(mon / freq), 2)
        records.append(
            {
                "customer_id": f"DEMO_CUST_{cid}",
                "recency": int(rec),
                "frequency": int(freq),
                "monetary": mon,
                "avg_order_value": aov,
                "return_rate": round(float(rng.uniform(0.04, 0.12)), 3),
                "unique_items": int(rng.integers(4, 18)),
                "true_segment_archetype": "At-Risk Customers",
                "is_synthetic": True,
            }
        )

    # 4. Recent Starters / Promising
    for _ in range(n_new):
        cid += 1
        rec = rng.integers(2, 35)
        freq = rng.integers(1, 3)
        mon = round(float(rng.uniform(100, 450)), 2)
        aov = round(float(mon / freq), 2)
        records.append(
            {
                "customer_id": f"DEMO_CUST_{cid}",
                "recency": int(rec),
                "frequency": int(freq),
                "monetary": mon,
                "avg_order_value": aov,
                "return_rate": round(float(rng.uniform(0.01, 0.05)), 3),
                "unique_items": int(rng.integers(2, 8)),
                "true_segment_archetype": "Recent Starters",
                "is_synthetic": True,
            }
        )

    # 5. Low-Engagement / Inactive
    for _ in range(n_low):
        cid += 1
        rec = rng.integers(180, 365)
        freq = 1
        mon = round(float(rng.uniform(20, 150)), 2)
        aov = mon
        records.append(
            {
                "customer_id": f"DEMO_CUST_{cid}",
                "recency": int(rec),
                "frequency": int(freq),
                "monetary": mon,
                "avg_order_value": aov,
                "return_rate": round(float(rng.uniform(0.0, 0.10)), 3),
                "unique_items": 1,
                "true_segment_archetype": "Low-Engagement Customers",
                "is_synthetic": True,
            }
        )

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info(f"Saved {len(df)} synthetic RFM customer records to {out_p}")

    return df
