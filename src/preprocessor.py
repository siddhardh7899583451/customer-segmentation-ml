"""
Transaction Preprocessing and Data Quality Assurance Module.

Implements strict validation, deduplication, missing CustomerID filtering,
cancellation segregation, price/quantity checks, and data quality reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import PreprocessingConfig
from src.utils import get_logger

logger = get_logger("preprocessor")


@dataclass
class DataQualityReport:
    """Detailed metadata tracking row counts and anomalies filtered during cleaning."""

    total_raw_rows: int = 0
    missing_customer_ids: int = 0
    duplicate_rows_removed: int = 0
    cancelled_transactions: int = 0
    zero_or_negative_price_rows: int = 0
    zero_or_negative_quantity_rows: int = 0
    final_valid_rows: int = 0
    unique_customers: int = 0
    earliest_date: str | None = None
    latest_date: str | None = None
    country_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_raw_rows": self.total_raw_rows,
            "missing_customer_ids": self.missing_customer_ids,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "cancelled_transactions": self.cancelled_transactions,
            "zero_or_negative_price_rows": self.zero_or_negative_price_rows,
            "zero_or_negative_quantity_rows": self.zero_or_negative_quantity_rows,
            "final_valid_rows": self.final_valid_rows,
            "unique_customers": self.unique_customers,
            "earliest_date": self.earliest_date,
            "latest_date": self.latest_date,
            "country_count": self.country_count,
            "retention_rate_pct": round(
                (self.final_valid_rows / self.total_raw_rows * 100)
                if self.total_raw_rows > 0
                else 0,
                2,
            ),
        }


class DataCleaner:
    """Preprocesses raw retail transaction logs into clean, verified data frames."""

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def clean(
        self, df_raw: pd.DataFrame
    ) -> tuple[pd.DataFrame, DataQualityReport, pd.DataFrame]:
        """Execute end-to-end transaction cleaning.

        Returns:
            Tuple of:
            - cleaned_df: Valid positive transactions for customer-level segmentation.
            - report: Comprehensive DataQualityReport.
            - cancellations_df: Segregated cancelled transactions (for return rate derivation).
        """
        report = DataQualityReport(total_raw_rows=len(df_raw))
        logger.info(
            f"Beginning cleaning pipeline on {report.total_raw_rows} raw records..."
        )

        df = df_raw.copy()

        # 1. Normalize Column Names
        col_map = {c: c.strip() for c in df.columns}
        df = df.rename(columns=col_map)

        # 2. Check for Missing CustomerID
        if "CustomerID" in df.columns:
            missing_cust_mask = df["CustomerID"].isna() | (df["CustomerID"] == "")
            report.missing_customer_ids = int(missing_cust_mask.sum())
            if self.config.remove_missing_customers:
                df = df[~missing_cust_mask].copy()
                logger.info(
                    f"Removed {report.missing_customer_ids} records without CustomerID."
                )
        elif "customer_id" in df.columns:
            df = df.rename(columns={"customer_id": "CustomerID"})

        # 3. Deduplication
        initial_count = len(df)
        df = df.drop_duplicates()
        report.duplicate_rows_removed = initial_count - len(df)
        if report.duplicate_rows_removed > 0:
            logger.info(f"Removed {report.duplicate_rows_removed} duplicate rows.")

        # 4. Standardize Data Types
        # Standardize CustomerID as integer or string without trailing .0
        if "CustomerID" in df.columns:
            df["CustomerID"] = df["CustomerID"].apply(
                lambda x: (
                    str(int(x))
                    if isinstance(x, (int, float)) and not np.isnan(x)
                    else str(x).strip()
                )
            )

        # Standardize Dates
        if "InvoiceDate" in df.columns:
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
            # Drop invalid dates if any
            df = df[df["InvoiceDate"].notna()].copy()

        # Standardize Numeric Columns
        df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0)
        df["UnitPrice"] = pd.to_numeric(
            df.get("UnitPrice", 0.0), errors="coerce"
        ).fillna(0.0)

        # 5. Segregate Cancellations
        # Invoices starting with 'C' or with negative quantities represent cancellations
        is_cancelled_invoice = (
            df["InvoiceNo"].astype(str).str.upper().str.startswith("C")
        )
        is_negative_qty = df["Quantity"] < 0
        cancellation_mask = is_cancelled_invoice | is_negative_qty

        report.cancelled_transactions = int(cancellation_mask.sum())
        cancellations_df = df[cancellation_mask].copy()

        # Retain only positive purchases for core RFM calculation
        valid_df = df[~cancellation_mask].copy()

        # 6. Validate Unit Price & Quantity
        invalid_price_mask = valid_df["UnitPrice"] < self.config.min_unit_price
        report.zero_or_negative_price_rows = int(invalid_price_mask.sum())
        valid_df = valid_df[~invalid_price_mask].copy()

        invalid_qty_mask = valid_df["Quantity"] < self.config.min_quantity
        report.zero_or_negative_quantity_rows = int(invalid_qty_mask.sum())
        valid_df = valid_df[~invalid_qty_mask].copy()

        # 7. Add Total Spend column
        valid_df["TotalPrice"] = valid_df["Quantity"] * valid_df["UnitPrice"]

        # Populate final report metrics
        report.final_valid_rows = len(valid_df)
        report.unique_customers = (
            int(valid_df["CustomerID"].nunique())
            if "CustomerID" in valid_df.columns
            else 0
        )
        if "InvoiceDate" in valid_df.columns and len(valid_df) > 0:
            report.earliest_date = str(valid_df["InvoiceDate"].min())
            report.latest_date = str(valid_df["InvoiceDate"].max())
        if "Country" in valid_df.columns:
            report.country_count = int(valid_df["Country"].nunique())

        logger.info(
            f"Cleaning complete. Retained {report.final_valid_rows} valid rows "
            f"across {report.unique_customers} unique customers."
        )
        return valid_df, report, cancellations_df
