"""
Customer Feature Engineering & RFM Calculation Module.

Aggregates transaction-level records into customer-level behavioral profiles:
- Recency: Days elapsed since the customer's last purchase.
- Frequency: Count of distinct purchase transactions.
- Monetary: Total sum spent by the customer.
- Average Order Value (AOV): Monetary / Frequency.
- Unique Items: Count of distinct product codes purchased.
- Return Rate: Proportion of cancelled items relative to total items.
- Log transforms and feature standardization for unsupervised algorithms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils import get_logger

logger = get_logger("feature_engineering")


class RFMFeatureExtractor:
    """Calculates customer-level RFM and behavioral features."""

    def __init__(self, log_transform: bool = True, scaler_type: str = "standard"):
        self.log_transform = log_transform
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.fitted = False
        self.feature_names: list[str] = ["recency", "frequency", "monetary"]

    def extract_from_transactions(
        self,
        df_valid: pd.DataFrame,
        df_cancellations: pd.DataFrame | None = None,
        reference_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Aggregate transaction rows into customer-level RFM and behavioral metrics."""
        logger.info(
            f"Extracting customer features from {len(df_valid)} transaction rows..."
        )

        if df_valid.empty:
            logger.warning("Empty transaction dataframe passed to feature extractor.")
            return pd.DataFrame(
                columns=[
                    "customer_id",
                    "recency",
                    "frequency",
                    "monetary",
                    "avg_order_value",
                    "unique_items",
                    "return_rate",
                ]
            )

        # Determine reference date (1 day after maximum invoice date)
        if reference_date is None:
            reference_date = df_valid["InvoiceDate"].max() + pd.Timedelta(days=1)

        # Core RFM aggregation
        grouped = df_valid.groupby("CustomerID")

        rfm = grouped.agg(
            latest_date=("InvoiceDate", "max"),
            frequency=("InvoiceNo", "nunique"),
            monetary=("TotalPrice", "sum"),
            unique_items=("StockCode", "nunique"),
            total_quantity=("Quantity", "sum"),
        ).reset_index()

        # Recency in full days
        rfm["recency"] = (reference_date - rfm["latest_date"]).dt.days
        rfm["recency"] = rfm["recency"].clip(lower=0)

        # Average Order Value (AOV)
        rfm["avg_order_value"] = (rfm["monetary"] / rfm["frequency"]).round(2)
        rfm["monetary"] = rfm["monetary"].round(2)

        # Calculate return rate using cancellations if available
        rfm["return_rate"] = 0.0
        if (
            df_cancellations is not None
            and not df_cancellations.empty
            and "CustomerID" in df_cancellations.columns
        ):
            canc_grouped = (
                df_cancellations.groupby("CustomerID")["Quantity"].sum().abs()
            )
            return_dict = canc_grouped.to_dict()

            def calc_return_rate(row):
                cid = row["CustomerID"]
                canc_qty = return_dict.get(cid, 0)
                tot = row["total_quantity"] + canc_qty
                return round(canc_qty / tot, 4) if tot > 0 else 0.0

            rfm["return_rate"] = rfm.apply(calc_return_rate, axis=1)

        rfm = rfm.rename(columns={"CustomerID": "customer_id"})
        cols_to_keep = [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "avg_order_value",
            "unique_items",
            "return_rate",
        ]
        res_df = rfm[cols_to_keep].copy()

        # Outlier flag via IQR
        res_df = self.flag_outliers(res_df)

        logger.info(
            f"Successfully extracted RFM profiles for {len(res_df)} unique customers."
        )
        return res_df

    def process_customer_rfm(self, df_rfm: pd.DataFrame) -> pd.DataFrame:
        """Processes an already customer-level RFM dataset (e.g. from user CSV or synthetic demo)."""
        df = df_rfm.copy()
        col_map = {c: c.lower().strip() for c in df.columns}
        df = df.rename(columns=col_map)

        # Standardize ID column
        if "customer_id" not in df.columns:
            for id_cand in ["customerid", "id", "customer"]:
                if id_cand in df.columns:
                    df = df.rename(columns={id_cand: "customer_id"})
                    break

        # Check required columns
        for col in ["recency", "frequency", "monetary"]:
            if col not in df.columns:
                raise ValueError(
                    f"Input customer dataset missing mandatory column: '{col}'"
                )

        df["recency"] = (
            pd.to_numeric(df["recency"], errors="coerce").fillna(0).clip(lower=0)
        )
        df["frequency"] = (
            pd.to_numeric(df["frequency"], errors="coerce").fillna(1).clip(lower=1)
        )
        df["monetary"] = pd.to_numeric(df["monetary"], errors="coerce").fillna(0.0)

        if "avg_order_value" not in df.columns:
            df["avg_order_value"] = (df["monetary"] / df["frequency"]).round(2)
        if "unique_items" not in df.columns:
            df["unique_items"] = df["frequency"]
        if "return_rate" not in df.columns:
            df["return_rate"] = 0.0

        return self.flag_outliers(df)

    @staticmethod
    def flag_outliers(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame:
        """Flags extreme outliers in monetary and frequency distributions."""
        out_df = df.copy()
        is_outlier = np.zeros(len(out_df), dtype=bool)

        for col in ["monetary", "frequency"]:
            if col in out_df.columns and len(out_df) > 4:
                q1 = out_df[col].quantile(0.25)
                q3 = out_df[col].quantile(0.75)
                iqr = q3 - q1
                threshold = q3 + iqr_multiplier * iqr
                is_outlier = is_outlier | (out_df[col] > threshold)

        out_df["is_extreme_outlier"] = is_outlier
        return out_df

    def prepare_features_for_clustering(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Transform numerical customer features using log transforms (if enabled) and scaling.

        Returns:
            X_scaled: Normalized 2D numpy array ready for distance-based clustering algorithms.
            feature_names: List of column names used in transformation.
        """
        feats = feature_cols or self.feature_names
        X = df[feats].copy().values

        if self.log_transform:
            # Apply log1p to compress right-skewed positive distributions
            X_log = np.zeros_like(X, dtype=float)
            for j in range(X.shape[1]):
                col_vals = X[:, j]
                # Shift minimum to >= 0 if any negatives exist
                min_v = np.min(col_vals)
                if min_v < 0:
                    col_vals = col_vals - min_v
                X_log[:, j] = np.log1p(col_vals)
            X = X_log

        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        return X_scaled, feats
