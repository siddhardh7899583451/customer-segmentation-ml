"""
Data Ingestion and Dataset Loading Module.

Supports downloading and extracting the official UCI Online Retail dataset,
loading pre-existing Excel/CSV transaction files, and falling back safely
to a clearly labeled synthetic demo dataset if offline or requested.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, UCI_ONLINE_RETAIL_URL
from src.synthetic_data import generate_synthetic_transactions
from src.utils import get_logger

logger = get_logger("data_loader")


def download_uci_dataset(dest_dir: Path = DATA_DIR) -> Path | None:
    """Download and extract the UCI Online Retail zip archive.

    Returns the path to the extracted Excel file, or None if download fails.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_xlsx = dest_dir / "Online Retail.xlsx"

    if target_xlsx.exists():
        logger.info(f"UCI dataset already exists at {target_xlsx}")
        return target_xlsx

    logger.info(
        f"Attempting to download UCI Online Retail dataset from {UCI_ONLINE_RETAIL_URL}..."
    )
    try:
        req = urllib.request.Request(
            UCI_ONLINE_RETAIL_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                z.extractall(dest_dir)
        logger.info(f"Successfully downloaded and extracted archive to {dest_dir}")
        for file in dest_dir.glob("*.xlsx"):
            return file
        return None
    except (urllib.error.URLError, TimeoutError, zipfile.BadZipFile, OSError, ValueError) as e:
        logger.warning(f"Download from UCI archive failed or timed out: {e}")
        return None


def load_dataset(
    file_path: str | Path | None = None,
    use_demo_fallback: bool = True,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Load raw transactions from file path, default UCI location, or synthetic generator.

    Returns:
        Tuple of (DataFrame, is_synthetic_boolean).
    """
    requested_path_missing = False
    if file_path:
        path = Path(file_path)
        if path.exists():
            logger.info(f"Loading data from user-specified path: {path}")
            if path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            is_synth = (
                "DataClassification" in df.columns or "is_synthetic" in df.columns
            )
            return df, bool(is_synth)
        else:
            logger.warning(f"Specified path does not exist: {path}")
            if not use_demo_fallback:
                raise FileNotFoundError(f"Specified dataset file not found: {path}")
            # Caller asked for a specific missing file with fallback enabled:
            # prefer clearly labeled synthetic over opportunistically loading
            # a different local UCI file that was never requested.
            requested_path_missing = True

    if not requested_path_missing:
        # Check for existing local files in DATA_DIR
        local_candidates = [
            DATA_DIR / "Online Retail.xlsx",
            DATA_DIR / "online_retail.xlsx",
            DATA_DIR / "online_retail.csv",
            DATA_DIR / "sample_customers_demo.csv",
        ]
        for cand in local_candidates:
            if cand.exists():
                logger.info(f"Found local dataset candidate: {cand}")
                if cand.suffix.lower() in [".xlsx", ".xls"]:
                    df = pd.read_excel(cand)
                else:
                    df = pd.read_csv(cand)
                is_synth = (
                    "DataClassification" in df.columns
                    or cand.name == "sample_customers_demo.csv"
                )
                return df, bool(is_synth)

        # Attempt download from UCI
        downloaded_path = download_uci_dataset(DATA_DIR)
        if downloaded_path and downloaded_path.exists():
            logger.info(f"Reading downloaded UCI dataset from {downloaded_path}")
            df = pd.read_excel(downloaded_path)
            return df, False

    # Fallback to clearly labeled synthetic dataset
    if use_demo_fallback:
        logger.info(
            "Using synthetic demo dataset generator (clearly labeled demo data)."
        )
        demo_path = DATA_DIR / "sample_customers_demo.csv"
        df = generate_synthetic_transactions(output_path=demo_path)
        return df, True

    raise FileNotFoundError(
        "Could not find or download any dataset, and synthetic fallback was disabled."
    )
