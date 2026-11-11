"""
Project Configuration and Hyperparameter Management.
Centralizes paths, validation thresholds, model settings, and defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment" / "huggingface-space"

# Dataset Source URLs
UCI_ONLINE_RETAIL_URL = (
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
)

# Core Schema Definitions
RAW_TRANSACTION_COLS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

CORE_RFM_COLS = ["recency", "frequency", "monetary"]

EXTENDED_BEHAVIORAL_COLS = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "return_rate",
    "unique_items",
]


@dataclass
class PreprocessingConfig:
    """Settings for data cleaning and validation."""

    min_unit_price: float = 0.001
    min_quantity: int = 1
    handle_cancellations: bool = True  # Invoices starting with 'C'
    remove_missing_customers: bool = True
    iqr_multiplier: float = (
        3.0  # Used for identifying extreme monetary/frequency outliers
    )
    log_transform_features: bool = (
        True  # Useful for reducing right-skewness of monetary & frequency
    )


@dataclass
class ModelingConfig:
    """Hyperparameters for clustering algorithms."""

    random_state: int = 42
    k_range: tuple[int, int] = (2, 8)
    default_k: int = 4
    kmeans_init: str = "k-means++"
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    # Sensitivity-tuned on standardized synthetic RFM. eps=0.65 collapses to
    # one cluster; eps=0.45 (min_samples=5) yields multiple clusters with ~1% noise.
    # Under the same min_samples=5, eps=0.40 can score a higher Silhouette with more
    # micro-clusters — see output/reports/dbscan_sensitivity_analysis.json.
    # K-Means++ remains the primary recommended model.
    dbscan_eps: float = 0.45
    dbscan_min_samples: int = 5
    hierarchical_linkage: str = "ward"
    features_to_use: list[str] = field(
        default_factory=lambda: ["recency", "frequency", "monetary"]
    )


@dataclass
class PipelineConfig:
    """Global configuration wrapper."""

    raw_data_path: Path = DATA_DIR / "online_retail.xlsx"
    synthetic_demo_path: Path = DATA_DIR / "sample_customers_demo.csv"
    output_dir: Path = OUTPUT_DIR
    # Lower default DPI reduces peak memory for matplotlib figure buffers on
    # constrained hosts. Mathematical results are unaffected. Override via
    # PipelineConfig(figure_dpi=150) for publication-quality plots.
    figure_dpi: int = 100
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)

    def __post_init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
