"""
Command-Line Interface for the Customer Segmentation & Behavioral Analytics Platform.

Usage:
    python main.py
    python main.py --demo
    python main.py --data data/Online Retail.xlsx --k 4
    python main.py --k 5 --output output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import PipelineConfig
from src.pipeline import CustomerSegmentationPipeline
from src.utils import get_logger

logger = get_logger("cli")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Customer Segmentation & Behavioral Analytics Machine Learning Platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to transaction Excel/CSV or pre-aggregated RFM CSV file. If omitted, attempts to use UCI Online Retail or synthetic demo.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of customer segments/clusters to fit with K-Means (default: 4).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force the generation and usage of the clearly labeled synthetic demo dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output directory for segmented data, figures, and reports.",
    )

    args = parser.parse_args()

    config = PipelineConfig()
    if args.output:
        config.output_dir = Path(args.output)

    pipeline = CustomerSegmentationPipeline(config=config)

    try:
        data_path = None
        if args.demo:
            data_path = config.synthetic_demo_path
            # Ensure demo data is generated if not exists
            from src.synthetic_data import generate_synthetic_transactions

            generate_synthetic_transactions(output_path=data_path)
        elif args.data:
            data_path = Path(args.data)

        results = pipeline.run(
            raw_data_path=data_path, k=args.k, use_demo_fallback=True
        )

        print("\n=================== EXECUTION SUMMARY ===================")
        print(
            f"Data source classification: {'DEMO / SYNTHETIC' if results['is_synthetic'] else 'REAL CUSTOMER DATA'}"
        )
        print(f"Total customers segmented: {len(results['df_segmented'])}")
        print(f"Recommended model: {results['comparison']['recommended_model']}")
        print("\nDiscovered Customer Personas:")
        for p in results["personas"]:
            print(
                f"  • [{p.cluster_id}] {p.persona_name}: {p.customer_count} customers ({p.customer_pct}%), "
                f"Avg Spend: ${p.avg_monetary:,.2f}, Avg Recency: {p.avg_recency_days} days"
            )
        print(f"\nOutputs saved to: {config.output_dir.resolve()}")
        print("  - Segmented CSV: output/segmented_customers.csv")
        print("  - Reports:       output/reports/")
        print("  - Figures:       output/figures/")
        print("=========================================================\n")
        return 0

    except Exception:
        logger.exception("Pipeline execution encountered an error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
