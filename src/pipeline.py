"""
End-to-End Machine Learning & Behavioral Analytics Pipeline Orchestrator.

Coordinates ingestion, quality reporting, RFM derivation, scaling,
multi-model clustering, evaluation, persona generation, visualization, and export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    PipelineConfig,
)
from src.data_loader import load_dataset
from src.feature_engineering import RFMFeatureExtractor
from src.models import DimensionalityReducer, KMeansSegmenter, ModelComparator
from src.personas import PersonaProfiler
from src.preprocessor import DataCleaner, DataQualityReport
from src.utils import get_logger, safe_export_csv, save_json_report
from src.visualization import (
    plot_elbow_and_silhouette,
    plot_pca_clusters,
    plot_persona_revenue_comparison,
    plot_rfm_distributions,
)

logger = get_logger("pipeline")


class CustomerSegmentationPipeline:
    """Orchestrates end-to-end data processing, clustering, and reporting."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.cleaner = DataCleaner(self.config.preprocessing)
        self.extractor = RFMFeatureExtractor(
            log_transform=self.config.preprocessing.log_transform_features,
            scaler_type="standard",
        )
        self.profiler = PersonaProfiler()
        self.dim_reducer = DimensionalityReducer(
            random_state=self.config.modeling.random_state
        )

    def run(
        self,
        raw_data_path: str | Path | None = None,
        k: int | None = None,
        use_demo_fallback: bool = True,
    ) -> dict[str, Any]:
        """Execute full segmentation pipeline."""
        logger.info("=================================================================")
        logger.info("STARTING CUSTOMER SEGMENTATION & BEHAVIORAL ANALYTICS PIPELINE")
        logger.info("=================================================================")

        # Step 1: Load Data
        df_raw, is_synthetic = load_dataset(
            raw_data_path, use_demo_fallback=use_demo_fallback
        )
        logger.info(
            f"Loaded dataset containing {len(df_raw)} records. (is_synthetic={is_synthetic})"
        )

        # Step 2: Determine if input is raw transactions or pre-aggregated RFM
        is_transaction_level = any(
            c in df_raw.columns for c in ["InvoiceNo", "invoiceno", "InvoiceDate"]
        )

        reports_dir = self.config.output_dir / "reports"
        figures_dir = self.config.output_dir / "figures"
        reports_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        if is_transaction_level:
            logger.info(
                "Input recognized as transaction-level logs. Cleaning records..."
            )
            valid_df, quality_report, canc_df = self.cleaner.clean(df_raw)
            # Save data quality report
            save_json_report(
                quality_report.to_dict(), reports_dir / "data_quality_report.json"
            )
            # Extract RFM features
            df_rfm = self.extractor.extract_from_transactions(valid_df, canc_df)
        else:
            logger.info("Input recognized as pre-aggregated customer-level RFM data.")
            quality_report = DataQualityReport(
                total_raw_rows=len(df_raw),
                final_valid_rows=len(df_raw),
                unique_customers=len(df_raw),
            )
            save_json_report(
                quality_report.to_dict(), reports_dir / "data_quality_report.json"
            )
            df_rfm = self.extractor.process_customer_rfm(df_raw)

        if len(df_rfm) < 4:
            raise ValueError(
                f"Insufficient unique customer records ({len(df_rfm)}) to perform clustering."
            )

        # Step 3: Feature Scaling & Transformation
        X_scaled, feature_names = self.extractor.prepare_features_for_clustering(df_rfm)
        logger.info(
            f"Prepared feature matrix X of shape {X_scaled.shape} using features: {feature_names}"
        )

        # Step 4: Model Exploration (Optimal K search)
        chosen_k = k or self.config.modeling.default_k
        kmeans_seg = KMeansSegmenter(self.config.modeling)
        elbow_data = kmeans_seg.search_optimal_k(X_scaled, self.config.modeling.k_range)
        save_json_report(
            {"elbow_analysis": elbow_data}, reports_dir / "elbow_analysis.json"
        )

        # Step 5: Multi-Model Comparison (K-Means, DBSCAN, Agglomerative)
        comparator = ModelComparator(self.config.modeling)
        comparison_res = comparator.compare_all(X_scaled, k=chosen_k)
        save_json_report(
            {
                "comparison_table": comparison_res["comparison_table"],
                "recommended_model": comparison_res["recommended_model"],
                "selection_rationale": comparison_res["selection_rationale"],
            },
            reports_dir / "model_comparison.json",
        )

        # Primary production model: K-Means
        primary_result = comparison_res["models"]["kmeans"]
        df_rfm["cluster"] = primary_result.labels

        # Step 6: Dimensionality Reduction (PCA)
        X_pca, var_ratio = self.dim_reducer.fit_transform_2d(X_scaled)
        df_rfm["pca_1"] = np.round(X_pca[:, 0], 3)
        df_rfm["pca_2"] = np.round(X_pca[:, 1], 3)

        # Step 7: Persona Derivation & Business Strategy Mapping
        personas = self.profiler.profile_clusters(df_rfm, cluster_col="cluster")
        df_segmented = self.profiler.attach_personas_to_dataframe(
            df_rfm, personas, cluster_col="cluster"
        )
        save_json_report(
            {"customer_personas": [p.to_dict() for p in personas]},
            reports_dir / "customer_personas.json",
        )

        # Step 8: Generate Visualizations
        plot_elbow_and_silhouette(
            elbow_data, figures_dir / "elbow_silhouette_curve.png"
        )
        plot_pca_clusters(
            X_pca,
            primary_result.labels,
            segment_names=[p.persona_name for p in personas],
            variance_ratio=var_ratio,
            save_path=figures_dir / "pca_cluster_projection.png",
        )
        plot_rfm_distributions(df_rfm, figures_dir / "rfm_distributions.png")
        plot_persona_revenue_comparison(
            personas, figures_dir / "persona_revenue_comparison.png"
        )

        # Step 9: Safe CSV Export
        out_csv_path = self.config.output_dir / "segmented_customers.csv"
        safe_export_csv(df_segmented, out_csv_path)
        logger.info(f"Segmented customer dataset safely exported to {out_csv_path}")

        logger.info("=================================================================")
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=================================================================")

        return {
            "df_segmented": df_segmented,
            "quality_report": quality_report,
            "elbow_data": elbow_data,
            "comparison": comparison_res,
            "personas": personas,
            "is_synthetic": is_synthetic,
            "output_csv": out_csv_path,
        }
