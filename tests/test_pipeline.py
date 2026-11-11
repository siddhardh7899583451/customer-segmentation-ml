"""
Integration tests for the end-to-end segmentation pipeline.
"""

from src.config import PipelineConfig
from src.pipeline import CustomerSegmentationPipeline


def test_pipeline_execution_end_to_end(tmp_path):
    config = PipelineConfig()
    config.output_dir = tmp_path
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pipeline = CustomerSegmentationPipeline(config=config)
    res = pipeline.run(k=3, use_demo_fallback=True)

    assert "df_segmented" in res
    assert len(res["df_segmented"]) >= 50
    assert "personas" in res
    assert len(res["personas"]) == 3
    assert "comparison" in res
    assert res["comparison"]["recommended_model"] == "K-Means (k-means++)"

    # Verify exported artifacts
    assert (tmp_path / "segmented_customers.csv").exists()
    assert (reports_dir / "data_quality_report.json").exists()
    assert (reports_dir / "customer_personas.json").exists()
    assert (reports_dir / "model_comparison.json").exists()
    assert (figures_dir / "pca_cluster_projection.png").exists()
    assert (figures_dir / "elbow_silhouette_curve.png").exists()
