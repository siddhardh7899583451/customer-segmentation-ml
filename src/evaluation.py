"""
Clustering Evaluation Metrics & Mathematical Validity Guardrails.

Strictly checks mathematical conditions before computing:
- Silhouette Score: requires 2 <= n_clusters < n_samples (excluding noise points).
- Davies-Bouldin Index: requires n_clusters >= 2.
- Calinski-Harabasz Score: requires 2 <= n_clusters < n_samples.
Prevents reporting undefined or misleading metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.utils import get_logger

logger = get_logger("evaluation")


@dataclass
class ClusterEvaluationMetrics:
    """Stores cluster validation metrics and validation flags."""

    n_samples: int
    n_clusters: int
    noise_count: int = 0
    noise_ratio: float = 0.0
    silhouette: float | None = None
    davies_bouldin: float | None = None
    calinski_harabasz: float | None = None
    inertia: float | None = None
    cluster_counts: dict[str, int] = field(default_factory=dict)
    cluster_proportions: dict[str, float] = field(default_factory=dict)
    metrics_valid: bool = False
    validity_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "noise_count": self.noise_count,
            "noise_ratio_pct": round(self.noise_ratio * 100, 2),
            "silhouette": round(self.silhouette, 4)
            if self.silhouette is not None
            else None,
            "davies_bouldin": round(self.davies_bouldin, 4)
            if self.davies_bouldin is not None
            else None,
            "calinski_harabasz": round(self.calinski_harabasz, 2)
            if self.calinski_harabasz is not None
            else None,
            "inertia": round(self.inertia, 2) if self.inertia is not None else None,
            "cluster_counts": self.cluster_counts,
            "cluster_proportions": self.cluster_proportions,
            "metrics_valid": self.metrics_valid,
            "validity_notes": self.validity_notes,
        }


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    inertia: float | None = None,
) -> ClusterEvaluationMetrics:
    """Evaluate clustering quality under strict mathematical validity checks.

    For DBSCAN or algorithms with noise (-1), evaluates silhouette and density metrics
    on non-noise core clusters, ensuring >= 2 distinct clusters exist.
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    has_noise = -1 in unique_labels

    non_noise_mask = labels != -1
    clean_labels = labels[non_noise_mask]
    clean_X = X[non_noise_mask]

    distinct_clusters = np.unique(clean_labels)
    n_clusters = len(distinct_clusters)

    noise_count = int(np.sum(~non_noise_mask)) if has_noise else 0
    noise_ratio = float(noise_count / n_samples) if n_samples > 0 else 0.0

    # Calculate cluster sizes
    cluster_counts = {}
    cluster_proportions = {}
    for lbl in unique_labels:
        key = "Noise (-1)" if lbl == -1 else f"Cluster {lbl}"
        cnt = int(np.sum(labels == lbl))
        cluster_counts[key] = cnt
        cluster_proportions[key] = round(cnt / n_samples, 4) if n_samples > 0 else 0.0

    result = ClusterEvaluationMetrics(
        n_samples=n_samples,
        n_clusters=n_clusters,
        noise_count=noise_count,
        noise_ratio=noise_ratio,
        inertia=inertia,
        cluster_counts=cluster_counts,
        cluster_proportions=cluster_proportions,
    )

    # Validity conditions
    if n_clusters < 2:
        result.metrics_valid = False
        result.validity_notes = f"Insufficient distinct clusters ({n_clusters}). Minimum 2 clusters required."
        logger.warning(result.validity_notes)
        return result

    if len(clean_X) < n_clusters + 1:
        result.metrics_valid = False
        result.validity_notes = f"Non-noise sample count ({len(clean_X)}) is too small for {n_clusters} clusters."
        logger.warning(result.validity_notes)
        return result

    # Compute metrics on valid non-noise clusters
    try:
        result.silhouette = float(silhouette_score(clean_X, clean_labels))
        result.davies_bouldin = float(davies_bouldin_score(clean_X, clean_labels))
        result.calinski_harabasz = float(calinski_harabasz_score(clean_X, clean_labels))
        result.metrics_valid = True
        result.validity_notes = f"Metrics computed successfully across {n_clusters} clusters on {len(clean_X)} non-noise points."
    except (ValueError, TypeError, ZeroDivisionError, FloatingPointError) as e:
        result.metrics_valid = False
        result.validity_notes = f"Mathematical error during metric calculation: {e}"
        logger.error(result.validity_notes)

    return result
