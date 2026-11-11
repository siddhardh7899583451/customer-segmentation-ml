"""
Unit tests for clustering models and PCA dimensionality reduction.
"""

import numpy as np
import pytest

from src.models import (
    AgglomerativeSegmenter,
    DBSCANSegmenter,
    DimensionalityReducer,
    KMeansSegmenter,
    ModelComparator,
)


@pytest.fixture
def synthetic_features():
    rng = np.random.default_rng(42)
    # Generate 3 distinct blobs of 30 samples each (total 90 samples)
    b1 = rng.normal(loc=[-2, -2, -2], scale=0.4, size=(30, 3))
    b2 = rng.normal(loc=[0, 0, 0], scale=0.4, size=(30, 3))
    b3 = rng.normal(loc=[3, 3, 3], scale=0.4, size=(30, 3))
    return np.vstack([b1, b2, b3])


def test_kmeans_fitting(synthetic_features):
    km = KMeansSegmenter()
    res = km.fit(synthetic_features, k=3)
    assert res.n_clusters == 3
    assert len(res.labels) == len(synthetic_features)
    assert res.metrics.metrics_valid is True
    assert res.metrics.silhouette > 0.4
    assert res.cluster_centers.shape == (3, 3)


def test_kmeans_search_optimal_k(synthetic_features):
    km = KMeansSegmenter()
    results = km.search_optimal_k(synthetic_features, k_range=(2, 5))
    assert len(results) == 4
    for r in results:
        assert "inertia" in r
        assert "silhouette" in r
        assert r["k"] in [2, 3, 4, 5]


def test_dbscan_fitting(synthetic_features):
    db = DBSCANSegmenter()
    res = db.fit(synthetic_features, eps=0.8, min_samples=5)
    assert len(res.labels) == len(synthetic_features)
    assert res.algorithm_name == "DBSCAN"


def test_agglomerative_fitting(synthetic_features):
    agg = AgglomerativeSegmenter()
    res = agg.fit(synthetic_features, k=3)
    assert res.n_clusters == 3
    assert len(res.labels) == len(synthetic_features)
    assert res.metrics.metrics_valid is True


def test_pca_dimensionality_reducer(synthetic_features):
    reducer = DimensionalityReducer(random_state=42)
    X_pca, var_ratio = reducer.fit_transform_2d(synthetic_features)
    assert X_pca.shape == (len(synthetic_features), 2)
    assert len(var_ratio) == 2
    assert sum(var_ratio) <= 1.0001


def test_model_comparator(synthetic_features):
    comparator = ModelComparator()
    comp = comparator.compare_all(synthetic_features, k=3)
    assert "comparison_table" in comp
    assert len(comp["comparison_table"]) == 3
    assert "recommended_model" in comp
    assert "selection_rationale" in comp
