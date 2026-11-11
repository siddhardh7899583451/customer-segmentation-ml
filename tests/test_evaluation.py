"""
Unit tests for clustering evaluation and mathematical validity guards.
"""

import numpy as np

from src.evaluation import evaluate_clustering


def test_evaluate_valid_clustering():
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[-3, -3], scale=0.5, size=(40, 2))
    c2 = rng.normal(loc=[3, 3], scale=0.5, size=(40, 2))
    X = np.vstack([c1, c2])
    labels = np.array([0] * 40 + [1] * 40)

    res = evaluate_clustering(X, labels)
    assert res.metrics_valid is True
    assert res.n_clusters == 2
    assert res.silhouette is not None
    assert res.silhouette > 0.6
    assert res.davies_bouldin is not None
    assert res.calinski_harabasz is not None


def test_evaluate_single_cluster_is_invalid():
    X = np.random.rand(20, 2)
    labels = np.zeros(20, dtype=int)  # All in cluster 0

    res = evaluate_clustering(X, labels)
    assert res.metrics_valid is False
    assert res.silhouette is None
    assert "Minimum 2 clusters required" in res.validity_notes


def test_evaluate_with_noise_points():
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[-3, -3], scale=0.3, size=(20, 2))
    c2 = rng.normal(loc=[3, 3], scale=0.3, size=(20, 2))
    noise = np.array([[10.0, 10.0], [-10.0, 10.0]])
    X = np.vstack([c1, c2, noise])
    labels = np.array([0] * 20 + [1] * 20 + [-1, -1])

    res = evaluate_clustering(X, labels)
    assert res.noise_count == 2
    assert res.noise_ratio > 0.0
    assert res.n_clusters == 2
    assert res.metrics_valid is True
