"""
Unit tests for persona profiling and business recommendations module.
"""

import pandas as pd
import pytest

from src.personas import CustomerPersona, PersonaProfiler


@pytest.fixture
def mock_clustered_rfm():
    return pd.DataFrame(
        [
            # High value cluster 0
            {
                "customer_id": "C1",
                "recency": 4,
                "frequency": 25,
                "monetary": 3500.0,
                "cluster": 0,
            },
            {
                "customer_id": "C2",
                "recency": 8,
                "frequency": 20,
                "monetary": 3200.0,
                "cluster": 0,
            },
            # Regular loyal cluster 1
            {
                "customer_id": "C3",
                "recency": 25,
                "frequency": 10,
                "monetary": 1100.0,
                "cluster": 1,
            },
            {
                "customer_id": "C4",
                "recency": 30,
                "frequency": 12,
                "monetary": 1250.0,
                "cluster": 1,
            },
            # At risk cluster 2
            {
                "customer_id": "C5",
                "recency": 200,
                "frequency": 4,
                "monetary": 400.0,
                "cluster": 2,
            },
            {
                "customer_id": "C6",
                "recency": 240,
                "frequency": 3,
                "monetary": 350.0,
                "cluster": 2,
            },
        ]
    )


def test_profiler_generates_grounded_personas(mock_clustered_rfm):
    profiler = PersonaProfiler()
    personas = profiler.profile_clusters(mock_clustered_rfm)

    assert len(personas) == 3
    for p in personas:
        assert isinstance(p, CustomerPersona)
        assert p.customer_count > 0
        assert len(p.marketing_recommendations) > 0
        assert len(p.business_interpretation) > 0

    # Total revenue contribution should sum close to 100%
    total_rev_pct = sum(p.revenue_contribution_pct for p in personas)
    assert abs(total_rev_pct - 100.0) < 1.0


def test_attach_personas_to_dataframe(mock_clustered_rfm):
    profiler = PersonaProfiler()
    personas = profiler.profile_clusters(mock_clustered_rfm)
    df_out = profiler.attach_personas_to_dataframe(mock_clustered_rfm, personas)

    assert "segment_name" in df_out.columns
    assert (df_out["segment_name"] != "Unassigned").all()


def test_profiler_missing_cluster_col():
    profiler = PersonaProfiler()
    with pytest.raises(ValueError):
        profiler.profile_clusters(
            pd.DataFrame({"recency": [1], "frequency": [1], "monetary": [100]})
        )
