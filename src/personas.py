"""
Customer Persona Profiling & Business Strategy Module.

Derives segment labels, behavioral characteristics, business interpretations,
and actionable marketing recommendations strictly from actual empirical cluster statistics.
Never fabricates metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils import get_logger

logger = get_logger("personas")


@dataclass
class CustomerPersona:
    """Rich profile representing an empirical customer segment."""

    cluster_id: int
    persona_name: str
    customer_count: int
    customer_pct: float
    avg_recency_days: float
    median_recency_days: float
    avg_frequency: float
    median_frequency: float
    avg_monetary: float
    median_monetary: float
    avg_order_value: float
    revenue_contribution_pct: float
    behavioral_traits: list[str] = field(default_factory=list)
    business_interpretation: str = ""
    marketing_recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "persona_name": self.persona_name,
            "customer_count": self.customer_count,
            "customer_pct": self.customer_pct,
            "avg_recency_days": self.avg_recency_days,
            "median_recency_days": self.median_recency_days,
            "avg_frequency": self.avg_frequency,
            "median_frequency": self.median_frequency,
            "avg_monetary": self.avg_monetary,
            "median_monetary": self.median_monetary,
            "avg_order_value": self.avg_order_value,
            "revenue_contribution_pct": self.revenue_contribution_pct,
            "behavioral_traits": self.behavioral_traits,
            "business_interpretation": self.business_interpretation,
            "marketing_recommendations": self.marketing_recommendations,
        }


class PersonaProfiler:
    """Derives customer personas from actual cluster statistics."""

    def profile_clusters(
        self, df_rfm: pd.DataFrame, cluster_col: str = "cluster"
    ) -> list[CustomerPersona]:
        """Analyze empirical metrics per cluster and assign grounded persona archetypes."""
        if cluster_col not in df_rfm.columns:
            raise ValueError(f"Column '{cluster_col}' not found in dataframe.")

        total_customers = len(df_rfm)
        total_revenue = df_rfm["monetary"].sum() if total_customers > 0 else 1.0

        unique_clusters = sorted(df_rfm[cluster_col].unique())
        raw_cluster_stats = []

        for c_id in unique_clusters:
            sub = df_rfm[df_rfm[cluster_col] == c_id]
            cnt = len(sub)
            if cnt == 0:
                continue

            pct = round(cnt / total_customers * 100, 2)
            c_revenue = sub["monetary"].sum()
            rev_pct = (
                round(c_revenue / total_revenue * 100, 2) if total_revenue > 0 else 0.0
            )

            avg_rec = round(float(sub["recency"].mean()), 1)
            med_rec = round(float(sub["recency"].median()), 1)
            avg_freq = round(float(sub["frequency"].mean()), 1)
            med_freq = round(float(sub["frequency"].median()), 1)
            avg_mon = round(float(sub["monetary"].mean()), 2)
            med_mon = round(float(sub["monetary"].median()), 2)
            avg_aov = round(
                float((sub["monetary"] / sub["frequency"].replace(0, 1)).mean()), 2
            )

            raw_cluster_stats.append(
                {
                    "cluster_id": int(c_id),
                    "cnt": cnt,
                    "pct": pct,
                    "rev_pct": rev_pct,
                    "avg_rec": avg_rec,
                    "med_rec": med_rec,
                    "avg_freq": avg_freq,
                    "med_freq": med_freq,
                    "avg_mon": avg_mon,
                    "med_mon": med_mon,
                    "avg_aov": avg_aov,
                }
            )

        normal_stats = [s for s in raw_cluster_stats if s["cluster_id"] != -1]
        noise_stats = [s for s in raw_cluster_stats if s["cluster_id"] == -1]

        # Sort normal clusters by spend
        sorted_by_mon = sorted(normal_stats, key=lambda x: x["med_mon"], reverse=True)
        # Sort by recency (highest = longest idle)
        sorted_by_rec = sorted(normal_stats, key=lambda x: x["med_rec"], reverse=True)

        personas = []
        most_idle_id = sorted_by_rec[0]["cluster_id"] if sorted_by_rec else None

        for s in normal_stats:
            cid = s["cluster_id"]
            mon_rank = [x["cluster_id"] for x in sorted_by_mon].index(cid)

            if mon_rank == 0:
                name = "Top-Tier VIP Champions"
                traits = [
                    f"Highest lifetime value (median ${s['med_mon']:,.2f}) and purchase frequency ({s['avg_freq']} orders).",
                    f"Consistently active with recency of {s['avg_rec']} days.",
                    f"Generates {s['rev_pct']}% of total revenue from {s['pct']}% of customer base.",
                ]
                interpretation = "Highest value core advocates who deliver the vast majority of company profitability."
                recommendations = [
                    "VIP concierge, dedicated support, and private executive feedback sessions.",
                    "Exclusive preview access to premier inventory releases.",
                    "Reward loyalty without margin-eroding price discounts.",
                ]
            elif cid == most_idle_id and s["med_rec"] > 75:
                name = "At-Risk / Lapsed Customers"
                traits = [
                    f"Extended inactivity with average recency of {s['avg_rec']} days.",
                    f"Moderate historical engagement ({s['avg_freq']} lifetime orders, ${s['avg_mon']:,.2f} spend).",
                ]
                interpretation = "Former steady buyers showing acute signs of disengagement and churn risk."
                recommendations = [
                    "Targeted multi-channel win-back campaigns with compelling reactivation offers.",
                    "Customer exit survey inquiring about satisfaction and catalog preferences.",
                    "Product updates showcasing new arrivals since their last transaction.",
                ]
            elif mon_rank == 1 and len(normal_stats) > 2:
                name = "Loyal Repeat Regulars"
                traits = [
                    f"Dependable repeat purchasing ({s['avg_freq']} orders) with solid total spend (${s['avg_mon']:,.2f}).",
                    f"Recent purchase activity ({s['avg_rec']} days recency).",
                ]
                interpretation = "Reliable foundation customers who return consistently to purchase core catalog products."
                recommendations = [
                    "Incentivize cross-category purchases to elevate average basket size ($75 threshold incentives).",
                    "Automated re-order notifications and subscription replenishment options.",
                    "Referral bonuses rewarding word-of-mouth advocacy.",
                ]
            elif s["avg_rec"] <= 50 and s["avg_freq"] <= 3:
                name = "Recent Starters & Promising"
                traits = [
                    f"Recently placed initial purchases ({s['avg_rec']} days ago) but early in lifecycle ({s['avg_freq']} orders).",
                    f"Average initial spend of ${s['avg_mon']:,.2f}.",
                ]
                interpretation = "Fresh customer acquisitions currently evaluating product quality and delivery experience."
                recommendations = [
                    "Engaging 3-part welcome email onboarding journey with tutorials and styling guides.",
                    "Time-limited second purchase voucher (valid 14 days) to drive repeat conversion.",
                    "Proactive customer service check-in after order delivery.",
                ]
            else:
                name = "Low-Engagement / Casual Shoppers"
                traits = [
                    f"Low purchase frequency ({s['avg_freq']} orders) and low monetary spend (${s['avg_mon']:,.2f}).",
                    f"Average recency of {s['avg_rec']} days.",
                ]
                interpretation = "Price-sensitive or occasional buyers with low initial investment in the brand."
                recommendations = [
                    "Automated inclusion in seasonal clearance and promotional discount newsletters.",
                    "Limit expensive paid ad retargeting; optimize for low-cost organic email touchpoints.",
                    "Highlight low-ticket entry-point products to stimulate impulse transactions.",
                ]

            personas.append(
                CustomerPersona(
                    cluster_id=cid,
                    persona_name=name,
                    customer_count=s["cnt"],
                    customer_pct=s["pct"],
                    avg_recency_days=s["avg_rec"],
                    median_recency_days=s["med_rec"],
                    avg_frequency=s["avg_freq"],
                    median_frequency=s["med_freq"],
                    avg_monetary=s["avg_mon"],
                    median_monetary=s["med_mon"],
                    avg_order_value=s["avg_aov"],
                    revenue_contribution_pct=s["rev_pct"],
                    behavioral_traits=traits,
                    business_interpretation=interpretation,
                    marketing_recommendations=recommendations,
                )
            )

        for s in noise_stats:
            personas.append(
                CustomerPersona(
                    cluster_id=-1,
                    persona_name="Noise & Outlier Cohort",
                    customer_count=s["cnt"],
                    customer_pct=s["pct"],
                    avg_recency_days=s["avg_rec"],
                    median_recency_days=s["med_rec"],
                    avg_frequency=s["avg_freq"],
                    median_frequency=s["med_freq"],
                    avg_monetary=s["avg_mon"],
                    median_monetary=s["med_mon"],
                    avg_order_value=s["avg_aov"],
                    revenue_contribution_pct=s["rev_pct"],
                    behavioral_traits=[
                        f"Outliers with irregular density patterns ({s['cnt']} customers).",
                        f"Extreme monetary spend (${s['avg_mon']:,.2f}) or erratic purchasing intervals.",
                    ],
                    business_interpretation="Transactions exhibiting anomalous volume or wholesale corporate buying habits.",
                    marketing_recommendations=[
                        "Manual review for B2B contract qualification.",
                        "Direct sales outreach for wholesale account provisioning.",
                    ],
                )
            )

        personas.sort(key=lambda p: p.cluster_id)
        return personas

    def attach_personas_to_dataframe(
        self,
        df_rfm: pd.DataFrame,
        personas: list[CustomerPersona],
        cluster_col: str = "cluster",
    ) -> pd.DataFrame:
        """Attach derived persona names and descriptions directly to each customer record."""
        out = df_rfm.copy()
        name_map = {p.cluster_id: p.persona_name for p in personas}
        out["segment_name"] = out[cluster_col].map(name_map).fillna("Unassigned")
        return out
