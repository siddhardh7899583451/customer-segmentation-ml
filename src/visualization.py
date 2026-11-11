"""
Data Visualization Module.

Generates publication-quality charts for:
- Elbow & Silhouette curve for optimal k selection
- PCA 2D cluster projection scatter plots
- RFM metric distribution histograms & KDE
- Customer count vs Revenue contribution by persona
- Recency vs Monetary behavioral scatter plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend suitable for headless scripts
import matplotlib.pyplot as plt

# Default DPI kept moderate to reduce peak memory on constrained hosts.
# Override per-call or via PipelineConfig.figure_dpi. Does not affect model math.
DEFAULT_FIGURE_DPI = 100
import numpy as np
import pandas as pd
import seaborn as sns

from src.personas import CustomerPersona
from src.utils import get_logger

logger = get_logger("visualization")

# Professional theme settings
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#6B7280"]


def plot_elbow_and_silhouette(
    elbow_data: list[dict[str, Any]],
    save_path: Path | str | None = None,
) -> Path:
    """Plot dual-axis Elbow (Inertia) and Silhouette score curve across k."""
    ks = [d["k"] for d in elbow_data]
    inertias = [d["inertia"] for d in elbow_data]
    silhouettes = [d["silhouette"] for d in elbow_data if d["silhouette"] is not None]

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=DEFAULT_FIGURE_DPI)

    color_inertia = "#1E40AF"
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(
        "Inertia (Within-Cluster Sum of Squares)",
        color=color_inertia,
        fontsize=11,
        fontweight="bold",
    )
    line1 = ax1.plot(
        ks,
        inertias,
        marker="o",
        color=color_inertia,
        linewidth=2.2,
        label="Inertia (Elbow)",
    )
    ax1.tick_params(axis="y", labelcolor=color_inertia)
    ax1.grid(True, linestyle="--", alpha=0.5)

    if silhouettes:
        ax2 = ax1.twinx()
        color_sil = "#059669"
        ax2.set_ylabel(
            "Mean Silhouette Score", color=color_sil, fontsize=11, fontweight="bold"
        )
        line2 = ax2.plot(
            ks[: len(silhouettes)],
            silhouettes,
            marker="s",
            color=color_sil,
            linewidth=2.2,
            linestyle="--",
            label="Silhouette Score",
        )
        ax2.tick_params(axis="y", labelcolor=color_sil)
        ax2.grid(False)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title(
        "Cluster Evaluation: Elbow Method & Silhouette Analysis",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()

    out_p = (
        Path(save_path)
        if save_path
        else Path("output/figures/elbow_silhouette_curve.png")
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Elbow/Silhouette plot to {out_p}")
    return out_p


def plot_pca_clusters(
    X_pca: np.ndarray,
    labels: np.ndarray,
    segment_names: list[str] | None = None,
    variance_ratio: list[float] | None = None,
    save_path: Path | str | None = None,
) -> Path:
    """Plot 2D PCA cluster projection with clear cluster separation and variance ratios."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=DEFAULT_FIGURE_DPI)

    unique_labels = sorted(np.unique(labels))
    var_text = ""
    if variance_ratio and len(variance_ratio) >= 2:
        var_text = f" (PC1: {variance_ratio[0] * 100:.1f}%, PC2: {variance_ratio[1] * 100:.1f}% Variance)"

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = "#6B7280" if lbl == -1 else PALETTE[i % len(PALETTE)]
        name = (
            "Noise (-1)"
            if lbl == -1
            else (
                segment_names[lbl]
                if segment_names and lbl < len(segment_names)
                else f"Cluster {lbl}"
            )
        )
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=f"{name} (n={mask.sum()})",
            alpha=0.65,
            edgecolors="none",
            s=40,
        )

    ax.set_title(
        f"Customer Clusters in 2D PCA Space{var_text}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Principal Component 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("Principal Component 2", fontsize=11, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_p = (
        Path(save_path)
        if save_path
        else Path("output/figures/pca_cluster_projection.png")
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved PCA plot to {out_p}")
    return out_p


def plot_rfm_distributions(
    df_rfm: pd.DataFrame,
    save_path: Path | str | None = None,
) -> Path:
    """Plot histograms and KDE for raw Recency, Frequency, and Monetary metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=DEFAULT_FIGURE_DPI)

    # Recency
    sns.histplot(df_rfm["recency"], kde=True, ax=axes[0], color="#2563EB", bins=30)
    axes[0].set_title("Recency Distribution (Days)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Days Since Last Purchase")
    axes[0].set_ylabel("Customer Count")

    # Frequency (log scale visual)
    sns.histplot(df_rfm["frequency"], kde=True, ax=axes[1], color="#10B981", bins=30)
    axes[1].set_title("Frequency Distribution (Orders)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Order Count")
    axes[1].set_ylabel("Customer Count")

    # Monetary (filtered to 98th percentile for clean display)
    cap_val = df_rfm["monetary"].quantile(0.98)
    filtered_mon = df_rfm["monetary"][df_rfm["monetary"] <= cap_val]
    sns.histplot(filtered_mon, kde=True, ax=axes[2], color="#F59E0B", bins=30)
    axes[2].set_title(
        f"Monetary Spend (Up to 98th percentile: ${cap_val:,.0f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[2].set_xlabel("Total Spend ($)")
    axes[2].set_ylabel("Customer Count")

    plt.suptitle(
        "Customer Behavioral & RFM Distributions",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_p = (
        Path(save_path) if save_path else Path("output/figures/rfm_distributions.png")
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved RFM distributions plot to {out_p}")
    return out_p


def plot_persona_revenue_comparison(
    personas: list[CustomerPersona],
    save_path: Path | str | None = None,
) -> Path:
    """Plot dual-bar chart showing customer volume % vs revenue contribution % per persona."""
    names = [p.persona_name for p in personas]
    cust_pcts = [p.customer_pct for p in personas]
    rev_pcts = [p.revenue_contribution_pct for p in personas]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DEFAULT_FIGURE_DPI)
    rects1 = ax.bar(
        x - width / 2, cust_pcts, width, label="% of Customer Base", color="#3B82F6"
    )
    rects2 = ax.bar(
        x + width / 2, rev_pcts, width, label="% of Total Revenue", color="#10B981"
    )

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Customer Volume vs. Revenue Contribution by Segment",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Label heights
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_p = (
        Path(save_path)
        if save_path
        else Path("output/figures/persona_revenue_comparison.png")
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Persona revenue comparison plot to {out_p}")
    return out_p
