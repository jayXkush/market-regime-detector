"""
Visualizer — Phase 3: Generate clustering and regime visualizations.

Generates three plots:
    1. PCA scatter plot (2D projection of PCA components, colored by cluster)
    2. Cluster visualization (cluster centroids, boundaries, and distributions)
    3. Regime distribution chart (bar chart showing regime proportions)

All plots saved to reports/ directory.

Public API:
    visualizer = Visualizer()
    visualizer.plot_pca_scatter(X_pca, labels, regime_map)
    visualizer.plot_cluster_visualization(X_pca, labels, regime_map)
    visualizer.plot_regime_distribution(labels, regime_map)
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "reports")

# Color palette for regimes (curated for visual distinction)
REGIME_COLORS = {
    "High Volatility": "#E74C3C",
    "Low Volatility": "#2ECC71",
    "Trending Up": "#3498DB",
    "Trending Down": "#E67E22",
    "Mean Reverting": "#9B59B6",
    "Transitional": "#F1C40F",
    "Noise": "#95A5A6",
}

# Fallback palette for cluster IDs
CLUSTER_COLORS = [
    "#E74C3C", "#2ECC71", "#3498DB", "#E67E22",
    "#9B59B6", "#F1C40F", "#1ABC9C", "#E91E63",
    "#00BCD4", "#FF9800",
]


class Visualizer:
    """Generate Phase 3 visualizations."""

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = output_dir or REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use("dark_background")
        logger.info("Visualizer initialised — output_dir=%s", self.output_dir)

    def plot_pca_scatter(
        self,
        X_pca: np.ndarray,
        labels: np.ndarray,
        regime_map: dict[int, str],
        filename: str = "pca_scatter.png",
    ) -> str:
        """
        2D PCA scatter plot colored by cluster/regime label.

        Parameters
        ----------
        X_pca : np.ndarray
            PCA-transformed features (uses first 2 components).
        labels : np.ndarray
            Cluster assignments.
        regime_map : dict[int, str]
            Cluster → regime label mapping.
        filename : str
            Output filename.

        Returns
        -------
        str
            Path to saved plot.
        """
        logger.info("Generating PCA scatter plot …")

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        unique_labels = sorted(np.unique(labels))
        legend_handles = []

        for cid in unique_labels:
            mask = labels == cid
            regime = regime_map.get(cid, f"Cluster {cid}")
            color = REGIME_COLORS.get(regime, CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])

            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[color],
                label=f"{regime} (n={mask.sum()})",
                alpha=0.65,
                s=30,
                edgecolors="white",
                linewidths=0.3,
            )
            legend_handles.append(
                mpatches.Patch(color=color, label=f"{regime} (n={mask.sum()})")
            )

        ax.set_xlabel("PC1", fontsize=12, color="white", fontweight="bold")
        ax.set_ylabel("PC2", fontsize=12, color="white", fontweight="bold")
        ax.set_title(
            "PCA Scatter Plot — Market Regimes",
            fontsize=16, color="white", fontweight="bold", pad=20,
        )

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=9,
            framealpha=0.7,
            facecolor="#1a1a2e",
            edgecolor="white",
        )

        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("Saved PCA scatter → %s", save_path)
        return save_path

    def plot_cluster_visualization(
        self,
        X_pca: np.ndarray,
        labels: np.ndarray,
        regime_map: dict[int, str],
        filename: str = "cluster_visualization.png",
    ) -> str:
        """
        Cluster visualization with density contours and centroids.

        Shows:
            - Scatter points colored by cluster
            - Cluster centroids marked with stars
            - 2D density contours for each cluster
            - Cluster boundary regions

        Parameters
        ----------
        X_pca : np.ndarray
            PCA-transformed features.
        labels : np.ndarray
            Cluster assignments.
        regime_map : dict[int, str]
            Cluster → regime mapping.
        filename : str
            Output filename.

        Returns
        -------
        str
            Path to saved plot.
        """
        logger.info("Generating cluster visualization …")

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.patch.set_facecolor("#1a1a2e")

        unique_labels = sorted([l for l in np.unique(labels) if l >= 0])

        # ── Left panel: Scatter with centroids ───────────────────────
        ax1 = axes[0]
        ax1.set_facecolor("#16213e")

        for cid in unique_labels:
            mask = labels == cid
            regime = regime_map.get(cid, f"Cluster {cid}")
            color = REGIME_COLORS.get(regime, CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])

            ax1.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=[color], alpha=0.5, s=20, label=regime,
            )

            # Centroid
            cx = X_pca[mask, 0].mean()
            cy = X_pca[mask, 1].mean()
            ax1.scatter(
                cx, cy,
                c=[color], marker="*", s=300,
                edgecolors="white", linewidths=1.5, zorder=5,
            )
            ax1.annotate(
                regime,
                (cx, cy),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
            )

        # Noise points
        noise_mask = labels == -1
        if noise_mask.any():
            ax1.scatter(
                X_pca[noise_mask, 0], X_pca[noise_mask, 1],
                c=["#95A5A6"], alpha=0.3, s=10, marker="x", label="Noise",
            )

        ax1.set_xlabel("PC1", fontsize=11, color="white")
        ax1.set_ylabel("PC2", fontsize=11, color="white")
        ax1.set_title(
            "Cluster Centroids & Boundaries",
            fontsize=14, color="white", fontweight="bold",
        )
        ax1.grid(True, alpha=0.15, color="white")
        ax1.tick_params(colors="white")

        # ── Right panel: Cluster size bar chart ──────────────────────
        ax2 = axes[1]
        ax2.set_facecolor("#16213e")

        regimes = [regime_map.get(cid, f"Cluster {cid}") for cid in unique_labels]
        sizes = [(labels == cid).sum() for cid in unique_labels]
        colors = [REGIME_COLORS.get(r, CLUSTER_COLORS[i % len(CLUSTER_COLORS)]) for i, r in enumerate(regimes)]

        bars = ax2.barh(regimes, sizes, color=colors, edgecolor="white", linewidth=0.5)

        for bar, size in zip(bars, sizes):
            ax2.text(
                bar.get_width() + max(sizes) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{size}",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

        ax2.set_xlabel("Number of Samples", fontsize=11, color="white")
        ax2.set_title(
            "Cluster Sizes",
            fontsize=14, color="white", fontweight="bold",
        )
        ax2.tick_params(colors="white")
        ax2.invert_yaxis()

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("Saved cluster visualization → %s", save_path)
        return save_path

    def plot_regime_distribution(
        self,
        labels: np.ndarray,
        regime_map: dict[int, str],
        filename: str = "regime_distribution.png",
    ) -> str:
        """
        Regime distribution chart — pie chart + statistics.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments.
        regime_map : dict[int, str]
            Cluster → regime mapping.
        filename : str
            Output filename.

        Returns
        -------
        str
            Path to saved plot.
        """
        logger.info("Generating regime distribution chart …")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor("#1a1a2e")

        # Count occurrences of each regime
        regime_counts = {}
        for cid in sorted(np.unique(labels)):
            regime = regime_map.get(cid, f"Cluster {cid}")
            count = int((labels == cid).sum())
            if regime in regime_counts:
                regime_counts[regime] += count
            else:
                regime_counts[regime] = count

        regimes = list(regime_counts.keys())
        counts = list(regime_counts.values())
        total = sum(counts)
        colors = [REGIME_COLORS.get(r, "#95A5A6") for r in regimes]

        # ── Left: Donut chart ────────────────────────────────────────
        ax1.set_facecolor("#1a1a2e")
        wedges, texts, autotexts = ax1.pie(
            counts,
            labels=regimes,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="#1a1a2e", linewidth=2),
        )

        for text in texts:
            text.set_color("white")
            text.set_fontsize(10)
            text.set_fontweight("bold")
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(9)
            autotext.set_fontweight("bold")

        # Center text
        ax1.text(
            0, 0,
            f"{total}\nsamples",
            ha="center", va="center",
            fontsize=16, color="white", fontweight="bold",
        )

        ax1.set_title(
            "Regime Distribution",
            fontsize=14, color="white", fontweight="bold", pad=20,
        )

        # ── Right: Summary table ─────────────────────────────────────
        ax2.set_facecolor("#16213e")
        ax2.axis("off")

        table_data = []
        for regime, count in zip(regimes, counts):
            pct = count / total * 100
            table_data.append([regime, str(count), f"{pct:.1f}%"])

        table = ax2.table(
            cellText=table_data,
            colLabels=["Regime", "Count", "Percentage"],
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.8)

        # Style table
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_linewidth(0.5)
            if row == 0:  # Header
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                regime_name = table_data[row - 1][0]
                base_color = REGIME_COLORS.get(regime_name, "#34495e")
                cell.set_facecolor(to_rgba(base_color, alpha=0.3))
                cell.set_text_props(color="white")

        ax2.set_title(
            "Regime Summary",
            fontsize=14, color="white", fontweight="bold", pad=20,
        )

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("Saved regime distribution → %s", save_path)
        return save_path

    def generate_all(
        self,
        X_pca: np.ndarray,
        labels: np.ndarray,
        regime_map: dict[int, str],
    ) -> list[str]:
        """Generate all three visualizations. Returns list of saved paths."""
        paths = [
            self.plot_pca_scatter(X_pca, labels, regime_map),
            self.plot_cluster_visualization(X_pca, labels, regime_map),
            self.plot_regime_distribution(labels, regime_map),
        ]
        logger.info("All visualizations generated: %d plots saved.", len(paths))
        return paths
