"""
RegimeLabeler — Phase 3: Assign interpretable regime labels to clusters.

For each cluster, computes:
    • Average volatility
    • Average return
    • Volume profile (average volume delta & imbalance)
    • Order flow profile (average bid/ask imbalance, spread, depth imbalance)

Then assigns one of six regime labels:
    1. High Volatility
    2. Low Volatility
    3. Trending Up
    4. Trending Down
    5. Mean Reverting
    6. Transitional

Public API:
    labeler = RegimeLabeler()
    regime_map, profiles = labeler.label_clusters(labels, features_df)
    report = labeler.generate_report(regime_map, profiles, scores)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "reports")

# Regime label definitions
REGIME_LABELS = [
    "High Volatility",
    "Low Volatility",
    "Trending Up",
    "Trending Down",
    "Mean Reverting",
    "Transitional",
]


class RegimeLabeler:
    """
    Assign interpretable regime labels to cluster IDs based on feature profiles.

    The labeling is done via a rule-based scoring system that examines
    each cluster's average volatility, return, volume, and order-flow
    characteristics.
    """

    def __init__(self) -> None:
        self.regime_map_: dict[int, str] = {}
        self.cluster_profiles_: dict[int, dict] = {}

    def label_clusters(
        self,
        labels: np.ndarray,
        features_df: pd.DataFrame,
    ) -> tuple[dict[int, str], dict[int, dict]]:
        """
        Compute cluster profiles and assign regime labels.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments (may include -1 for noise).
        features_df : pd.DataFrame
            Original feature DataFrame (pre-PCA, from features.parquet).

        Returns
        -------
        regime_map : dict[int, str]
            Mapping of cluster ID → regime label.
        cluster_profiles : dict[int, dict]
            Per-cluster feature statistics.
        """
        logger.info("=" * 60)
        logger.info("  RegimeLabeler — label_clusters()")
        logger.info("=" * 60)

        # Identify feature columns by category
        meta_cols = {"open_time", "symbol"}
        feature_cols = [c for c in features_df.columns if c not in meta_cols]

        vol_cols = [c for c in feature_cols if c.startswith("volatility_")]
        mom_cols = [c for c in feature_cols if c.startswith("momentum_")]
        vol_delta_cols = [c for c in feature_cols if c.startswith("volume_delta_")]
        vol_imb_cols = [c for c in feature_cols if c.startswith("volume_imbalance_")]
        of_ba_cols = [c for c in feature_cols if c.startswith("orderflow_ba_imbalance_")]
        of_spread_cols = [c for c in feature_cols if c.startswith("orderflow_spread_")]
        of_depth_cols = [c for c in feature_cols if c.startswith("orderflow_depth_imbalance_")]

        # Add labels to DataFrame
        df = features_df.copy()
        df["cluster"] = labels

        # Compute profiles for each cluster
        unique_clusters = sorted([c for c in np.unique(labels) if c >= 0])
        cluster_profiles = {}

        for cid in unique_clusters:
            mask = df["cluster"] == cid
            cluster_data = df[mask]
            n = len(cluster_data)

            profile = {
                "count": n,
                "fraction": n / len(df),
                # Volatility
                "avg_volatility": cluster_data[vol_cols].mean().mean() if vol_cols else 0,
                "volatility_detail": cluster_data[vol_cols].mean().to_dict() if vol_cols else {},
                # Returns / Momentum
                "avg_return": cluster_data[mom_cols].mean().mean() if mom_cols else 0,
                "momentum_detail": cluster_data[mom_cols].mean().to_dict() if mom_cols else {},
                # Volume profile
                "avg_volume_delta": cluster_data[vol_delta_cols].mean().mean() if vol_delta_cols else 0,
                "avg_volume_imbalance": cluster_data[vol_imb_cols].mean().mean() if vol_imb_cols else 0.5,
                "volume_detail": {
                    **cluster_data[vol_delta_cols].mean().to_dict(),
                    **cluster_data[vol_imb_cols].mean().to_dict(),
                } if vol_delta_cols else {},
                # Order flow profile
                "avg_ba_imbalance": cluster_data[of_ba_cols].mean().mean() if of_ba_cols else 0,
                "avg_spread": cluster_data[of_spread_cols].mean().mean() if of_spread_cols else 0,
                "avg_depth_imbalance": cluster_data[of_depth_cols].mean().mean() if of_depth_cols else 0,
                "orderflow_detail": {
                    **cluster_data[of_ba_cols].mean().to_dict(),
                    **cluster_data[of_spread_cols].mean().to_dict(),
                    **cluster_data[of_depth_cols].mean().to_dict(),
                } if of_ba_cols else {},
            }

            cluster_profiles[cid] = profile

            logger.info(
                "  Cluster %d: n=%d (%.1f%%), vol=%.6f, ret=%.6f, vol_delta=%.4f, ba_imb=%.4f",
                cid, n, profile["fraction"] * 100,
                profile["avg_volatility"], profile["avg_return"],
                profile["avg_volume_delta"], profile["avg_ba_imbalance"],
            )

        # Assign regime labels
        regime_map = self._assign_labels(cluster_profiles)
        self.regime_map_ = regime_map
        self.cluster_profiles_ = cluster_profiles

        logger.info("━" * 60)
        logger.info("  Regime Assignments:")
        for cid, label in regime_map.items():
            logger.info("    Cluster %d → %s", cid, label)

        # Handle noise cluster
        if -1 in np.unique(labels):
            regime_map[-1] = "Noise"
            noise_count = int((labels == -1).sum())
            logger.info("    Cluster -1 → Noise (%d points)", noise_count)

        logger.info("=" * 60)

        return regime_map, cluster_profiles

    def _assign_labels(self, profiles: dict[int, dict]) -> dict[int, str]:
        """
        Rule-based regime label assignment.

        Strategy:
            1. Compute z-scores of each cluster's metrics relative to all clusters.
            2. Score each cluster for each regime label.
            3. Assign labels greedily (no duplicates unless forced).
        """
        cluster_ids = sorted(profiles.keys())
        n = len(cluster_ids)

        if n == 0:
            return {}

        # Extract metrics as arrays
        vols = np.array([profiles[c]["avg_volatility"] for c in cluster_ids])
        rets = np.array([profiles[c]["avg_return"] for c in cluster_ids])
        vol_deltas = np.array([profiles[c]["avg_volume_delta"] for c in cluster_ids])
        ba_imb = np.array([profiles[c]["avg_ba_imbalance"] for c in cluster_ids])

        # Z-score normalize (handle zero std)
        def z(x):
            s = np.std(x)
            if s < 1e-10:
                return np.zeros_like(x)
            return (x - np.mean(x)) / s

        z_vol = z(vols)
        z_ret = z(rets)
        z_vd = z(vol_deltas)
        z_ba = z(ba_imb)

        # Score matrix: (n_clusters, n_labels)
        scores = np.zeros((n, len(REGIME_LABELS)))

        for i in range(n):
            # High Volatility: high vol, low absolute return
            scores[i, 0] = z_vol[i] * 2.0 - abs(z_ret[i]) * 0.5

            # Low Volatility: low vol, low absolute return
            scores[i, 1] = -z_vol[i] * 2.0 - abs(z_ret[i]) * 0.5

            # Trending Up: positive return, positive momentum indicators
            scores[i, 2] = z_ret[i] * 2.0 + z_ba[i] * 1.0 + z_vd[i] * 0.5

            # Trending Down: negative return, negative momentum indicators
            scores[i, 3] = -z_ret[i] * 2.0 - z_ba[i] * 1.0 - z_vd[i] * 0.5

            # Mean Reverting: moderate vol, returns close to zero, mixed signals
            scores[i, 4] = -abs(z_ret[i]) * 2.0 - abs(z_vol[i]) * 0.5

            # Transitional: high vol, moderate returns, changing vol/flow
            scores[i, 5] = abs(z_vol[i]) * 0.5 + abs(z_vd[i]) * 1.0 - abs(z_ret[i]) * 1.0

        # Greedy assignment (each label used at most once)
        assigned_labels = {}
        used_labels = set()
        used_clusters = set()

        # Sort all (cluster, label) pairs by descending score
        pairs = []
        for i, cid in enumerate(cluster_ids):
            for j in range(len(REGIME_LABELS)):
                pairs.append((scores[i, j], cid, j))
        pairs.sort(reverse=True)

        for score, cid, label_idx in pairs:
            if cid in used_clusters:
                continue
            if label_idx in used_labels and len(used_labels) < len(REGIME_LABELS):
                continue
            assigned_labels[cid] = REGIME_LABELS[label_idx]
            used_labels.add(label_idx)
            used_clusters.add(cid)

            if len(used_clusters) == n:
                break

        # If some clusters remain unassigned (more clusters than labels), assign Transitional
        for cid in cluster_ids:
            if cid not in assigned_labels:
                assigned_labels[cid] = "Transitional"

        return assigned_labels

    def generate_report(
        self,
        regime_map: dict[int, str],
        profiles: dict[int, dict],
        scores: dict[str, dict],
        best_model_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate a text-based regime report.

        Parameters
        ----------
        regime_map : dict[int, str]
            Cluster → regime label mapping.
        profiles : dict[int, dict]
            Per-cluster statistics.
        scores : dict[str, dict]
            Evaluation scores for all models.
        best_model_name : str
            Name of the selected best model.
        save_path : str, optional
            Where to save the report. Default: reports/regime_report.txt

        Returns
        -------
        str
            The report text.
        """
        if save_path is None:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            save_path = os.path.join(REPORTS_DIR, "regime_report.txt")

        lines = []
        lines.append("=" * 70)
        lines.append("  MARKET REGIME DETECTION — PHASE 3 REPORT")
        lines.append("=" * 70)
        lines.append("")

        # ── Model Evaluation Summary ─────────────────────────────────
        lines.append("─" * 70)
        lines.append("  MODEL EVALUATION SUMMARY")
        lines.append("─" * 70)
        lines.append(f"  {'Model':<12}  {'Silhouette':>12}  {'Davies-Bouldin':>14}  {'Stability':>10}  {'Clusters':>8}")
        lines.append(f"  {'─'*12}  {'─'*12}  {'─'*14}  {'─'*10}  {'─'*8}")
        for name, s in scores.items():
            marker = " ★" if name == best_model_name else ""
            lines.append(
                f"  {name:<12}  {s['silhouette']:>12.4f}  {s['davies_bouldin']:>14.4f}  "
                f"{s['stability']:>10.4f}  {s['n_clusters']:>8d}{marker}"
            )
        lines.append(f"\n  Selected Model: {best_model_name}")
        lines.append("")

        # ── Regime Profiles ──────────────────────────────────────────
        lines.append("─" * 70)
        lines.append("  REGIME PROFILES")
        lines.append("─" * 70)

        for cid in sorted(profiles.keys()):
            p = profiles[cid]
            label = regime_map.get(cid, "Unknown")
            lines.append(f"\n  Cluster {cid} — {label}")
            lines.append(f"  {'─' * 40}")
            lines.append(f"    Sample count:        {p['count']} ({p['fraction']*100:.1f}%)")
            lines.append(f"    Average Volatility:  {p['avg_volatility']:.6f}")
            lines.append(f"    Average Return:      {p['avg_return']:.6f}")
            lines.append(f"    Volume Delta (avg):  {p['avg_volume_delta']:.4f}")
            lines.append(f"    Volume Imbalance:    {p['avg_volume_imbalance']:.4f}")
            lines.append(f"    B/A Imbalance:       {p['avg_ba_imbalance']:.4f}")
            lines.append(f"    Spread (avg):        {p['avg_spread']:.6f}")
            lines.append(f"    Depth Imbalance:     {p['avg_depth_imbalance']:.4f}")

        lines.append("")
        lines.append("─" * 70)
        lines.append("  REGIME DISTRIBUTION")
        lines.append("─" * 70)

        total = sum(p["count"] for p in profiles.values())
        for cid in sorted(profiles.keys()):
            p = profiles[cid]
            label = regime_map.get(cid, "Unknown")
            bar_len = int(p["fraction"] * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            lines.append(f"  {label:<18} {bar} {p['count']:>4} ({p['fraction']*100:.1f}%)")

        lines.append("")
        lines.append("=" * 70)
        lines.append("  END OF REPORT")
        lines.append("=" * 70)

        report_text = "\n".join(lines)

        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("Saved regime report → %s", save_path)

        return report_text
