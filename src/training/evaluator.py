"""
ClusterEvaluator — Phase 3: Evaluate clustering quality.

Metrics:
    1. Silhouette Score   — measures how similar a point is to its own cluster
                            vs. neighbouring clusters. Range [-1, 1], higher = better.
    2. Davies-Bouldin Score — ratio of within-cluster to between-cluster distances.
                             Lower is better.
    3. Cluster Stability   — measured via subsampled re-clustering consistency.
                             Higher = more stable. Range [0, 1].

Public API:
    evaluator = ClusterEvaluator()
    scores = evaluator.evaluate_all(X, model_labels)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClusterEvaluator:
    """
    Evaluate clustering models using Silhouette, Davies-Bouldin, and Stability.

    Parameters
    ----------
    stability_n_runs : int
        Number of subsampling runs for stability measurement.
    stability_subsample : float
        Fraction of data to use in each stability run.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        stability_n_runs: int = 10,
        stability_subsample: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.stability_n_runs = stability_n_runs
        self.stability_subsample = stability_subsample
        self.random_state = random_state

    def evaluate_all(
        self,
        X: np.ndarray,
        model_labels: dict[str, np.ndarray],
        models: dict | None = None,
    ) -> dict[str, dict]:
        """
        Evaluate all models and return scores.

        Parameters
        ----------
        X : np.ndarray
            PCA-transformed feature matrix.
        model_labels : dict[str, np.ndarray]
            Mapping of model name → cluster labels.
        models : dict, optional
            Mapping of model name → fitted model (for stability eval).

        Returns
        -------
        dict[str, dict]
            Per-model scores: {'silhouette': float, 'davies_bouldin': float,
                               'stability': float}.
        """
        logger.info("=" * 60)
        logger.info("  ClusterEvaluator — evaluate_all()")
        logger.info("=" * 60)

        all_scores = {}

        for name, labels in model_labels.items():
            logger.info("  Evaluating: %s", name)
            scores = self._evaluate_single(X, labels, name, models)
            all_scores[name] = scores

        # Print comparison table
        logger.info("━" * 60)
        logger.info("  %-12s  %-12s  %-14s  %-10s", "Model", "Silhouette", "Davies-Bouldin", "Stability")
        logger.info("  %-12s  %-12s  %-14s  %-10s", "─" * 12, "─" * 12, "─" * 14, "─" * 10)
        for name, s in all_scores.items():
            logger.info(
                "  %-12s  %12.4f  %14.4f  %10.4f",
                name, s["silhouette"], s["davies_bouldin"], s["stability"],
            )
        logger.info("=" * 60)

        return all_scores

    def _evaluate_single(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        name: str,
        models: dict | None = None,
    ) -> dict:
        """Evaluate a single set of labels."""
        scores = {}

        # Filter out noise labels (-1) for metrics that require at least 2 clusters
        valid_mask = labels >= 0
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]

        n_clusters = len(np.unique(labels_valid))
        n_noise = int((~valid_mask).sum())

        if n_noise > 0:
            logger.info("    Noise points excluded: %d", n_noise)

        # ── Silhouette Score ─────────────────────────────────────────
        if n_clusters >= 2 and len(X_valid) > n_clusters:
            sil = silhouette_score(X_valid, labels_valid)
            logger.info("    Silhouette Score: %.4f", sil)
        else:
            sil = -1.0
            logger.warning("    Silhouette Score: N/A (need ≥2 clusters)")
        scores["silhouette"] = sil

        # ── Davies-Bouldin Score ─────────────────────────────────────
        if n_clusters >= 2 and len(X_valid) > n_clusters:
            db = davies_bouldin_score(X_valid, labels_valid)
            logger.info("    Davies-Bouldin Score: %.4f", db)
        else:
            db = float("inf")
            logger.warning("    Davies-Bouldin Score: N/A (need ≥2 clusters)")
        scores["davies_bouldin"] = db

        # ── Cluster Stability ────────────────────────────────────────
        stability = self._compute_stability(X, labels, name, models)
        scores["stability"] = stability
        logger.info("    Cluster Stability: %.4f", stability)

        scores["n_clusters"] = n_clusters
        scores["n_noise"] = n_noise

        return scores

    def _compute_stability(
        self,
        X: np.ndarray,
        original_labels: np.ndarray,
        name: str,
        models: dict | None = None,
    ) -> float:
        """
        Measure cluster stability via subsampled re-clustering.

        For each run:
            1. Sample `stability_subsample` fraction of the data.
            2. Re-fit a new model of the same type.
            3. Compute Adjusted Rand Index (ARI) between original and new labels.
            4. Average ARI across all runs.
        """
        from sklearn.metrics import adjusted_rand_score
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture

        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        subsample_size = int(n_samples * self.stability_subsample)

        ari_scores = []

        for run in range(self.stability_n_runs):
            indices = rng.choice(n_samples, size=subsample_size, replace=False)
            X_sub = X[indices]
            original_sub = original_labels[indices]

            # Skip runs where original has < 2 unique non-noise labels
            valid_original = original_sub[original_sub >= 0]
            if len(np.unique(valid_original)) < 2:
                continue

            try:
                if name == "KMeans":
                    n_clusters = len(np.unique(original_labels[original_labels >= 0]))
                    sub_model = KMeans(
                        n_clusters=n_clusters,
                        random_state=self.random_state + run,
                        n_init=5,
                    )
                    sub_labels = sub_model.fit_predict(X_sub)

                elif name == "GMM":
                    n_clusters = len(np.unique(original_labels[original_labels >= 0]))
                    sub_model = GaussianMixture(
                        n_components=n_clusters,
                        covariance_type="full",
                        random_state=self.random_state + run,
                        n_init=2,
                    )
                    sub_model.fit(X_sub)
                    sub_labels = sub_model.predict(X_sub)

                elif name == "HDBSCAN":
                    try:
                        import hdbscan as hdbscan_lib
                        sub_model = hdbscan_lib.HDBSCAN(
                            min_cluster_size=max(10, len(X_sub) // 50),
                            min_samples=5,
                        )
                    except ImportError:
                        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
                        sub_model = SklearnHDBSCAN(
                            min_cluster_size=max(10, len(X_sub) // 50),
                            min_samples=5,
                        )
                    sub_labels = sub_model.fit_predict(X_sub)
                else:
                    continue

                # Compute ARI (handles noise by treating -1 as its own cluster)
                ari = adjusted_rand_score(original_sub, sub_labels)
                ari_scores.append(ari)

            except Exception:
                continue

        if not ari_scores:
            return 0.0

        return float(np.mean(ari_scores))
