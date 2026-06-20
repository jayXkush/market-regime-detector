"""
ClusteringTrainer — Phase 3: Train and compare clustering models.

Trains three clustering algorithms on PCA-transformed data:
    1. KMeans
    2. Gaussian Mixture Model (GMM)
    3. HDBSCAN

Each model is evaluated and the best one is selected automatically
based on composite scoring of evaluation metrics.

Public API:
    trainer = ClusteringTrainer()
    results = trainer.fit_all(X_pca)
    best_name, best_model, best_labels = trainer.select_best()
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


class ClusteringTrainer:
    """
    Train and compare KMeans, GMM, and HDBSCAN on PCA-transformed features.

    Parameters
    ----------
    n_clusters : int
        Number of clusters for KMeans and GMM. Default: 6 (matches 6 regime labels).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 6,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.models: dict[str, Any] = {}
        self.labels: dict[str, np.ndarray] = {}
        self.fitted_: bool = False

        logger.info(
            "ClusteringTrainer initialised — n_clusters=%d, seed=%d",
            self.n_clusters, self.random_state,
        )

    def fit_all(self, X: np.ndarray) -> dict[str, dict]:
        """
        Train all three clustering models.

        Parameters
        ----------
        X : np.ndarray
            PCA-transformed feature matrix, shape (n_samples, n_components).

        Returns
        -------
        dict[str, dict]
            Per-model info: {'model': obj, 'labels': ndarray, 'n_clusters': int}.
        """
        logger.info("=" * 60)
        logger.info("  ClusteringTrainer — fit_all()")
        logger.info("=" * 60)
        logger.info("  Input shape: %s", X.shape)

        results = {}

        # ── 1. KMeans ────────────────────────────────────────────────
        results["KMeans"] = self._fit_kmeans(X)

        # ── 2. Gaussian Mixture Model ────────────────────────────────
        results["GMM"] = self._fit_gmm(X)

        # ── 3. HDBSCAN ──────────────────────────────────────────────
        results["HDBSCAN"] = self._fit_hdbscan(X)

        self.fitted_ = True
        logger.info("=" * 60)
        return results

    def _fit_kmeans(self, X: np.ndarray) -> dict:
        """Train KMeans clustering."""
        logger.info("  Training KMeans (k=%d) …", self.n_clusters)

        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        labels = model.fit_predict(X)

        unique_labels = np.unique(labels)
        cluster_sizes = {int(l): int((labels == l).sum()) for l in unique_labels}

        self.models["KMeans"] = model
        self.labels["KMeans"] = labels

        logger.info("    Converged in %d iterations.", model.n_iter_)
        logger.info("    Inertia: %.4f", model.inertia_)
        logger.info("    Cluster sizes: %s", cluster_sizes)

        return {
            "model": model,
            "labels": labels,
            "n_clusters": len(unique_labels),
        }

    def _fit_gmm(self, X: np.ndarray) -> dict:
        """Train Gaussian Mixture Model."""
        logger.info("  Training GMM (n_components=%d) …", self.n_clusters)

        model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",
            random_state=self.random_state,
            n_init=3,
            max_iter=200,
        )
        model.fit(X)
        labels = model.predict(X)

        unique_labels = np.unique(labels)
        cluster_sizes = {int(l): int((labels == l).sum()) for l in unique_labels}

        self.models["GMM"] = model
        self.labels["GMM"] = labels

        logger.info("    Converged: %s", model.converged_)
        logger.info("    BIC: %.4f", model.bic(X))
        logger.info("    AIC: %.4f", model.aic(X))
        logger.info("    Cluster sizes: %s", cluster_sizes)

        return {
            "model": model,
            "labels": labels,
            "n_clusters": len(unique_labels),
        }

    def _fit_hdbscan(self, X: np.ndarray) -> dict:
        """Train HDBSCAN clustering."""
        logger.info("  Training HDBSCAN …")

        try:
            import hdbscan as hdbscan_lib

            model = hdbscan_lib.HDBSCAN(
                min_cluster_size=max(10, len(X) // 50),
                min_samples=5,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            labels = model.fit_predict(X)

        except ImportError:
            logger.warning("    hdbscan package not available, using sklearn HDBSCAN.")
            from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

            model = SklearnHDBSCAN(
                min_cluster_size=max(10, len(X) // 50),
                min_samples=5,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = model.fit_predict(X)

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        noise_count = int((labels == -1).sum())
        cluster_sizes = {int(l): int((labels == l).sum()) for l in unique_labels}

        self.models["HDBSCAN"] = model
        self.labels["HDBSCAN"] = labels

        logger.info("    Clusters found: %d", n_clusters)
        logger.info("    Noise points: %d (%.1f%%)", noise_count, 100 * noise_count / len(labels))
        logger.info("    Cluster sizes: %s", cluster_sizes)

        return {
            "model": model,
            "labels": labels,
            "n_clusters": n_clusters,
        }

    def select_best(self, scores: dict[str, dict]) -> tuple[str, Any, np.ndarray]:
        """
        Select the best model based on evaluation scores.

        Selection criteria (composite score):
            - Higher silhouette is better (weight: 0.4)
            - Lower Davies-Bouldin is better (weight: 0.3)
            - Higher stability is better (weight: 0.3)

        HDBSCAN is penalized if it has too many noise points (>30%).

        Parameters
        ----------
        scores : dict[str, dict]
            Evaluation scores from ClusterEvaluator.evaluate_all().

        Returns
        -------
        tuple[str, model, labels]
            Name, model object, and cluster labels for the best model.
        """
        logger.info("━" * 60)
        logger.info("  Model Selection")
        logger.info("━" * 60)

        composite_scores = {}

        for name, s in scores.items():
            sil = s.get("silhouette", -1)
            db = s.get("davies_bouldin", float("inf"))
            stab = s.get("stability", 0)

            # Normalize Davies-Bouldin (lower is better, invert it)
            # Use 1 / (1 + db) to map to [0, 1]
            db_norm = 1.0 / (1.0 + db) if db != float("inf") else 0

            # Composite score
            composite = 0.4 * sil + 0.3 * db_norm + 0.3 * stab

            # Penalize HDBSCAN for excessive noise
            if name == "HDBSCAN":
                labels = self.labels[name]
                noise_ratio = (labels == -1).sum() / len(labels)
                if noise_ratio > 0.30:
                    composite *= (1 - noise_ratio)
                    logger.info(
                        "    %s: penalized for %.1f%% noise",
                        name, noise_ratio * 100,
                    )

            composite_scores[name] = composite
            logger.info(
                "    %s: silhouette=%.4f, DB=%.4f, stability=%.4f → composite=%.4f",
                name, sil, db, stab, composite,
            )

        best_name = max(composite_scores, key=composite_scores.get)
        best_model = self.models[best_name]
        best_labels = self.labels[best_name]

        logger.info("  ✓ Best model: %s (composite=%.4f)", best_name, composite_scores[best_name])
        logger.info("━" * 60)

        return best_name, best_model, best_labels

    def save_model(
        self,
        model: Any,
        model_name: str,
        labels: np.ndarray,
        metadata: Optional[dict] = None,
        path: Optional[str] = None,
    ) -> str:
        """
        Save the selected model to disk as regime_model.pkl.

        The saved object includes:
            - model: the fitted clustering model
            - model_name: name of the algorithm
            - labels: cluster assignments
            - metadata: any additional info

        Parameters
        ----------
        model : Any
            The fitted clustering model.
        model_name : str
            Name of the algorithm (e.g. 'KMeans').
        labels : np.ndarray
            Cluster labels.
        metadata : dict, optional
            Additional metadata to save.
        path : str, optional
            Save path. Default: models/regime_model.pkl.

        Returns
        -------
        str
            Path to saved model file.
        """
        save_path = path or os.path.join(MODELS_DIR, "regime_model.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        artifact = {
            "model": model,
            "model_name": model_name,
            "labels": labels,
            "n_clusters": len(np.unique(labels[labels >= 0])),
            "metadata": metadata or {},
        }

        with open(save_path, "wb") as f:
            pickle.dump(artifact, f)

        logger.info("Saved regime model → %s", save_path)
        return save_path
