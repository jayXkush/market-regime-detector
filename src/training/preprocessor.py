"""
Preprocessor — Phase 3: Missing value handling, StandardScaler, and PCA.

Handles the full preprocessing pipeline:
    1. Missing value imputation (median fill + drop rows with > 50% NaN)
    2. StandardScaler (zero mean, unit variance)
    3. PCA (dimensionality reduction preserving 95% variance)

Saves:
    - models/scaler.pkl
    - models/pca.pkl

Public API:
    preprocessor = Preprocessor()
    X_pca, feature_names = preprocessor.fit_transform(df)
    preprocessor.save()
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# Columns to exclude from feature matrix (metadata, not features)
_META_COLS = {"open_time", "symbol"}


class Preprocessor:
    """
    Preprocessing pipeline: missing values → StandardScaler → PCA.

    Parameters
    ----------
    pca_variance : float
        Fraction of variance to retain in PCA. Default: 0.95 (95%).
    scaler_path : str, optional
        Where to save the fitted scaler. Default: models/scaler.pkl.
    pca_path : str, optional
        Where to save the fitted PCA. Default: models/pca.pkl.
    """

    def __init__(
        self,
        pca_variance: float = 0.95,
        scaler_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ) -> None:
        self.pca_variance = pca_variance
        self.scaler_path = scaler_path or os.path.join(MODELS_DIR, "scaler.pkl")
        self.pca_path = pca_path or os.path.join(MODELS_DIR, "pca.pkl")

        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.feature_columns_: list[str] = []
        self.pca_columns_: list[str] = []
        self.meta_df_: Optional[pd.DataFrame] = None

        logger.info(
            "Preprocessor initialised — pca_variance=%.2f", self.pca_variance
        )

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw feature DataFrame from Phase 2 (data/features/features.parquet).

        Returns
        -------
        X_pca : np.ndarray
            PCA-transformed feature matrix, shape (n_samples, n_components).
        pca_columns : list[str]
            Names of the PCA components (e.g. ['PC1', 'PC2', ...]).
        """
        logger.info("=" * 60)
        logger.info("  Preprocessor — fit_transform()")
        logger.info("=" * 60)

        # ── Step 1: Separate metadata from features ──────────────────
        meta_cols = [c for c in df.columns if c in _META_COLS]
        self.feature_columns_ = [c for c in df.columns if c not in _META_COLS]
        self.meta_df_ = df[meta_cols].copy() if meta_cols else pd.DataFrame()

        X = df[self.feature_columns_].copy()
        logger.info("  Input shape: %s", X.shape)
        logger.info("  Feature columns: %d", len(self.feature_columns_))

        # ── Step 2: Missing value handling ───────────────────────────
        X = self._handle_missing(X)

        # Also trim metadata to match
        if self.meta_df_ is not None and len(self.meta_df_) > len(X):
            self.meta_df_ = self.meta_df_.iloc[X.index].reset_index(drop=True)
            X = X.reset_index(drop=True)

        # ── Step 3: StandardScaler ───────────────────────────────────
        X_scaled = self._fit_scaler(X)

        # ── Step 4: PCA ──────────────────────────────────────────────
        X_pca = self._fit_pca(X_scaled)

        self.pca_columns_ = [f"PC{i+1}" for i in range(X_pca.shape[1])]

        logger.info("  Final output shape: %s", X_pca.shape)
        logger.info("=" * 60)

        return X_pca, self.pca_columns_

    def _handle_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values: drop rows with >50% NaN, median-fill rest."""
        initial_rows = len(X)
        nan_counts = X.isnull().sum()
        total_nan = nan_counts.sum()

        logger.info("  Step 1: Missing value handling")
        logger.info("    Total NaN values: %d", total_nan)

        if total_nan == 0:
            logger.info("    No missing values found — skipping imputation.")
            return X

        # Drop rows where more than 50% of features are NaN
        threshold = len(X.columns) * 0.5
        rows_before = len(X)
        X = X.dropna(thresh=int(threshold))
        rows_dropped = rows_before - len(X)
        if rows_dropped > 0:
            logger.info(
                "    Dropped %d rows with >50%% NaN values.", rows_dropped
            )

        # Median-fill remaining NaN values
        remaining_nan = X.isnull().sum().sum()
        if remaining_nan > 0:
            medians = X.median()
            X = X.fillna(medians)
            logger.info(
                "    Median-filled %d remaining NaN values.", remaining_nan
            )

        logger.info(
            "    Rows: %d → %d (dropped %d)",
            initial_rows, len(X), initial_rows - len(X),
        )
        return X

    def _fit_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """Fit StandardScaler and transform features."""
        logger.info("  Step 2: StandardScaler")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.values)

        logger.info(
            "    Scaled %d features to zero mean / unit variance.", X.shape[1]
        )
        logger.info(
            "    Mean range: [%.6f, %.6f]",
            X_scaled.mean(axis=0).min(),
            X_scaled.mean(axis=0).max(),
        )
        logger.info(
            "    Std range:  [%.6f, %.6f]",
            X_scaled.std(axis=0).min(),
            X_scaled.std(axis=0).max(),
        )

        return X_scaled

    def _fit_pca(self, X_scaled: np.ndarray) -> np.ndarray:
        """Fit PCA to retain specified variance and transform."""
        logger.info("  Step 3: PCA (%.0f%% variance retention)", self.pca_variance * 100)

        self.pca = PCA(n_components=self.pca_variance, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        n_components = self.pca.n_components_
        explained = self.pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        logger.info("    Components selected: %d (of %d original features)", n_components, X_scaled.shape[1])
        logger.info("    Total variance explained: %.4f", cumulative[-1])
        logger.info("    Per-component variance:")
        for i, (ev, cum) in enumerate(zip(explained, cumulative)):
            logger.info("      PC%d: %.4f (cumulative: %.4f)", i + 1, ev, cum)

        return X_pca

    def save(self) -> tuple[str, str]:
        """
        Persist scaler and PCA objects to disk.

        Returns
        -------
        tuple[str, str]
            Paths to (scaler.pkl, pca.pkl).
        """
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.pca_path), exist_ok=True)

        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info("Saved scaler → %s", self.scaler_path)

        with open(self.pca_path, "wb") as f:
            pickle.dump(self.pca, f)
        logger.info("Saved PCA → %s", self.pca_path)

        return self.scaler_path, self.pca_path

    def get_metadata(self) -> pd.DataFrame:
        """Return the metadata columns (open_time, symbol) aligned with processed data."""
        return self.meta_df_.copy() if self.meta_df_ is not None else pd.DataFrame()
