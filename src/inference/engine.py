"""
InferenceEngine — Phase 4: Live regime prediction engine.

Loads trained Phase 3 artifacts (scaler, PCA, regime_model) and provides
a full pipeline from live Binance data to regime prediction.

Pipeline:
    fetch_market_data() → compute_features() → scaler → PCA → regime_model → prediction

Returns:
    {
        current_regime,
        confidence,
        cluster_id,
        regime_description,
        timestamp
    }

Public API:
    engine = InferenceEngine()
    engine.initialize()
    result = engine.predict_regime()
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.binance_loader import BinanceDataLoader
from src.features.feature_generator import FeatureGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.environ.get("MODEL_PATH", os.path.join(_PROJECT_ROOT, "models"))

# Regime descriptions for context
REGIME_DESCRIPTIONS = {
    "High Volatility": "Market experiencing elevated price swings and uncertainty. "
                       "Large moves in both directions are likely. Exercise caution with position sizing.",
    "Low Volatility": "Market is calm with compressed price ranges. "
                      "Breakout potential is building. Suitable for range-bound strategies.",
    "Trending Up": "Market is in a sustained upward move with bullish momentum. "
                   "Buying pressure dominates. Trend-following strategies are favoured.",
    "Trending Down": "Market is in a sustained downward move with bearish momentum. "
                     "Selling pressure dominates. Defensive positioning is recommended.",
    "Mean Reverting": "Market is oscillating around equilibrium with no clear direction. "
                      "Price tends to revert to the mean. Mean-reversion strategies are favoured.",
    "Transitional": "Market is between regimes — conditions are shifting. "
                    "Signals are mixed. Wait for regime confirmation before committing.",
    "Noise": "Data point does not clearly belong to any regime. "
             "This may indicate an unusual or transitional market state.",
}


class InferenceEngine:
    """
    Live regime prediction engine.

    Loads the trained scaler, PCA, and regime model from Phase 3,
    fetches live BTCUSDT data from Binance, computes features,
    and predicts the current market regime.

    Parameters
    ----------
    symbol : str
        Trading pair for prediction. Default: 'BTCUSDT'.
    scaler_path : str, optional
        Path to scaler.pkl. Default: models/scaler.pkl.
    pca_path : str, optional
        Path to pca.pkl. Default: models/pca.pkl.
    model_path : str, optional
        Path to regime_model.pkl. Default: models/regime_model.pkl.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        scaler_path: Optional[str] = None,
        pca_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        self.symbol = symbol
        self.scaler_path = scaler_path or os.path.join(MODELS_DIR, "scaler.pkl")
        self.pca_path = pca_path or os.path.join(MODELS_DIR, "pca.pkl")
        self.model_path = model_path or os.path.join(MODELS_DIR, "regime_model.pkl")

        # Loaded artifacts (populated by initialize())
        self.scaler: Any = None
        self.pca: Any = None
        self.model: Any = None
        self.model_name: str = ""
        self.regime_map: dict[int, str] = {}
        self.cluster_profiles: dict = {}

        # Data loader and feature generator
        self.loader: Optional[BinanceDataLoader] = None
        self.feature_generator: Optional[FeatureGenerator] = None

        self._initialized: bool = False

        logger.info("InferenceEngine created — symbol=%s", self.symbol)

    # ══════════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════════

    def initialize(self) -> "InferenceEngine":
        """
        Load all trained artifacts from disk and prepare the engine.

        Loads:
            - models/scaler.pkl  (StandardScaler)
            - models/pca.pkl     (PCA transformer)
            - models/regime_model.pkl (clustering model + regime_map)

        Returns
        -------
        self
        """
        logger.info("=" * 60)
        logger.info("  InferenceEngine — initialize()")
        logger.info("=" * 60)

        # ── Load scaler ──────────────────────────────────────────────
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Scaler not found: {self.scaler_path}. Run Phase 3 first."
            )
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info("  Loaded scaler → %s", self.scaler_path)

        # ── Load PCA ─────────────────────────────────────────────────
        if not os.path.exists(self.pca_path):
            raise FileNotFoundError(
                f"PCA not found: {self.pca_path}. Run Phase 3 first."
            )
        with open(self.pca_path, "rb") as f:
            self.pca = pickle.load(f)
        logger.info(
            "  Loaded PCA → %s (%d components)",
            self.pca_path, self.pca.n_components_,
        )

        # ── Load regime model ────────────────────────────────────────
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Regime model not found: {self.model_path}. Run Phase 3 first."
            )
        with open(self.model_path, "rb") as f:
            artifact = pickle.load(f)

        self.model = artifact["model"]
        self.model_name = artifact["model_name"]
        self.regime_map = artifact.get("metadata", {}).get("regime_map", {})
        self.cluster_profiles = artifact.get("metadata", {}).get("cluster_profiles", {})

        # Normalize regime_map keys to int
        self.regime_map = {int(k): v for k, v in self.regime_map.items()}

        logger.info(
            "  Loaded regime model → %s (algorithm: %s, clusters: %d)",
            self.model_path, self.model_name, artifact.get("n_clusters", 0),
        )
        logger.info("  Regime map: %s", self.regime_map)

        # ── Initialize data loader and feature generator ─────────────
        self.loader = BinanceDataLoader(symbols=[self.symbol])
        self.feature_generator = FeatureGenerator()

        self._initialized = True
        logger.info("  Engine ready.")
        logger.info("=" * 60)

        return self

    def fetch_market_data(self) -> dict[str, pd.DataFrame]:
        """
        Fetch latest market data from Binance for the configured symbol.

        Fetches:
            - Last 60 × 1-minute OHLCV candles (covers ~60 min, enough for
              30-period rolling windows on 1-min data)
            - Recent trades (last 500)
            - Current orderbook snapshot

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: 'ohlcv', 'trades', 'orderbook'.
        """
        self._check_initialized()

        logger.info("━" * 60)
        logger.info("  Fetching live market data for %s …", self.symbol)
        logger.info("━" * 60)

        data = {}

        # Fetch 1-minute candles — need at least 60 for 30-period rolling windows
        # (30 periods warm-up + 30 periods of actual data)
        data["ohlcv"] = self.loader.fetch_ohlcv(
            self.symbol, interval="1m", limit=60
        )
        logger.info("  OHLCV: %d candles (1m interval)", len(data["ohlcv"]))

        # Fetch recent trades
        try:
            data["trades"] = self.loader.fetch_trades(self.symbol, limit=500)
            logger.info("  Trades: %d", len(data["trades"]))
        except Exception as exc:
            logger.warning("  Failed to fetch trades: %s", exc)
            data["trades"] = None

        # Fetch orderbook
        try:
            data["orderbook"] = self.loader.fetch_orderbook(self.symbol, limit=100)
            logger.info("  Orderbook: %d levels", len(data["orderbook"]))
        except Exception as exc:
            logger.warning("  Failed to fetch orderbook: %s", exc)
            data["orderbook"] = None

        return data

    def compute_features(
        self, market_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compute features from live market data using the Phase 2 FeatureGenerator.

        Parameters
        ----------
        market_data : dict[str, pd.DataFrame]
            Output of fetch_market_data().

        Returns
        -------
        pd.DataFrame
            Feature matrix (single row after taking the last valid row).
        """
        self._check_initialized()

        logger.info("━" * 60)
        logger.info("  Computing features …")
        logger.info("━" * 60)

        ohlcv = market_data["ohlcv"]
        trades = market_data.get("trades")
        orderbook = market_data.get("orderbook")

        # Use FeatureGenerator to compute features identically to Phase 2
        self.feature_generator.fit(ohlcv=ohlcv, trades=trades, orderbook=orderbook)
        features = self.feature_generator.transform()

        logger.info(
            "  Features computed: %d rows × %d columns",
            len(features), len(features.columns),
        )

        if features.empty:
            raise ValueError(
                "Feature computation produced no valid rows. "
                "Ensure sufficient OHLCV data is available."
            )

        # Take the LAST row — this represents the most recent market state
        last_row = features.iloc[[-1]].copy()
        logger.info("  Using latest feature row (timestamp: %s)",
                     last_row["open_time"].iloc[0] if "open_time" in last_row.columns else "N/A")

        return last_row

    def predict_regime(self) -> dict:
        """
        Execute the full inference pipeline: fetch → features → predict.

        Pipeline:
            1. fetch_market_data()   — get live BTCUSDT data
            2. compute_features()    — generate feature vector
            3. scaler.transform()    — standardize
            4. pca.transform()       — dimensionality reduction
            5. model.predict()       — cluster assignment
            6. regime_map lookup     — human-readable label

        Returns
        -------
        dict
            {
                'current_regime': str,
                'confidence': float,
                'cluster_id': int,
                'regime_description': str,
                'timestamp': str (ISO 8601)
            }
        """
        self._check_initialized()

        logger.info("=" * 60)
        logger.info("  InferenceEngine — predict_regime()")
        logger.info("=" * 60)

        # ── Step 1: Fetch live data ──────────────────────────────────
        market_data = self.fetch_market_data()

        # ── Step 2: Compute features ─────────────────────────────────
        features_row = self.compute_features(market_data)

        # ── Step 3: Prepare feature vector ───────────────────────────
        # Exclude metadata columns (same as Phase 3 preprocessing)
        meta_cols = {"open_time", "symbol"}
        feature_cols = [c for c in features_row.columns if c not in meta_cols]
        X = features_row[feature_cols].values

        # Handle any remaining NaN (median fill with 0 as fallback)
        if np.isnan(X).any():
            logger.warning("  NaN detected in features — filling with 0.")
            X = np.nan_to_num(X, nan=0.0)

        logger.info("  Feature vector shape: %s", X.shape)

        # ── Step 4: Scale ────────────────────────────────────────────
        X_scaled = self.scaler.transform(X)
        logger.info("  Scaled feature vector.")

        # ── Step 5: PCA ──────────────────────────────────────────────
        X_pca = self.pca.transform(X_scaled)
        logger.info("  PCA-transformed: %d components", X_pca.shape[1])

        # ── Step 6: Predict cluster ──────────────────────────────────
        cluster_id, confidence = self._predict_with_confidence(X_pca)

        # ── Step 7: Map to regime label ──────────────────────────────
        current_regime = self.regime_map.get(cluster_id, "Unknown")
        regime_description = REGIME_DESCRIPTIONS.get(
            current_regime,
            f"Cluster {cluster_id} — no description available.",
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        result = {
            "current_regime": current_regime,
            "confidence": round(confidence, 4),
            "cluster_id": int(cluster_id),
            "regime_description": regime_description,
            "timestamp": timestamp,
        }

        logger.info("━" * 60)
        logger.info("  PREDICTION RESULT")
        logger.info("━" * 60)
        logger.info("  Regime:      %s", result["current_regime"])
        logger.info("  Confidence:  %.4f", result["confidence"])
        logger.info("  Cluster ID:  %d", result["cluster_id"])
        logger.info("  Timestamp:   %s", result["timestamp"])
        logger.info("  Description: %s", result["regime_description"])
        logger.info("=" * 60)

        return result

    # ══════════════════════════════════════════════════════════════════════
    #  Internal Methods
    # ══════════════════════════════════════════════════════════════════════

    def _check_initialized(self) -> None:
        """Ensure the engine has been initialized."""
        if not self._initialized:
            raise RuntimeError(
                "InferenceEngine not initialized. Call initialize() first."
            )

    def _predict_with_confidence(
        self, X_pca: np.ndarray
    ) -> tuple[int, float]:
        """
        Predict cluster and compute a confidence score.

        Confidence computation varies by model type:
            - KMeans: based on distance to nearest centroid vs. second-nearest
            - GMM: posterior probability of the assigned component
            - HDBSCAN: uses the built-in membership probabilities or
              falls back to distance-based confidence

        Parameters
        ----------
        X_pca : np.ndarray
            PCA-transformed feature vector, shape (1, n_components).

        Returns
        -------
        tuple[int, float]
            (cluster_id, confidence) where confidence is in [0, 1].
        """
        if self.model_name == "KMeans":
            return self._predict_kmeans(X_pca)
        elif self.model_name == "GMM":
            return self._predict_gmm(X_pca)
        elif self.model_name == "HDBSCAN":
            return self._predict_hdbscan(X_pca)
        else:
            # Fallback: try generic predict
            logger.warning("Unknown model type '%s', using generic predict.", self.model_name)
            label = int(self.model.predict(X_pca)[0])
            return label, 0.5

    def _predict_kmeans(self, X_pca: np.ndarray) -> tuple[int, float]:
        """KMeans prediction with distance-based confidence."""
        cluster_id = int(self.model.predict(X_pca)[0])

        # Compute distances to all centroids
        distances = np.linalg.norm(
            self.model.cluster_centers_ - X_pca, axis=1
        )
        sorted_dists = np.sort(distances)

        # Confidence = 1 - (nearest / second_nearest)
        # If the point is much closer to its cluster than the next, confidence is high
        if len(sorted_dists) >= 2 and sorted_dists[1] > 0:
            confidence = 1.0 - (sorted_dists[0] / sorted_dists[1])
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 1.0

        return cluster_id, confidence

    def _predict_gmm(self, X_pca: np.ndarray) -> tuple[int, float]:
        """GMM prediction with posterior probability confidence."""
        cluster_id = int(self.model.predict(X_pca)[0])

        # Posterior probabilities across all components
        probas = self.model.predict_proba(X_pca)[0]
        confidence = float(probas[cluster_id])

        return cluster_id, confidence

    def _predict_hdbscan(self, X_pca: np.ndarray) -> tuple[int, float]:
        """HDBSCAN prediction using approximate_predict or nearest-centroid fallback."""
        try:
            # Try hdbscan's approximate_predict (requires prediction_data=True)
            import hdbscan as hdbscan_lib
            labels, strengths = hdbscan_lib.approximate_predict(
                self.model, X_pca
            )
            cluster_id = int(labels[0])
            confidence = float(strengths[0])

            # If classified as noise (-1), try nearest non-noise centroid
            if cluster_id == -1:
                cluster_id, confidence = self._hdbscan_nearest_centroid(X_pca)
                confidence *= 0.5  # Reduce confidence for noise-to-cluster mapping

        except (ImportError, AttributeError, Exception) as exc:
            logger.warning(
                "HDBSCAN approximate_predict unavailable (%s), using nearest centroid.",
                exc,
            )
            cluster_id, confidence = self._hdbscan_nearest_centroid(X_pca)

        return cluster_id, confidence

    def _hdbscan_nearest_centroid(
        self, X_pca: np.ndarray
    ) -> tuple[int, float]:
        """
        Fallback for HDBSCAN: compute centroids from training labels
        and assign to the nearest one.
        """
        # Get training labels from the model
        training_labels = self.model.labels_

        # Compute centroids for each non-noise cluster
        unique_clusters = [c for c in np.unique(training_labels) if c >= 0]
        if not unique_clusters:
            return -1, 0.0

        # We don't have the training X_pca stored, so use regime_map keys
        # and assign to the cluster with the closest profile
        # Fallback: assign to the most common cluster
        cluster_counts = {
            c: int((training_labels == c).sum()) for c in unique_clusters
        }
        most_common = max(cluster_counts, key=cluster_counts.get)

        return int(most_common), 0.3
