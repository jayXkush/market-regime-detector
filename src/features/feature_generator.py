"""
FeatureGenerator — Phase 2: Feature Engineering Pipeline.

Builds a comprehensive feature matrix from raw OHLCV, trades, and orderbook
data produced by Phase 1 (data/raw/).

Feature categories:
    • Volatility   — rolling volatility, ATR-style volatility, realized variance
    • Momentum     — returns, cumulative returns, price acceleration
    • Volume       — volume delta, volume imbalance
    • Order Flow   — bid/ask imbalance, spread, depth imbalance

Each feature is computed at three rolling-window sizes: 5, 15, and 30 periods.
The base time resolution is whatever the OHLCV candle interval is (typically 1h).

Public API:
    generator = FeatureGenerator()
    generator.fit(ohlcv_df, trades_df, orderbook_df)
    features_df = generator.transform()
    generator.save("path/to/config.json")
    generator = FeatureGenerator.load("path/to/config.json")
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
WINDOWS = [5, 15, 30]

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "features")
DEFAULT_CONFIG_DIR = os.path.join(_PROJECT_ROOT, "models", "feature_config")


class FeatureGenerator:
    """
    Generate market-regime features from raw OHLCV, trades, and orderbook data.

    Parameters
    ----------
    windows : list[int], optional
        Rolling window sizes in number of periods.  Defaults to [5, 15, 30].

    Attributes
    ----------
    feature_names_ : list[str]
        Names of all generated features (set after ``fit`` is called).
    fitted_ : bool
        Whether the generator has been fitted.
    """

    def __init__(self, windows: Optional[list[int]] = None) -> None:
        self.windows = windows or list(WINDOWS)
        self.feature_names_: list[str] = []
        self.fitted_: bool = False

        # Internal state populated by fit()
        self._ohlcv: Optional[pd.DataFrame] = None
        self._trades: Optional[pd.DataFrame] = None
        self._orderbook: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None

        logger.info(
            "FeatureGenerator initialised — windows=%s", self.windows
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════════

    def fit(
        self,
        ohlcv: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
        orderbook: Optional[pd.DataFrame] = None,
    ) -> "FeatureGenerator":
        """
        Ingest raw data and prepare internal state for feature computation.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            OHLCV candlestick data with columns:
            open_time, open, high, low, close, volume, quote_volume,
            trades, taker_buy_base_vol, taker_buy_quote_vol.
        trades : pd.DataFrame, optional
            Recent trades with columns: price, qty, time, isBuyerMaker.
        orderbook : pd.DataFrame, optional
            Order book snapshot with columns: side, price, quantity.

        Returns
        -------
        self
        """
        logger.info("fit() — ingesting raw data …")

        # Validate and store OHLCV
        self._ohlcv = self._prepare_ohlcv(ohlcv.copy())
        logger.info("  OHLCV: %d rows", len(self._ohlcv))

        # Optional data sources
        if trades is not None and not trades.empty:
            self._trades = trades.copy()
            logger.info("  Trades: %d rows", len(self._trades))
        else:
            self._trades = None
            logger.info("  Trades: not provided")

        if orderbook is not None and not orderbook.empty:
            self._orderbook = orderbook.copy()
            logger.info("  Orderbook: %d rows", len(self._orderbook))
        else:
            self._orderbook = None
            logger.info("  Orderbook: not provided")

        self.fitted_ = True
        logger.info("fit() — complete")
        return self

    def transform(self) -> pd.DataFrame:
        """
        Compute all features and return the feature DataFrame.

        Must be called after ``fit()``.

        Returns
        -------
        pd.DataFrame
            Feature matrix indexed by ``open_time``.
        """
        if not self.fitted_:
            raise RuntimeError("Must call fit() before transform().")

        logger.info("transform() — computing features …")
        ohlcv = self._ohlcv.copy()

        # Start with timestamp index
        features = pd.DataFrame(index=ohlcv.index)
        features["open_time"] = ohlcv["open_time"]

        # ── Volatility features ───────────────────────────────────────
        logger.info("  Computing volatility features …")
        vol_features = self._compute_volatility(ohlcv)
        features = pd.concat([features, vol_features], axis=1)

        # ── Momentum features ─────────────────────────────────────────
        logger.info("  Computing momentum features …")
        mom_features = self._compute_momentum(ohlcv)
        features = pd.concat([features, mom_features], axis=1)

        # ── Volume features ───────────────────────────────────────────
        logger.info("  Computing volume features …")
        volume_features = self._compute_volume(ohlcv)
        features = pd.concat([features, volume_features], axis=1)

        # ── Order flow features ───────────────────────────────────────
        logger.info("  Computing order flow features …")
        orderflow_features = self._compute_order_flow(ohlcv)
        features = pd.concat([features, orderflow_features], axis=1)

        # Drop rows with NaN from rolling windows
        initial_rows = len(features)
        features = features.dropna().reset_index(drop=True)
        dropped = initial_rows - len(features)
        logger.info(
            "  Dropped %d rows with NaN (from rolling warm-up). %d rows remain.",
            dropped, len(features),
        )

        # Store feature names (exclude open_time)
        self.feature_names_ = [
            col for col in features.columns if col != "open_time"
        ]
        self._features = features

        logger.info(
            "transform() — complete. %d features × %d rows",
            len(self.feature_names_), len(features),
        )
        return features

    def save(self, config_path: Optional[str] = None) -> str:
        """
        Save generator configuration (windows + feature list) to JSON.

        Parameters
        ----------
        config_path : str, optional
            Path for the JSON config file.
            Default: ``models/feature_config/feature_generator_config.json``

        Returns
        -------
        str
            Absolute path to the saved config file.
        """
        if config_path is None:
            os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
            config_path = os.path.join(
                DEFAULT_CONFIG_DIR, "feature_generator_config.json"
            )

        config = {
            "windows": self.windows,
            "feature_names": self.feature_names_,
            "fitted": self.fitted_,
        }

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved FeatureGenerator config → %s", config_path)
        return config_path

    @classmethod
    def load(cls, config_path: str) -> "FeatureGenerator":
        """
        Restore a FeatureGenerator from a previously saved JSON config.

        Parameters
        ----------
        config_path : str
            Path to the JSON config file produced by ``save()``.

        Returns
        -------
        FeatureGenerator
            A new instance with the stored configuration.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        gen = cls(windows=config.get("windows", WINDOWS))
        gen.feature_names_ = config.get("feature_names", [])
        gen.fitted_ = config.get("fitted", False)

        logger.info(
            "Loaded FeatureGenerator config from %s — %d features",
            config_path, len(gen.feature_names_),
        )
        return gen

    # ══════════════════════════════════════════════════════════════════════
    #  Internal: Data Preparation
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLCV columns are typed correctly and sorted by time."""
        df["open_time"] = pd.to_datetime(df["open_time"])
        for col in ("open", "high", "low", "close", "volume",
                     "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "trades" in df.columns:
            df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype(int)

        df = df.sort_values("open_time").reset_index(drop=True)
        return df

    # ══════════════════════════════════════════════════════════════════════
    #  Internal: Feature Computation Blocks
    # ══════════════════════════════════════════════════════════════════════

    # ── Volatility ─────────────────────────────────────────────────────

    def _compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility features for each window size.

        1. Rolling Volatility   — std dev of log-returns over the window.
        2. ATR-style Volatility — Average True Range over the window.
        3. Realized Variance    — sum of squared log-returns over the window.
        """
        result = pd.DataFrame(index=df.index)
        log_returns = np.log(df["close"] / df["close"].shift(1))

        # True Range components
        tr = pd.DataFrame(index=df.index)
        tr["hl"] = df["high"] - df["low"]
        tr["hc"] = (df["high"] - df["close"].shift(1)).abs()
        tr["lc"] = (df["low"] - df["close"].shift(1)).abs()
        true_range = tr.max(axis=1)

        for w in self.windows:
            # Rolling Volatility (std of log returns)
            result[f"volatility_rolling_{w}"] = log_returns.rolling(w).std()

            # ATR-style Volatility (mean of True Range)
            result[f"volatility_atr_{w}"] = true_range.rolling(w).mean()

            # Realized Variance (sum of squared log returns)
            result[f"volatility_realized_var_{w}"] = (
                log_returns.pow(2).rolling(w).sum()
            )

        return result

    # ── Momentum ───────────────────────────────────────────────────────

    def _compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum features for each window size.

        1. Returns            — simple return over the window.
        2. Cumulative Returns  — cumulative log-return over the window.
        3. Price Acceleration  — second derivative of price (change in returns).
        """
        result = pd.DataFrame(index=df.index)
        close = df["close"]

        for w in self.windows:
            # Simple return over w periods
            result[f"momentum_return_{w}"] = close.pct_change(w)

            # Cumulative log-return over the window
            log_ret = np.log(close / close.shift(1))
            result[f"momentum_cumulative_logret_{w}"] = log_ret.rolling(w).sum()

            # Price acceleration (change in period-over-period returns)
            period_return = close.pct_change(1)
            result[f"momentum_acceleration_{w}"] = (
                period_return - period_return.shift(w)
            )

        return result

    # ── Volume ─────────────────────────────────────────────────────────

    def _compute_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume features for each window size.

        1. Volume Delta     — taker buy volume minus taker sell volume,
                              smoothed over the rolling window.
        2. Volume Imbalance — ratio of taker buy volume to total volume,
                              smoothed over the rolling window.
        """
        result = pd.DataFrame(index=df.index)

        # Taker buy volume is given; sell volume = total - buy
        taker_buy = df["taker_buy_base_vol"]
        total_vol = df["volume"]
        taker_sell = total_vol - taker_buy

        # Per-candle delta and imbalance
        vol_delta = taker_buy - taker_sell
        # Avoid division by zero
        vol_imbalance = taker_buy / total_vol.replace(0, np.nan)

        for w in self.windows:
            # Smoothed volume delta
            result[f"volume_delta_{w}"] = vol_delta.rolling(w).mean()

            # Smoothed volume imbalance ratio
            result[f"volume_imbalance_{w}"] = vol_imbalance.rolling(w).mean()

        return result

    # ── Order Flow ─────────────────────────────────────────────────────

    def _compute_order_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Order-flow features for each window size.

        These proxy order-flow dynamics from OHLCV candle data:

        1. Bid/Ask Imbalance — estimated from the candle body direction
           and taker volumes. Positive = buying pressure.
        2. Spread            — (high - low) / close as a proxy for
           intra-candle spread, smoothed over the window.
        3. Depth Imbalance   — ratio of taker-buy quote volume to total
           quote volume, smoothed. Proxies depth since real-time depth
           cannot be reconstructed from a single orderbook snapshot.

        If a live orderbook snapshot is available, exact spread and depth
        values are computed from it and appended as point-in-time features
        (not windowed).
        """
        result = pd.DataFrame(index=df.index)

        # ── Proxy features from OHLCV ────────────────────────────────
        taker_buy = df["taker_buy_base_vol"]
        total_vol = df["volume"]
        taker_sell = total_vol - taker_buy

        # Bid/Ask imbalance: (buy - sell) / (buy + sell)
        denom = (taker_buy + taker_sell).replace(0, np.nan)
        ba_imbalance = (taker_buy - taker_sell) / denom

        # Spread proxy: (high - low) / close
        spread_proxy = (df["high"] - df["low"]) / df["close"]

        # Depth imbalance proxy from quote volumes
        taker_buy_quote = df.get("taker_buy_quote_vol")
        quote_vol = df.get("quote_volume")
        if taker_buy_quote is not None and quote_vol is not None:
            depth_imb = taker_buy_quote / quote_vol.replace(0, np.nan)
        else:
            depth_imb = pd.Series(np.nan, index=df.index)

        for w in self.windows:
            result[f"orderflow_ba_imbalance_{w}"] = ba_imbalance.rolling(w).mean()
            result[f"orderflow_spread_{w}"] = spread_proxy.rolling(w).mean()
            result[f"orderflow_depth_imbalance_{w}"] = depth_imb.rolling(w).mean()

        # ── Point-in-time orderbook features (if available) ──────────
        if self._orderbook is not None:
            ob = self._orderbook
            bids = ob[ob["side"] == "bid"]
            asks = ob[ob["side"] == "ask"]

            if not bids.empty and not asks.empty:
                best_bid = bids["price"].max()
                best_ask = asks["price"].min()
                total_bid_qty = bids["quantity"].sum()
                total_ask_qty = asks["quantity"].sum()

                result["orderflow_ob_spread"] = best_ask - best_bid
                result["orderflow_ob_spread_bps"] = (
                    (best_ask - best_bid) / ((best_ask + best_bid) / 2) * 10_000
                )
                depth_denom = total_bid_qty + total_ask_qty
                result["orderflow_ob_depth_imbalance"] = (
                    (total_bid_qty - total_ask_qty) / depth_denom
                    if depth_denom > 0
                    else np.nan
                )

                logger.info(
                    "  Orderbook snapshot: spread=%.4f, bid_depth=%.4f, ask_depth=%.4f",
                    best_ask - best_bid, total_bid_qty, total_ask_qty,
                )

        return result
