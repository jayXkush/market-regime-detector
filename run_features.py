"""
run_features.py — CLI entry point for Phase 2: Feature Engineering.

Reads raw OHLCV, trades, and orderbook CSVs from data/raw/, generates all
features via FeatureGenerator, and saves the output to data/features/.

Usage:
    python run_features.py                         # All symbols (latest files)
    python run_features.py --symbol BTCUSDT         # Single symbol
    python run_features.py --show-sample            # Print sample rows
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

from src.features.feature_generator import FeatureGenerator
from src.utils.logger import get_logger

logger = get_logger("run_features")

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
FEATURES_DIR = os.path.join(_PROJECT_ROOT, "data", "features")
CONFIG_DIR = os.path.join(_PROJECT_ROOT, "models", "feature_config")

SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _find_latest_csv(directory: str, pattern: str) -> str | None:
    """Return the most-recently modified CSV matching the glob pattern."""
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _load_raw_data(symbol: str) -> dict[str, pd.DataFrame | None]:
    """Load the latest raw CSV files for a given symbol."""
    ohlcv_path = _find_latest_csv(RAW_DIR, f"{symbol}_ohlcv_*.csv")
    trades_path = _find_latest_csv(RAW_DIR, f"{symbol}_trades_*.csv")
    orderbook_path = _find_latest_csv(RAW_DIR, f"{symbol}_orderbook_*.csv")

    data: dict[str, pd.DataFrame | None] = {}

    if ohlcv_path:
        data["ohlcv"] = pd.read_csv(ohlcv_path)
        logger.info("  Loaded OHLCV: %s (%d rows)", os.path.basename(ohlcv_path), len(data["ohlcv"]))
    else:
        logger.error("  No OHLCV file found for %s — cannot proceed.", symbol)
        data["ohlcv"] = None

    if trades_path:
        data["trades"] = pd.read_csv(trades_path)
        logger.info("  Loaded trades: %s (%d rows)", os.path.basename(trades_path), len(data["trades"]))
    else:
        data["trades"] = None
        logger.warning("  No trades file found for %s — volume features may be limited.", symbol)

    if orderbook_path:
        data["orderbook"] = pd.read_csv(orderbook_path)
        logger.info("  Loaded orderbook: %s (%d rows)", os.path.basename(orderbook_path), len(data["orderbook"]))
    else:
        data["orderbook"] = None
        logger.warning("  No orderbook file found for %s — orderflow features may be limited.", symbol)

    return data


def run_feature_pipeline(symbols: list[str], show_sample: bool = False) -> None:
    """Run the full feature engineering pipeline for the given symbols."""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    all_features: list[pd.DataFrame] = []

    for symbol in symbols:
        logger.info("━" * 60)
        logger.info("Processing %s", symbol)
        logger.info("━" * 60)

        raw_data = _load_raw_data(symbol)
        if raw_data["ohlcv"] is None:
            logger.error("Skipping %s — OHLCV data required.", symbol)
            continue

        generator = FeatureGenerator()
        generator.fit(
            ohlcv=raw_data["ohlcv"],
            trades=raw_data.get("trades"),
            orderbook=raw_data.get("orderbook"),
        )
        features = generator.transform()
        features["symbol"] = symbol

        # Save per-symbol config
        config_path = os.path.join(CONFIG_DIR, f"{symbol}_feature_config.json")
        generator.save(config_path)

        all_features.append(features)

        logger.info("  Generated %d features × %d rows for %s",
                     len(generator.feature_names_), len(features), symbol)

        if show_sample:
            logger.info("\n  Sample (first 5 rows):\n%s", features.head().to_string())

    if not all_features:
        logger.error("No features generated. Check that data/raw/ has OHLCV files.")
        sys.exit(1)

    # Combine all symbols
    combined = pd.concat(all_features, ignore_index=True)

    # Save combined output
    output_path = os.path.join(FEATURES_DIR, "features.parquet")
    combined.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("━" * 60)
    logger.info("Saved combined features → %s", output_path)
    logger.info("  Total rows: %d", len(combined))
    logger.info("  Total features: %d", len([c for c in combined.columns if c not in ("open_time", "symbol")]))

    # Also save as CSV for easy inspection
    csv_path = os.path.join(FEATURES_DIR, "features.csv")
    combined.to_csv(csv_path, index=False)
    logger.info("  CSV copy → %s", csv_path)

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Phase 2: Feature Engineering — Summary")
    logger.info("=" * 60)
    logger.info("  Symbols processed: %s", symbols)
    logger.info("  Window sizes:      %s", [5, 15, 30])
    logger.info("  Feature columns:   %d", len([c for c in combined.columns if c not in ("open_time", "symbol")]))
    logger.info("  Total rows:        %d", len(combined))
    logger.info("  Output:            %s", output_path)
    logger.info("")

    # Print feature list
    feature_cols = [c for c in combined.columns if c not in ("open_time", "symbol")]
    logger.info("  Feature List:")
    for i, col in enumerate(feature_cols, 1):
        logger.info("    %2d. %s", i, col)
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Market Regime Detection — Phase 2: Feature Engineering",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol (e.g. BTCUSDT). Default: all supported.",
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Print a sample of the generated features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Market Regime Detection — Phase 2: Feature Engineering")
    logger.info("=" * 60)

    symbols = [args.symbol] if args.symbol else list(SUPPORTED_SYMBOLS)
    run_feature_pipeline(symbols, show_sample=args.show_sample)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
