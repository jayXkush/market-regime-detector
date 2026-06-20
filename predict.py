"""
predict.py — CLI entry point for Phase 4: Inference Engine.

Loads trained Phase 3 models and predicts the current market regime
for BTCUSDT using live Binance data.

Usage:
    python predict.py                          # Predict current regime
    python predict.py --symbol ETHUSDT         # Different symbol
    python predict.py --json                   # JSON-only output

Output:
    Prints the current regime, confidence, cluster ID, description,
    and timestamp.

Requires:
    - models/scaler.pkl     (from Phase 3)
    - models/pca.pkl        (from Phase 3)
    - models/regime_model.pkl (from Phase 3)
"""

from __future__ import annotations

import argparse
import json
import sys

from src.inference.engine import InferenceEngine
from src.utils.logger import get_logger

logger = get_logger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Market Regime Detection — Phase 4: Live Regime Prediction",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair to predict. Default: BTCUSDT.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output result as JSON only (no log messages).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.json_output:
        logger.info("=" * 60)
        logger.info("  Market Regime Detection — Phase 4: Live Prediction")
        logger.info("=" * 60)
        logger.info("  Symbol: %s", args.symbol)
        logger.info("")

    # ── Initialize engine ────────────────────────────────────────────
    engine = InferenceEngine(symbol=args.symbol)
    engine.initialize()

    # ── Run prediction ───────────────────────────────────────────────
    result = engine.predict_regime()

    # ── Display result ───────────────────────────────────────────────
    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  PREDICTION RESULT")
        logger.info("=" * 60)
        logger.info("  Current Regime:  %s", result["current_regime"])
        logger.info("  Confidence:      %.2f%%", result["confidence"] * 100)
        logger.info("  Cluster ID:      %d", result["cluster_id"])
        logger.info("  Timestamp:       %s", result["timestamp"])
        logger.info("")
        logger.info("  Description:")
        logger.info("    %s", result["regime_description"])
        logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
