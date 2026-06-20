"""
main.py — CLI entry point for Phase 1: Data Ingestion.

Usage:
    python main.py                           # Fetch everything for all symbols
    python main.py --type ohlcv              # OHLCV only
    python main.py --type trades             # Trades only
    python main.py --type orderbook          # Order book only
    python main.py --symbol BTCUSDT          # Single symbol
    python main.py --interval 4h --limit 200 # Custom OHLCV params
"""

from __future__ import annotations

import argparse
import sys

from src.data.binance_loader import BinanceDataLoader
from src.utils.logger import get_logger

logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Market Regime Detection — Phase 1: Data Ingestion",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol to fetch (e.g. BTCUSDT). Default: all supported.",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["ohlcv", "trades", "orderbook", "all"],
        default="all",
        help="Type of data to fetch. Default: all.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="OHLCV candle interval (default: 1h).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of records to fetch (default: 500).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Market Regime Detection — Phase 1: Data Ingestion")
    logger.info("=" * 60)

    # Determine symbols
    symbols = [args.symbol] if args.symbol else None
    loader = BinanceDataLoader(symbols=symbols)

    data_type = args.type

    if data_type == "all":
        results = loader.fetch_all(
            interval=args.interval,
            ohlcv_limit=args.limit,
            trades_limit=args.limit,
            orderbook_limit=min(args.limit, 1000),
        )
        _print_summary(results)
    else:
        for symbol in loader.symbols:
            logger.info("━" * 60)
            logger.info("Fetching %s for %s", data_type, symbol)
            logger.info("━" * 60)

            if data_type == "ohlcv":
                df = loader.fetch_ohlcv(symbol, interval=args.interval, limit=args.limit)
                loader.save_to_csv(df, symbol, "ohlcv", extra_tag=args.interval)
            elif data_type == "trades":
                df = loader.fetch_trades(symbol, limit=args.limit)
                loader.save_to_csv(df, symbol, "trades")
            elif data_type == "orderbook":
                df = loader.fetch_orderbook(symbol, limit=args.limit)
                loader.save_to_csv(df, symbol, "orderbook")

            logger.info("Preview (%s — %s):", symbol, data_type)
            logger.info("\n%s", df.head().to_string())


def _print_summary(results: dict) -> None:
    """Print a formatted summary of all fetched data."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Ingestion Summary")
    logger.info("=" * 60)

    for symbol, data_dict in results.items():
        logger.info("  %s:", symbol)
        for dtype, df in data_dict.items():
            logger.info("    %-12s → %d rows", dtype, len(df))
        logger.info("")

    logger.info("Raw data saved to: data/raw/")
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
