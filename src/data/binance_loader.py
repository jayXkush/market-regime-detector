"""
BinanceDataLoader — Phase 1: Data Ingestion from Binance REST API.

Fetches three types of market data for configured trading pairs:
    • OHLCV   — candlestick (kline) data
    • Trades  — recent aggregate trades
    • Orderbook — order book depth snapshot

Includes automatic fallback: if api.binance.com is geo-blocked (HTTP 451),
the loader switches to data-api.binance.vision for market data.

All data is returned as normalised pandas DataFrames ready for
downstream feature engineering (Phase 2).

Public API:
    loader = BinanceDataLoader(symbols=["BTCUSDT"])
    ohlcv  = loader.fetch_ohlcv("BTCUSDT", interval="1h", limit=500)
    trades = loader.fetch_trades("BTCUSDT", limit=500)
    book   = loader.fetch_orderbook("BTCUSDT", limit=100)
    all_data = loader.fetch_all()
    loader.save_to_csv(ohlcv, "BTCUSDT", "ohlcv")
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Ordered list of Binance API base URLs to try.
# data-api.binance.vision is the official data-only endpoint that is
# typically NOT geo-blocked, unlike api.binance.com.
_BINANCE_URLS = [
    os.environ.get("BINANCE_API_URL", "https://api.binance.com"),
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")

# Binance kline column mapping (positional from the API response)
_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]


class BinanceDataLoader:
    """
    Fetch market data from the Binance public REST API.

    Automatically tries multiple Binance API endpoints.  If the primary
    endpoint returns HTTP 451 (geo-blocked), the loader falls back to
    ``data-api.binance.vision`` and other mirrors.

    Parameters
    ----------
    symbols : list[str], optional
        Trading pairs to fetch.  Defaults to ``["BTCUSDT", "ETHUSDT"]``.
    base_url : str, optional
        Binance API base URL.  Reads ``BINANCE_API_URL`` env var first.
    output_dir : str, optional
        Directory for ``save_to_csv()`` output.  Default: ``data/raw/``.
    """

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        base_url: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.symbols: list[str] = symbols or list(DEFAULT_SYMBOLS)
        self.output_dir: str = output_dir or DEFAULT_RAW_DIR
        self._session = requests.Session()

        # Build the ordered list of base URLs to try
        if base_url:
            self._base_urls = [base_url.rstrip("/")]
        else:
            # De-duplicate while preserving order
            seen: set[str] = set()
            self._base_urls = []
            for u in _BINANCE_URLS:
                u = u.rstrip("/")
                if u not in seen:
                    seen.add(u)
                    self._base_urls.append(u)

        # The "active" base URL (updated after successful fallback)
        self.base_url: str = self._base_urls[0]

        logger.info(
            "BinanceDataLoader created — symbols=%s, base_url=%s (fallbacks: %d)",
            self.symbols, self.base_url, len(self._base_urls) - 1,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  OHLCV (Klines)
    # ══════════════════════════════════════════════════════════════════════

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch candlestick / kline data from Binance.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
        interval : str
            Candle interval (``"1m"``, ``"5m"``, ``"1h"``, ``"4h"``, ``"1d"``, …).
        limit : int
            Number of candles to return (max 1000).

        Returns
        -------
        pd.DataFrame
            Columns: open_time, open, high, low, close, volume,
            quote_volume, trades, taker_buy_base_vol, taker_buy_quote_vol.
        """
        path = "/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}

        logger.info("Fetching OHLCV — %s %s (limit=%d) …", symbol, interval, params["limit"])
        resp = self._request_with_fallback(path, params)

        df = pd.DataFrame(resp, columns=_KLINE_COLUMNS)

        # Convert types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype(int)

        # Drop helper columns
        df = df.drop(columns=["close_time", "ignore"], errors="ignore")

        logger.info("  OHLCV fetched: %d rows", len(df))
        return df

    # ══════════════════════════════════════════════════════════════════════
    #  Recent Trades
    # ══════════════════════════════════════════════════════════════════════

    def fetch_trades(
        self,
        symbol: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch recent aggregate trades.

        Parameters
        ----------
        symbol : str
            Trading pair.
        limit : int
            Number of trades to return (max 1000).

        Returns
        -------
        pd.DataFrame
            Columns: agg_trade_id, price, qty, first_trade_id,
            last_trade_id, time, isBuyerMaker.
        """
        path = "/api/v3/aggTrades"
        params = {"symbol": symbol, "limit": min(limit, 1000)}

        logger.info("Fetching trades — %s (limit=%d) …", symbol, params["limit"])
        resp = self._request_with_fallback(path, params)

        df = pd.DataFrame(resp)

        # Rename Binance fields to friendlier names
        rename_map = {
            "a": "agg_trade_id",
            "p": "price",
            "q": "qty",
            "f": "first_trade_id",
            "l": "last_trade_id",
            "T": "time",
            "m": "isBuyerMaker",
        }
        df = df.rename(columns=rename_map)

        # Types
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        for col in ("price", "qty"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("  Trades fetched: %d rows", len(df))
        return df

    # ══════════════════════════════════════════════════════════════════════
    #  Orderbook (Depth)
    # ══════════════════════════════════════════════════════════════════════

    def fetch_orderbook(
        self,
        symbol: str,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch order book depth snapshot.

        Parameters
        ----------
        symbol : str
            Trading pair.
        limit : int
            Number of price levels per side (max 5000, valid: 5/10/20/50/100/500/1000/5000).

        Returns
        -------
        pd.DataFrame
            Columns: side (``"bid"``/``"ask"``), price, quantity.
        """
        # Binance only accepts specific limit values
        valid_limits = [5, 10, 20, 50, 100, 500, 1000, 5000]
        actual_limit = min(valid_limits, key=lambda x: abs(x - limit))

        path = "/api/v3/depth"
        params = {"symbol": symbol, "limit": actual_limit}

        logger.info("Fetching orderbook — %s (limit=%d) …", symbol, actual_limit)
        resp = self._request_with_fallback(path, params)

        rows = []
        for price, qty in resp.get("bids", []):
            rows.append({"side": "bid", "price": float(price), "quantity": float(qty)})
        for price, qty in resp.get("asks", []):
            rows.append({"side": "ask", "price": float(price), "quantity": float(qty)})

        df = pd.DataFrame(rows)
        logger.info("  Orderbook fetched: %d levels", len(df))
        return df

    # ══════════════════════════════════════════════════════════════════════
    #  Fetch All
    # ══════════════════════════════════════════════════════════════════════

    def fetch_all(
        self,
        interval: str = "1h",
        ohlcv_limit: int = 500,
        trades_limit: int = 500,
        orderbook_limit: int = 100,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Fetch OHLCV, trades, and orderbook data for every configured symbol.

        Parameters
        ----------
        interval : str
            OHLCV candle interval.
        ohlcv_limit : int
            Number of OHLCV candles per symbol.
        trades_limit : int
            Number of recent trades per symbol.
        orderbook_limit : int
            Number of orderbook levels per side per symbol.

        Returns
        -------
        dict[str, dict[str, pd.DataFrame]]
            ``{symbol: {"ohlcv": df, "trades": df, "orderbook": df}}``
        """
        results: dict[str, dict[str, pd.DataFrame]] = {}

        for symbol in self.symbols:
            logger.info("━" * 60)
            logger.info("Fetching all data for %s …", symbol)
            logger.info("━" * 60)

            data: dict[str, pd.DataFrame] = {}

            data["ohlcv"] = self.fetch_ohlcv(symbol, interval=interval, limit=ohlcv_limit)
            self.save_to_csv(data["ohlcv"], symbol, "ohlcv", extra_tag=interval)

            try:
                data["trades"] = self.fetch_trades(symbol, limit=trades_limit)
                self.save_to_csv(data["trades"], symbol, "trades")
            except Exception as exc:
                logger.warning("  Failed to fetch trades for %s: %s", symbol, exc)
                data["trades"] = pd.DataFrame()

            try:
                data["orderbook"] = self.fetch_orderbook(symbol, limit=orderbook_limit)
                self.save_to_csv(data["orderbook"], symbol, "orderbook")
            except Exception as exc:
                logger.warning("  Failed to fetch orderbook for %s: %s", symbol, exc)
                data["orderbook"] = pd.DataFrame()

            results[symbol] = data

        return results

    # ══════════════════════════════════════════════════════════════════════
    #  CSV Persistence
    # ══════════════════════════════════════════════════════════════════════

    def save_to_csv(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: str,
        extra_tag: Optional[str] = None,
    ) -> str:
        """
        Save a DataFrame to CSV in ``data/raw/``.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save.
        symbol : str
            Trading pair name (used in filename).
        data_type : str
            One of ``"ohlcv"``, ``"trades"``, ``"orderbook"``.
        extra_tag : str, optional
            Additional tag to include in the filename (e.g. the interval).

        Returns
        -------
        str
            Absolute path to the saved CSV.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{extra_tag}" if extra_tag else ""
        filename = f"{symbol}_{data_type}{tag}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        df.to_csv(filepath, index=False)
        logger.info("  Saved → %s (%d rows)", filepath, len(df))
        return filepath

    # ══════════════════════════════════════════════════════════════════════
    #  Internal — request with automatic endpoint fallback
    # ══════════════════════════════════════════════════════════════════════

    def _request_with_fallback(self, path: str, params: dict) -> dict | list:
        """
        Try each base URL in order until one succeeds.

        On HTTP 451 (geo-blocked) or connection errors, automatically
        retries with the next base URL.  Once a working URL is found it
        becomes the new default for subsequent calls.

        Parameters
        ----------
        path : str
            API path, e.g. ``"/api/v3/klines"``.
        params : dict
            Query parameters.

        Returns
        -------
        dict | list
            Parsed JSON response.

        Raises
        ------
        requests.HTTPError
            If ALL endpoints fail.
        """
        last_exc: Exception | None = None

        for base_url in self._base_urls:
            url = f"{base_url}{path}"
            try:
                resp = self._session.get(url, params=params, timeout=30)

                # HTTP 451 = geo-blocked → try next endpoint
                if resp.status_code == 451:
                    logger.warning(
                        "  Geo-blocked (HTTP 451) by %s — trying next endpoint …",
                        base_url,
                    )
                    continue

                resp.raise_for_status()

                # Success — remember this base URL for future calls
                if base_url != self.base_url:
                    logger.info(
                        "  Switched to working endpoint: %s", base_url,
                    )
                    self.base_url = base_url
                    # Move this URL to the front for faster future lookups
                    self._base_urls.remove(base_url)
                    self._base_urls.insert(0, base_url)

                return resp.json()

            except requests.ConnectionError as exc:
                logger.warning(
                    "  Connection failed for %s — trying next endpoint … (%s)",
                    base_url, exc,
                )
                last_exc = exc
            except requests.HTTPError as exc:
                logger.warning(
                    "  HTTP error from %s: %s — trying next endpoint …",
                    base_url, exc,
                )
                last_exc = exc

        # All endpoints failed
        msg = f"All Binance API endpoints failed for {path}. Last error: {last_exc}"
        logger.error(msg)
        raise requests.ConnectionError(msg)
