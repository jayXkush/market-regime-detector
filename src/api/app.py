"""
FastAPI application — Phase 5: API Backend.

Exposes the Phase 4 InferenceEngine through HTTP endpoints:

    GET /health          → service health & model status
    GET /regime/current  → live regime prediction
    GET /regime/history  → stored prediction history

Models are loaded ONCE at startup via the FastAPI lifespan context.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    RegimeHistoryEntry,
    RegimeHistoryResponse,
    RegimeResponse,
)
from src.inference.engine import InferenceEngine
from src.utils.logger import get_logger

logger = get_logger("api")

# ═══════════════════════════════════════════════════════════════════════════
#  Application State
# ═══════════════════════════════════════════════════════════════════════════

# Global references populated at startup
_engine: Optional[InferenceEngine] = None
_start_time: float = 0.0

# In-memory prediction history (capped at 100 entries)
_prediction_history: deque[dict] = deque(maxlen=100)


# ═══════════════════════════════════════════════════════════════════════════
#  Lifespan — load models once at startup
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the InferenceEngine (and all Phase 3 model artifacts) once
    when the server starts up. Avoids per-request loading overhead.
    """
    global _engine, _start_time

    logger.info("=" * 60)
    logger.info("  Phase 7 — Production Deployment starting up")
    logger.info("=" * 60)

    _start_time = time.time()

    # Log environment configuration
    import os
    model_path = os.environ.get("MODEL_PATH", "models/")
    binance_url = os.environ.get("BINANCE_API_URL", "https://api.binance.com")
    logger.info("  MODEL_PATH:      %s", model_path)
    logger.info("  BINANCE_API_URL: %s", binance_url)
    logger.info("  PYTHON_ENV:      %s", os.environ.get("PYTHON_ENV", "development"))

    try:
        _engine = InferenceEngine(symbol="BTCUSDT")
        _engine.initialize()
        logger.info("  InferenceEngine loaded successfully.")

        # Log model memory footprint
        import sys
        model_size = sys.getsizeof(_engine.model) if _engine.model else 0
        scaler_size = sys.getsizeof(_engine.scaler) if _engine.scaler else 0
        pca_size = sys.getsizeof(_engine.pca) if _engine.pca else 0
        logger.info("  Approx model objects size: ~%.1f KB",
                     (model_size + scaler_size + pca_size) / 1024)

    except Exception as exc:
        logger.exception("  FATAL: Failed to load InferenceEngine: %s", exc)
        raise

    load_time = time.time() - _start_time
    logger.info("  Startup complete in %.2fs.", load_time)
    logger.info("=" * 60)

    yield  # ← server is running

    logger.info("  Shutting down Phase 7 API …")


# ═══════════════════════════════════════════════════════════════════════════
#  FastAPI App
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Market Regime Detector API",
    description=(
        "Phase 5 — FastAPI backend for the Market Regime Detection system. "
        "Exposes live regime inference via REST endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# ── CORS — allow the dashboard frontend to call this API ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your dashboard URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health, model readiness, and uptime.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Check service health and model status."""
    logger.info("GET /health")

    model_loaded = _engine is not None and _engine._initialized
    model_algorithm = _engine.model_name if model_loaded else "N/A"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_algorithm=model_algorithm,
        uptime_seconds=round(time.time() - _start_time, 2),
        timestamp=datetime.now(timezone.utc),
    )


@app.get(
    "/regime/current",
    response_model=RegimeResponse,
    summary="Current regime prediction",
    description=(
        "Fetches live market data from Binance, runs the full inference "
        "pipeline, and returns the current market regime."
    ),
    responses={
        500: {"model": ErrorResponse, "description": "Inference failed"},
    },
    tags=["Regime"],
)
async def get_current_regime(
    symbol: str = Query(
        default="BTCUSDT",
        description="Trading pair to predict.",
        examples=["BTCUSDT", "ETHUSDT"],
    ),
) -> RegimeResponse:
    """Run the inference pipeline and return the current regime."""
    logger.info("GET /regime/current — symbol=%s", symbol)

    if _engine is None or not _engine._initialized:
        logger.error("  Engine not initialized.")
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")

    try:
        # Update symbol if different from default
        if symbol != _engine.symbol:
            _engine.symbol = symbol
            _engine.loader = None  # Force re-initialization of data loader
            from src.data.binance_loader import BinanceDataLoader
            _engine.loader = BinanceDataLoader(symbols=[symbol])

        result = _engine.predict_regime()

        response = RegimeResponse(
            current_regime=result["current_regime"],
            confidence=result["confidence"],
            cluster_id=result["cluster_id"],
            regime_description=result["regime_description"],
            symbol=symbol,
            timestamp=result["timestamp"],
        )

        # Store in history
        _prediction_history.appendleft(response.model_dump())

        logger.info("  Prediction: %s (confidence=%.4f)", response.current_regime, response.confidence)
        return response

    except Exception as exc:
        logger.exception("  Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


@app.get(
    "/regime/history",
    response_model=RegimeHistoryResponse,
    summary="Prediction history",
    description=(
        "Returns the most recent regime predictions stored in memory. "
        "History is limited to the last 100 predictions and resets on server restart."
    ),
    tags=["Regime"],
)
async def get_regime_history(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of history entries to return.",
    ),
) -> RegimeHistoryResponse:
    """Return recent prediction history."""
    logger.info("GET /regime/history — limit=%d", limit)

    entries = list(_prediction_history)[:limit]

    return RegimeHistoryResponse(
        total=len(entries),
        predictions=[RegimeHistoryEntry(**entry) for entry in entries],
    )
