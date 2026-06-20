"""
Pydantic schemas for Phase 5 — FastAPI Backend.

Defines request/response models with full validation for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
#  Health Check
# ═══════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(
        ...,
        description="Service health status.",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the inference model is loaded and ready.",
    )
    model_algorithm: str = Field(
        ...,
        description="Name of the loaded clustering algorithm.",
        examples=["KMeans"],
    )
    uptime_seconds: float = Field(
        ...,
        ge=0,
        description="Server uptime in seconds.",
    )
    timestamp: datetime = Field(
        ...,
        description="Current server time (UTC).",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Regime Prediction
# ═══════════════════════════════════════════════════════════════════════════

class RegimeResponse(BaseModel):
    """Response for GET /regime/current."""

    current_regime: str = Field(
        ...,
        description="Human-readable regime label.",
        examples=["High Volatility"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the prediction (0–1).",
    )
    cluster_id: int = Field(
        ...,
        ge=-1,
        description="Numeric cluster ID assigned by the model.",
    )
    regime_description: str = Field(
        ...,
        description="Detailed description of the regime.",
    )
    symbol: str = Field(
        ...,
        description="Trading pair used for this prediction.",
        examples=["BTCUSDT"],
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of the prediction (UTC).",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Regime History
# ═══════════════════════════════════════════════════════════════════════════

class RegimeHistoryEntry(BaseModel):
    """Single entry in the regime history."""

    current_regime: str = Field(..., description="Regime label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence.")
    cluster_id: int = Field(..., ge=-1, description="Cluster ID.")
    regime_description: str = Field(..., description="Regime description.")
    symbol: str = Field(..., description="Trading pair.")
    timestamp: str = Field(..., description="ISO 8601 timestamp (UTC).")


class RegimeHistoryResponse(BaseModel):
    """Response for GET /regime/history."""

    total: int = Field(
        ...,
        ge=0,
        description="Total number of predictions in the history.",
    )
    predictions: List[RegimeHistoryEntry] = Field(
        ...,
        description="List of past predictions, newest first.",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Error
# ═══════════════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(..., description="Error message.")
