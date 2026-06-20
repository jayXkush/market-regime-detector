# Market Regime Detection using Unsupervised Learning

Detect market regimes (trending, mean-reverting, volatile, calm) in cryptocurrency markets using unsupervised machine learning techniques.

## Project Overview

This project applies unsupervised learning algorithms to classify cryptocurrency market conditions into distinct regimes. By identifying the current market regime, traders and systems can adapt strategies accordingly.

**Supported Symbols:** `BTCUSDT`, `ETHUSDT`

## Phase Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Setup & Data Ingestion | ✅ Complete |
| 2 | Feature Engineering | ✅ Complete |
| 3 | Model Training (Clustering) | 🔲 Pending |
| 4 | Inference & Visualization | 🔲 Pending |
| 5 | Backtesting & Evaluation | 🔲 Pending |

## Setup

### Prerequisites
- Python 3.9+

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd market-regime

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Phase 1: Data Ingestion

Fetch raw market data from the Binance public API.

### Usage

```bash
# Fetch all data (OHLCV + trades + orderbook) for both symbols
python main.py

# Fetch specific data type
python main.py --type ohlcv
python main.py --type trades
python main.py --type orderbook

# Fetch for a specific symbol
python main.py --symbol BTCUSDT

# Custom OHLCV interval and limit
python main.py --type ohlcv --interval 1h --limit 500
```

### Output

Raw data is stored as CSV files in `data/raw/`:

```
data/raw/
├── BTCUSDT_ohlcv_1h_20260618_112700.csv
├── BTCUSDT_trades_20260618_112700.csv
├── BTCUSDT_orderbook_20260618_112700.csv
├── ETHUSDT_ohlcv_1h_20260618_112700.csv
├── ETHUSDT_trades_20260618_112700.csv
└── ETHUSDT_orderbook_20260618_112700.csv
```

## Phase 2: Feature Engineering

Generate market-regime features from the raw data.

### Usage

```bash
# Generate features for all symbols
python run_features.py

# Single symbol
python run_features.py --symbol BTCUSDT

# Show sample output
python run_features.py --show-sample
```

### Output

Features saved to `data/features/`:

```
data/features/
├── features.parquet      # Primary output (compressed)
└── features.csv          # Human-readable copy
```

Generator configs saved to `models/feature_config/`.

### Features (36 total)

| Category | Feature | Windows |
|----------|---------|--------|
| Volatility | Rolling Volatility (std of log returns) | 5, 15, 30 |
| Volatility | ATR-style Volatility (mean True Range) | 5, 15, 30 |
| Volatility | Realized Variance (sum of squared log returns) | 5, 15, 30 |
| Momentum | Returns (simple pct change) | 5, 15, 30 |
| Momentum | Cumulative Log Returns | 5, 15, 30 |
| Momentum | Price Acceleration (change in returns) | 5, 15, 30 |
| Volume | Volume Delta (buy − sell volume) | 5, 15, 30 |
| Volume | Volume Imbalance (buy / total ratio) | 5, 15, 30 |
| Order Flow | Bid/Ask Imbalance | 5, 15, 30 |
| Order Flow | Spread (high−low / close proxy) | 5, 15, 30 |
| Order Flow | Depth Imbalance (quote volume ratio) | 5, 15, 30 |
| Order Flow | Orderbook Spread (point-in-time) | — |
| Order Flow | Orderbook Spread (bps) | — |
| Order Flow | Orderbook Depth Imbalance | — |

## Project Structure

```
market-regime/
├── data/
│   └── raw/                  # Raw fetched data (CSV)
├── notebooks/                # Jupyter notebooks for exploration
├── models/                   # Saved model artifacts
├── src/
│   ├── data/
│   │   └── binance_loader.py # BinanceDataLoader class
│   ├── features/             # Feature engineering (Phase 2)
│   ├── training/             # Model training (Phase 3)
│   ├── inference/            # Inference pipeline (Phase 4)
│   └── utils/
│       └── logger.py         # Centralized logging
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

## License

MIT
