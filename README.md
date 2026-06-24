# Market Regime Detector

Figures out what "mood" the crypto market is in right now — trending, mean-reverting, volatile, calm, etc. — using unsupervised ML on live Binance data.

The idea is simple: markets behave differently in different conditions, so if you can detect the current regime, you can pick the right strategy for it.

## How it works

The whole thing is a pipeline that goes like this:

1. **Pull data** — Grabs OHLCV candles, recent trades, and orderbook snapshots from the Binance API for BTC and ETH.

2. **Build features** — Computes ~36 features from the raw data: rolling volatility, ATR, realized variance, momentum signals, volume deltas, bid/ask imbalance, spread, etc. Each one is calculated across multiple lookback windows (5, 15, 30 periods).

3. **Cluster** — Runs the features through StandardScaler → PCA (for dimensionality reduction) → clustering. Tries KMeans, GMM, and HDBSCAN, picks the best one based on silhouette scores, and assigns human-readable labels to each cluster (like "High Volatility Trending" or "Calm Mean-Reverting").

4. **Predict** — For a live prediction, it pulls fresh data from Binance, computes the same features, transforms them through the saved scaler/PCA, and asks the trained model which cluster this looks like. Returns the regime label + confidence score.

5. **Serve** — A FastAPI backend wraps the inference engine so you can hit an endpoint and get the current regime as JSON. There's also a React dashboard (Vite) that shows it visually.

## Getting started

```bash
git clone https://github.com/jayXkush/market-regime-detector.git
cd market-regime-detector

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Running the pipeline

Each phase has its own script. You need to run them in order the first time (data → features → training), after that you can just run predictions.

```bash
# Step 1: Fetch raw data from Binance
python main.py

# Step 2: Generate features
python run_features.py

# Step 3: Train clustering models
python run_training.py

# Step 4: Predict current regime
python predict.py
python predict.py --symbol ETHUSDT
python predict.py --json              # machine-readable output
```

### API server

```bash
python run_api.py                     # starts on port 10000
python run_api.py --port 8080         # custom port
```

Endpoints:
- `GET /health` — is the server up, is the model loaded
- `GET /regime/current?symbol=BTCUSDT` — live regime prediction
- `GET /regime/history?limit=10` — recent predictions

Swagger docs at `http://localhost:10000/docs`

### Dashboard

The `dashboard/` folder has a React + Vite frontend that talks to the API.

```bash
cd dashboard
npm install
npm run dev
```

## CLI options

Most scripts accept flags to tweak behavior:

```bash
# data ingestion
python main.py --type ohlcv --symbol BTCUSDT --interval 1h --limit 500

# feature engineering
python run_features.py --symbol BTCUSDT --show-sample

# training
python run_training.py --n-clusters 4 --pca-variance 0.90
```

## Project layout

```
├── main.py                  # data ingestion CLI
├── run_features.py          # feature engineering CLI
├── run_training.py          # model training CLI
├── predict.py               # inference CLI
├── run_api.py               # FastAPI server
├── src/
│   ├── data/                # Binance data loader
│   ├── features/            # feature generators
│   ├── training/            # preprocessor, clustering, evaluation, labeling, viz
│   ├── inference/           # inference engine
│   ├── api/                 # FastAPI app + schemas
│   └── utils/               # logger
├── dashboard/               # React frontend (Vite)
├── models/                  # saved model artifacts (scaler, PCA, regime_model)
├── data/
│   ├── raw/                 # fetched CSVs
│   └── features/            # computed features (parquet + csv)
├── requirements.txt
└── render.yaml              # Render deployment config
```

## Features computed

| Category | What it measures | Windows |
|----------|-----------------|---------|
| Volatility | Rolling std of log returns, ATR, realized variance | 5, 15, 30 |
| Momentum | Returns, cumulative log returns, price acceleration | 5, 15, 30 |
| Volume | Buy/sell volume delta, volume imbalance ratio | 5, 15, 30 |
| Order Flow | Bid/ask imbalance, spread, depth imbalance, orderbook metrics | 5, 15, 30 |

## Deployment

Backend is set up to deploy on Render (free tier). See `render.yaml` for the config. The dashboard can be deployed as a static site.

## Tech stack

- Python, pandas, numpy, scikit-learn, HDBSCAN
- FastAPI + Uvicorn
- React + Vite (dashboard)
- Binance public API (no key needed)

## License

MIT
