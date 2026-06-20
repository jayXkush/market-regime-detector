# Deployment Guide — Render (Free Tier)

Deploy the Market Regime Detector API to Render using native Python runtime.

**What gets deployed:** Inference-only API. Training stays offline.

**What happens on each request:**
```
Load Models (once at startup) → Fetch Live Binance Data → Compute Features → Predict Regime
```

---

## Prerequisites

- A free [Render](https://render.com) account
- A [GitHub](https://github.com) account
- Trained model files in `models/` (you should already have these from Phase 3):
  - `scaler.pkl` (1.3 KB)
  - `pca.pkl` (4.7 KB)
  - `regime_model.pkl` (280.6 KB)

---

## Step 1 — Push Project to GitHub

### 1.1 Create a GitHub repository

1. Go to [github.com/new](https://github.com/new)
2. Name it `market-regime-detector`
3. Keep it **Public** or **Private** (both work with Render)
4. **Do NOT** initialize with README (you already have one)
5. Click **Create repository**

### 1.2 Initialize Git and push

Open a terminal in your project folder and run:

```bash
git init
git add .
git commit -m "Initial commit — Market Regime Detector"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/market-regime-detector.git
git push -u origin main
```

> **Replace** `YOUR_USERNAME` with your actual GitHub username.

### 1.3 Verify on GitHub

Go to your repo on GitHub and confirm these files exist:

```
✅ render.yaml
✅ requirements-deploy.txt
✅ run_api.py
✅ models/scaler.pkl
✅ models/pca.pkl
✅ models/regime_model.pkl
✅ src/api/app.py
✅ src/api/schemas.py
✅ src/data/binance_loader.py
✅ src/features/feature_generator.py
✅ src/inference/engine.py
✅ src/utils/logger.py
```

---

## Step 2 — Create Render Web Service

### Option A: Blueprint (Recommended — one click)

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub account (if not already connected)
4. Select your `market-regime-detector` repository
5. Render auto-detects `render.yaml` and shows the service config
6. Click **"Apply"**
7. Done — Render starts building and deploying

### Option B: Manual setup

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select `market-regime-detector`
4. Fill in:

| Field | Value |
|-------|-------|
| Name | `market-regime-detector` |
| Region | Oregon (US West) |
| Branch | `main` |
| Runtime | **Python** |
| Build Command | `pip install -r requirements-deploy.txt` |
| Start Command | `python run_api.py` |
| Instance Type | **Free** |

5. Click **"Create Web Service"**
6. After it's created, go to **Step 3** to add environment variables

---

## Step 3 — Set Environment Variables

Go to your service on Render → **"Environment"** tab → **"Add Environment Variable"**:

| Key | Value |
|-----|-------|
| `BINANCE_API_URL` | `https://api.binance.com` |
| `MODEL_PATH` | `models` |
| `PYTHON_ENV` | `production` |
| `PYTHON_VERSION` | `3.11.9` |

Click **"Save Changes"**.

> **Note:** If you used the Blueprint (Option A), these are already set from `render.yaml`. You can skip this step.

---

## Step 4 — Wait for Build

Render will now:

1. Clone your repository
2. Detect Python 3.11.9
3. Run `pip install -r requirements-deploy.txt`
4. Start your app with `python run_api.py`
5. Ping `/health` to confirm it's alive

This takes **2–4 minutes** on the first deploy.

### What the logs should look like

Go to your service → **"Logs"** tab. You should see:

```
==> Cloning from GitHub...
==> Using Python 3.11.9
==> Running build command: pip install -r requirements-deploy.txt
    Successfully installed fastapi uvicorn pandas numpy scikit-learn ...

==> Starting service with: python run_api.py
============================================================
  Market Regime Detection — Phase 7: Production Deployment
============================================================
  Host:   0.0.0.0
  Port:   10000
  Env:    production
============================================================
  Loaded scaler → models/scaler.pkl
  Loaded PCA → models/pca.pkl
  Loaded regime model → models/regime_model.pkl
  Engine ready.
============================================================
  InferenceEngine loaded successfully.
  Startup complete in 1.32s.
============================================================
INFO:     Uvicorn running on http://0.0.0.0:10000
```

Once you see `Uvicorn running`, the service is live.

---

## Step 5 — Verify It Works

Render gives you a URL like: `https://market-regime-detector.onrender.com`

### Test 1: Health Check

```bash
curl https://market-regime-detector.onrender.com/health
```

Expected:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_algorithm": "HDBSCAN",
  "uptime_seconds": 45.2,
  "timestamp": "2026-06-20T13:00:00Z"
}
```

### Test 2: Get Current Regime

```bash
curl https://market-regime-detector.onrender.com/regime/current
```

Expected (regime will vary based on current market conditions):
```json
{
  "current_regime": "Trending Down",
  "confidence": 0.15,
  "cluster_id": 0,
  "regime_description": "Market is in a sustained downward move...",
  "symbol": "BTCUSDT",
  "timestamp": "2026-06-20T13:20:54+00:00"
}
```

### Test 3: Different Symbol

```bash
curl "https://market-regime-detector.onrender.com/regime/current?symbol=ETHUSDT"
```

### Test 4: Prediction History

```bash
curl "https://market-regime-detector.onrender.com/regime/history?limit=5"
```

### Test 5: Swagger Docs (browser)

Open in your browser:
```
https://market-regime-detector.onrender.com/docs
```

---

## Endpoints Summary

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health + model status |
| `GET` | `/regime/current` | Live regime prediction |
| `GET` | `/regime/current?symbol=ETHUSDT` | Predict for specific symbol |
| `GET` | `/regime/history?limit=10` | Recent prediction history |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Free Tier Limitations

| Limitation | Detail |
|-----------|--------|
| **Spin-down** | Service sleeps after **15 min** of no requests |
| **Cold start** | First request after sleep takes **30–60 seconds** |
| **Hours** | 750 free hours/month (enough for 1 service) |
| **RAM** | 512 MB limit (this app uses ~180–220 MB ✅) |
| **Bandwidth** | 100 GB/month |

> To keep the service always-on, upgrade to the **Starter plan** ($7/month).

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Build fails with `ModuleNotFoundError` | Missing package | Add it to `requirements-deploy.txt` and push |
| `FileNotFoundError: scaler.pkl` | Model files not in repo | Run `git add models/*.pkl`, commit, push |
| Health check never passes | App crashed on startup | Check Logs tab for the error |
| `503 Service Unavailable` | Service is sleeping (free tier) | Wait 30–60s and retry |
| `429 Too Many Requests` from Binance | Rate limited | Built-in retry handles this automatically |
| High memory / OOM | Unlikely at ~220 MB | Check if `workers=1` is set in `run_api.py` |

### Force Redeploy

If something is stuck:

1. Go to your service on Render
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

---

## Updating After Deployment

When you make code changes locally:

```bash
git add .
git commit -m "your change description"
git push
```

Render auto-deploys on every push to `main`. No manual steps needed.

---

## Project Files Used by Render

```
Market regime detector/
├── render.yaml               ← Render reads this for config
├── requirements-deploy.txt   ← Render installs these packages
├── run_api.py                ← Render runs this to start the server
├── models/
│   ├── scaler.pkl            ← Loaded at startup
│   ├── pca.pkl               ← Loaded at startup
│   └── regime_model.pkl      ← Loaded at startup
└── src/
    ├── api/
    │   ├── app.py            ← FastAPI app with /health, /regime/*
    │   └── schemas.py        ← Response models
    ├── data/
    │   └── binance_loader.py ← Fetches live Binance data
    ├── features/
    │   └── feature_generator.py ← Computes features from market data
    ├── inference/
    │   └── engine.py         ← Loads models, runs prediction pipeline
    └── utils/
        └── logger.py         ← Logging config
```

Files **NOT** deployed (excluded by `.gitignore`): `venv/`, `data/`, `logs/`, `notebooks/`, `.env`
