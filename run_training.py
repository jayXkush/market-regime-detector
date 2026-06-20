"""
run_training.py — CLI entry point for Phase 3: Clustering & Regime Discovery.

Reads data/features/features.parquet, preprocesses it, trains clustering
models, evaluates them, selects the best, assigns regime labels, generates
a report, and saves all artifacts.

Usage:
    python run_training.py                     # Full pipeline
    python run_training.py --n-clusters 4      # Custom cluster count
    python run_training.py --pca-variance 0.90 # Custom PCA variance

Output:
    models/scaler.pkl           — Fitted StandardScaler
    models/pca.pkl              — Fitted PCA transformer
    models/regime_model.pkl     — Best clustering model + labels + metadata
    reports/regime_report.txt   — Text report with regime profiles
    reports/pca_scatter.png     — PCA scatter plot
    reports/cluster_visualization.png — Cluster centroids and sizes
    reports/regime_distribution.png   — Regime distribution chart
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from src.training.preprocessor import Preprocessor
from src.training.clustering import ClusteringTrainer
from src.training.evaluator import ClusterEvaluator
from src.training.regime_labeler import RegimeLabeler
from src.training.visualizer import Visualizer
from src.utils.logger import get_logger

logger = get_logger("run_training")

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
FEATURES_PATH = os.path.join(_PROJECT_ROOT, "data", "features", "features.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Market Regime Detection — Phase 3: Clustering & Regime Discovery",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=FEATURES_PATH,
        help="Path to features.parquet. Default: data/features/features.parquet.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=6,
        help="Number of clusters for KMeans and GMM. Default: 6.",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="PCA variance retention threshold. Default: 0.95.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Market Regime Detection — Phase 3: Clustering & Regime Discovery")
    logger.info("=" * 60)

    # ── 1. Load features ─────────────────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("  Step 1: Loading features")
    logger.info("━" * 60)

    if not os.path.exists(args.features_path):
        logger.error("Features file not found: %s", args.features_path)
        logger.error("Run Phase 2 first: python run_features.py")
        sys.exit(1)

    df = pd.read_parquet(args.features_path)
    logger.info("  Loaded: %s", args.features_path)
    logger.info("  Shape: %s", df.shape)
    logger.info("  Columns: %d", len(df.columns))
    if "symbol" in df.columns:
        logger.info("  Symbols: %s", df["symbol"].unique().tolist())

    # ── 2. Preprocessing ─────────────────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("  Step 2: Preprocessing")
    logger.info("━" * 60)

    preprocessor = Preprocessor(pca_variance=args.pca_variance)
    X_pca, pca_columns = preprocessor.fit_transform(df)

    # Save scaler and PCA
    scaler_path, pca_path = preprocessor.save()
    logger.info("  Saved: %s", scaler_path)
    logger.info("  Saved: %s", pca_path)

    # Get metadata aligned with processed data
    meta_df = preprocessor.get_metadata()

    # Also get the raw features (pre-PCA) aligned with processed data for regime profiling
    meta_cols = {"open_time", "symbol"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    features_raw = df[feature_cols].copy()

    # Handle missing values the same way the preprocessor does
    nan_threshold = int(len(feature_cols) * 0.5)
    features_raw = features_raw.dropna(thresh=nan_threshold)
    remaining_nan = features_raw.isnull().sum().sum()
    if remaining_nan > 0:
        features_raw = features_raw.fillna(features_raw.median())

    # Align with PCA output length
    if len(features_raw) > len(X_pca):
        features_raw = features_raw.iloc[:len(X_pca)]
    features_raw = features_raw.reset_index(drop=True)

    # Rebuild aligned metadata
    if "open_time" in df.columns or "symbol" in df.columns:
        aligned_meta = df[[c for c in df.columns if c in meta_cols]].copy()
        aligned_meta = aligned_meta.iloc[:len(X_pca)].reset_index(drop=True)
        features_for_profiling = pd.concat([aligned_meta, features_raw], axis=1)
    else:
        features_for_profiling = features_raw

    # ── 3. Clustering ────────────────────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("  Step 3: Clustering")
    logger.info("━" * 60)

    trainer = ClusteringTrainer(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )
    results = trainer.fit_all(X_pca)

    # ── 4. Evaluation ────────────────────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("  Step 4: Evaluation")
    logger.info("━" * 60)

    evaluator = ClusterEvaluator(random_state=args.random_state)
    scores = evaluator.evaluate_all(
        X_pca,
        trainer.labels,
        models=trainer.models,
    )

    # ── 5. Model Selection ───────────────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("  Step 5: Model Selection")
    logger.info("━" * 60)

    best_name, best_model, best_labels = trainer.select_best(scores)

    # ── 6. Regime Labeling ───────────────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("  Step 6: Regime Labeling")
    logger.info("━" * 60)

    labeler = RegimeLabeler()
    regime_map, profiles = labeler.label_clusters(best_labels, features_for_profiling)

    # ── 7. Save model ────────────────────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("  Step 7: Save Model")
    logger.info("━" * 60)

    model_path = trainer.save_model(
        model=best_model,
        model_name=best_name,
        labels=best_labels,
        metadata={
            "regime_map": regime_map,
            "cluster_profiles": {k: {kk: vv for kk, vv in v.items() if kk != "volatility_detail" and kk != "momentum_detail" and kk != "volume_detail" and kk != "orderflow_detail"} for k, v in profiles.items()},
            "scores": {name: {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in s.items()} for name, s in scores.items()},
            "n_clusters": args.n_clusters,
            "pca_variance": args.pca_variance,
            "random_state": args.random_state,
            "pca_components": X_pca.shape[1],
            "n_samples": len(X_pca),
        },
    )

    # ── 8. Generate Report ───────────────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("  Step 8: Generate Report")
    logger.info("━" * 60)

    report = labeler.generate_report(
        regime_map=regime_map,
        profiles=profiles,
        scores=scores,
        best_model_name=best_name,
    )

    # Print report to console
    logger.info("\n%s", report)

    # ── 9. Visualizations ────────────────────────────────────────────
    logger.info("")
    logger.info("━" * 60)
    logger.info("  Step 9: Visualizations")
    logger.info("━" * 60)

    visualizer = Visualizer()
    plot_paths = visualizer.generate_all(X_pca, best_labels, regime_map)

    for p in plot_paths:
        logger.info("  Plot saved: %s", p)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Phase 3: Clustering & Regime Discovery — COMPLETE")
    logger.info("=" * 60)
    logger.info("  Artifacts:")
    logger.info("    models/scaler.pkl")
    logger.info("    models/pca.pkl")
    logger.info("    models/regime_model.pkl")
    logger.info("    reports/regime_report.txt")
    for p in plot_paths:
        logger.info("    %s", os.path.relpath(p, _PROJECT_ROOT))
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
