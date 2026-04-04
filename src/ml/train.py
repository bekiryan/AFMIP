"""
AFMIP — Stage 3: Model Training
================================
Trains a binary classifier (UP / DOWN) on the feature matrix.

Supports:
  - RandomForestClassifier
  - XGBoostClassifier
  - Hyperparameter tuning via RandomizedSearchCV
  - Walk-forward (time-series safe) cross-validation
  - Model persistence (joblib)

Usage:
    python -m src.ml.train                        # train both models, auto-select best
    python -m src.ml.train --model xgboost        # train only XGBoost
    python -m src.ml.train --no-tune              # skip hyperparameter search (faster)
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
import joblib

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.ml.features import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")


# ---------------------------------------------------------------------------
# Train / test split (time-series safe — no random shuffling)
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time: first (1 - test_ratio) of dates → train,
    remaining → test. Never shuffles to prevent look-ahead bias.
    """
    dates = df["date"].sort_values().unique()
    cutoff_idx = int(len(dates) * (1 - test_ratio))
    cutoff = dates[cutoff_idx]

    train = df[df["date"] < cutoff].copy()
    test  = df[df["date"] >= cutoff].copy()

    logger.info(
        f"Train: {len(train):,} rows  ({train['date'].min().date()} → {train['date'].max().date()})"
    )
    logger.info(
        f"Test:  {len(test):,} rows  ({test['date'].min().date()} → {test['date'].max().date()})"
    )
    return train, test


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _build_xgb_pipeline() -> Pipeline:
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------

RF_PARAM_DIST = {
    "model__n_estimators":   [100, 200, 300, 500],
    "model__max_depth":      [4, 6, 8, 10, None],
    "model__min_samples_leaf": [10, 20, 40],
    "model__max_features":   ["sqrt", "log2", 0.5],
}

XGB_PARAM_DIST = {
    "model__n_estimators":   [100, 200, 300, 500],
    "model__max_depth":      [3, 5, 7],
    "model__learning_rate":  [0.01, 0.05, 0.1, 0.2],
    "model__subsample":      [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
}


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    model_type: Literal["rf", "xgboost"] = "rf",
    tune: bool = True,
    n_iter: int = 20,
) -> tuple[Pipeline, dict]:
    """
    Train a classifier and return (fitted_pipeline, metrics_dict).

    Uses TimeSeriesSplit for CV to respect temporal ordering.
    """
    train, test = time_split(df)

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL]

    # --- Build pipeline ---
    if model_type == "rf":
        pipeline   = _build_rf_pipeline()
        param_dist = RF_PARAM_DIST
    elif model_type == "xgboost":
        pipeline   = _build_xgb_pipeline()
        param_dist = XGB_PARAM_DIST
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'rf' or 'xgboost'.")

    # --- Hyperparameter tuning ---
    tscv = TimeSeriesSplit(n_splits=5)

    if tune:
        logger.info(f"Running RandomizedSearchCV ({n_iter} iterations, 5-fold TimeSeriesSplit) …")
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        logger.info(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        logger.info(f"Best params: {search.best_params_}")
    else:
        logger.info("Fitting with default hyperparameters (--no-tune) …")
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline

    # --- Evaluate on held-out test set ---
    y_pred      = best_pipeline.predict(X_test)
    y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_pred_prob),
        "report":    classification_report(y_test, y_pred, target_names=["DOWN", "UP"]),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"MODEL: {model_type.upper()}")
    logger.info(f"  Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    logger.info(f"\n{metrics['report']}")

    return best_pipeline, metrics


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_model(pipeline: Pipeline, model_type: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{model_type}_pipeline.joblib"
    joblib.dump(pipeline, path)
    logger.info(f"Model saved → {path}")
    return path


def load_model(model_type: str) -> Pipeline:
    path = MODELS_DIR / f"{model_type}_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at {path}. Train first.")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train AFMIP ML model")
    parser.add_argument(
        "--model", choices=["rf", "xgboost", "both"], default="both",
        help="Which model to train (default: both, pick best)"
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip hyperparameter search (faster, useful for testing)"
    )
    parser.add_argument(
        "--features-path", default="data/datasets/features.parquet",
        help="Path to features.parquet"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_parquet(args.features_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded features: {df.shape}")

    tune = not args.no_tune
    results = {}

    models_to_train = ["rf", "xgboost"] if args.model == "both" else [args.model]

    for model_type in models_to_train:
        if model_type == "xgboost" and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not installed — skipping. Run: pip install xgboost")
            continue
        pipeline, metrics = train_model(df, model_type=model_type, tune=tune)
        save_model(pipeline, model_type)
        results[model_type] = metrics

    # Auto-select best by ROC-AUC
    if len(results) > 1:
        best = max(results, key=lambda k: results[k]["roc_auc"])
        logger.info(f"\n✅ Best model: {best.upper()} (ROC-AUC {results[best]['roc_auc']:.4f})")


if __name__ == "__main__":
    main()
