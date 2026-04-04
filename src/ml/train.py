"""
AFMIP — Multi-Horizon Model Training (Startup Grade)
======================================================
- Trains RF, XGBoost, LightGBM per horizon
- Auto-selects best model per horizon by ROC-AUC
- Saves versioned models with full metadata
- Supports confidence threshold tuning

Usage:
    python -m src.ml.train                    # all horizons
    python -m src.ml.train --horizon 1d       # single horizon
    python -m src.ml.train --no-tune          # skip hyperparameter search
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.ml.features import FEATURE_COLS, TARGET_COLS, HORIZON_LABELS

logger = logging.getLogger(__name__)
MODELS_DIR = Path("data/models")
REGISTRY_FILE = MODELS_DIR / "registry.json"


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])

def _build_xgb():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )),
    ])

def _build_lgbm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )),
    ])

BUILDERS = {"rf": _build_rf, "xgboost": _build_xgb, "lgbm": _build_lgbm}

PARAM_GRIDS = {
    "rf": {
        "model__n_estimators":     [200, 300, 500],
        "model__max_depth":        [6, 8, 10, None],
        "model__min_samples_leaf": [10, 20, 40],
    },
    "xgboost": {
        "model__n_estimators":  [200, 300, 500],
        "model__max_depth":     [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample":     [0.7, 0.8, 1.0],
    },
    "lgbm": {
        "model__n_estimators":  [300, 500, 700],
        "model__num_leaves":    [31, 63, 127],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth":     [4, 6, 8],
    },
}


# ---------------------------------------------------------------------------
# Model versioning / registry
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}


def _save_registry(registry: dict):
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def _next_version(registry: dict, horizon: str) -> str:
    existing = [
        v for k, v in registry.items()
        if k.startswith(horizon)
    ]
    return f"v{len(existing) + 1}"


# ---------------------------------------------------------------------------
# Confidence threshold calibration
# ---------------------------------------------------------------------------

def _find_confidence_threshold(
    pipeline, X_test, y_test, min_precision: float = 0.54
) -> float:
    """
    Find the lowest confidence threshold that achieves min_precision on UP signals.
    Falls back to 0.52 if not achievable.
    """
    probs = pipeline.predict_proba(X_test)[:, 1]
    for threshold in np.arange(0.70, 0.50, -0.02):
        mask = probs >= threshold
        if mask.sum() < 100:
            continue
        prec = precision_score(y_test[mask], (probs[mask] >= 0.5).astype(int), zero_division=0)
        if prec >= min_precision:
            return round(float(threshold), 2)
    return 0.52


# ---------------------------------------------------------------------------
# Train one model for one horizon
# ---------------------------------------------------------------------------

def train_one(
    df: pd.DataFrame,
    target_col: str,
    model_type: str = "lgbm",
    tune: bool = True,
    n_iter: int = 15,
) -> tuple[Pipeline, dict]:

    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].unique()
    cutoff = dates[int(len(dates) * 0.8)]

    train = df[df["date"] < cutoff]
    test  = df[df["date"] >= cutoff]

    X_train = train[FEATURE_COLS]
    y_train = train[target_col]
    X_test  = test[FEATURE_COLS]
    y_test  = test[target_col]

    logger.info(
        f"  [{target_col}] train={len(train):,} test={len(test):,} "
        f"UP%={y_train.mean():.1%}"
    )

    if model_type not in BUILDERS:
        raise ValueError(f"Unknown model: {model_type}")

    pipeline = BUILDERS[model_type]()

    if tune:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=PARAM_GRIDS[model_type],
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=5),
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        logger.info(f"  [{target_col}] CV ROC-AUC: {search.best_score_:.4f}")
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    threshold = _find_confidence_threshold(pipeline, X_test, y_test)

    metrics = {
        "horizon":    target_col,
        "model_type": model_type,
        "accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":    round(roc_auc_score(y_test, y_prob), 4),
        "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":         round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confidence_threshold": threshold,
        "train_rows": int(len(train)),
        "test_rows":  int(len(test)),
        "train_from": str(train["date"].min().date()),
        "train_to":   str(train["date"].max().date()),
        "test_from":  str(test["date"].min().date()),
        "test_to":    str(test["date"].max().date()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        f"  [{target_col}] {model_type.upper()} "
        f"acc={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f} "
        f"threshold={threshold}"
    )

    return pipeline, metrics


# ---------------------------------------------------------------------------
# Train all horizons
# ---------------------------------------------------------------------------

def train_all_horizons(
    df: pd.DataFrame,
    tune: bool = True,
    horizons: list | None = None,
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    available = ["rf"]
    if XGBOOST_AVAILABLE:
        available.append("xgboost")
    if LIGHTGBM_AVAILABLE:
        available.append("lgbm")

    logger.info(f"Available models: {available}")

    targets = horizons or list(TARGET_COLS.keys())
    registry = _load_registry()
    final = {}

    for target_col in targets:
        label = HORIZON_LABELS[target_col]
        logger.info(f"\n{'='*55}\nHORIZON: {label} ({target_col})\n{'='*55}")

        best_pipeline = None
        best_auc      = 0
        best_metrics  = None

        for model_type in available:
            try:
                pipeline, metrics = train_one(df, target_col, model_type, tune)
                if metrics["roc_auc"] > best_auc:
                    best_auc      = metrics["roc_auc"]
                    best_pipeline = pipeline
                    best_metrics  = metrics
            except Exception as e:
                logger.warning(f"  [{target_col}] {model_type} failed: {e}")

        if best_pipeline and best_metrics:
            version = _next_version(registry, target_col)
            model_key = f"{target_col}_{version}"

            # Save model file
            model_path = MODELS_DIR / f"model_{target_col}.joblib"
            archive_path = MODELS_DIR / f"model_{model_key}.joblib"
            joblib.dump(best_pipeline, model_path)
            joblib.dump(best_pipeline, archive_path)

            # Save metrics sidecar
            metrics_path = MODELS_DIR / f"model_{target_col}.json"
            metrics_path.write_text(json.dumps(best_metrics, indent=2))

            # Update registry
            registry[model_key] = {
                **best_metrics,
                "version": version,
                "model_file": str(model_path),
                "is_current": True,
            }
            # Mark old versions as not current
            for k in registry:
                if k.startswith(target_col) and k != model_key:
                    registry[k]["is_current"] = False

            _save_registry(registry)

            logger.info(
                f"\n  BEST [{label}]: {best_metrics['model_type'].upper()} "
                f"AUC={best_auc:.4f} threshold={best_metrics['confidence_threshold']} "
                f"version={version}"
            )
            final[target_col] = best_metrics

    return final


def load_model(target_col: str) -> tuple:
    """Returns (pipeline, metrics_dict)."""
    model_path   = MODELS_DIR / f"model_{target_col}.joblib"
    metrics_path = MODELS_DIR / f"model_{target_col}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No model for {target_col}. Run: python -m src.ml.train")

    pipeline = joblib.load(model_path)
    metrics  = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return pipeline, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon",
        choices=["1d", "5d", "21d", "63d", "252d", "all"],
        default="all",
    )
    parser.add_argument("--no-tune",        action="store_true")
    parser.add_argument("--features-path",  default="data/datasets/features.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not LIGHTGBM_AVAILABLE:
        logger.warning("LightGBM missing. Run: pip install lightgbm")
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost missing. Run: pip install xgboost")

    df = pd.read_parquet(args.features_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded: {df.shape}")

    horizons = (
        [f"target_{args.horizon}"] if args.horizon != "all"
        else list(TARGET_COLS.keys())
    )

    results = train_all_horizons(df, tune=not args.no_tune, horizons=horizons)

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'Horizon':<14} {'Model':<10} {'Accuracy':>10} {'ROC-AUC':>10} {'Threshold':>10}")
    print(f"{'-'*70}")
    for h, m in results.items():
        label = HORIZON_LABELS[h]
        print(
            f"{label:<14} {m['model_type'].upper():<10} "
            f"{m['accuracy']:>10.4f} {m['roc_auc']:>10.4f} "
            f"{m['confidence_threshold']:>10.2f}"
        )
    print(f"{'='*70}")
    print(f"\nModels saved → {MODELS_DIR}/")
    print(f"Registry    → {REGISTRY_FILE}")


if __name__ == "__main__":
    main()
