"""
AFMIP — Multi-Horizon Model Training (Final Version)
=====================================================
- Trains RF, XGBoost, LightGBM per horizon
- Auto-selects best model by ROC-AUC
- Full-train mode: trains on ALL data (Person 2 provides new data as test)
- Smart horizon detection: skips horizons without enough valid data
- Fixed threshold: always uses 0.52 (not auto-calibrated)

Usage:
    python -m src.ml.train --horizon 1d --full-train
    python -m src.ml.train --horizon 5d --full-train
    python -m src.ml.train --horizon all --full-train
"""

import argparse
import json
import logging
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

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

from src.ml.gold_adapter import ALL_FEATURE_COLS as FEATURE_COLS
from src.ml.gold_adapter import ALL_TARGET_COLS as TARGET_COLS
from src.ml.gold_adapter import HORIZON_LABELS

logger = logging.getLogger(__name__)
MODELS_DIR    = Path("data/models")
REGISTRY_FILE = MODELS_DIR / "registry.json"

# Fixed confidence threshold — 0.52 is the right value for this dataset
CONFIDENCE_THRESHOLD = 0.52


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
# Registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}

def _save_registry(registry: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))

def _next_version(registry: dict, horizon: str) -> str:
    count = sum(1 for k in registry if k.startswith(horizon))
    return f"v{count + 1}"


# ---------------------------------------------------------------------------
# Validate horizon has enough data
# ---------------------------------------------------------------------------

def _validate_horizon(df: pd.DataFrame, target_col: str) -> tuple[bool, str]:
    """
    Check if a horizon has enough valid labeled data to train on.
    Returns (is_valid, reason).
    """
    src_col = TARGET_COLS.get(target_col)
    if not src_col or src_col not in df.columns:
        return False, "column not found"

    valid_df = df[df[src_col].notna()]
    n_valid  = len(valid_df)
    pct      = n_valid / len(df)

    if n_valid < 1000:
        return False, f"only {n_valid} valid rows (need 1000+)"

    if pct < 0.50:
        return False, f"only {pct:.0%} valid rows (need 50%+)"

    classes = valid_df[src_col].unique()
    if len(classes) < 2:
        return False, f"only one class ({classes}) — no variation to learn"

    up_pct = valid_df[src_col].mean()
    if up_pct < 0.05 or up_pct > 0.95:
        return False, f"target too imbalanced (UP={up_pct:.0%})"

    return True, f"{n_valid:,} valid rows ({pct:.0%})"


# ---------------------------------------------------------------------------
# Train one model for one horizon
# ---------------------------------------------------------------------------

def train_one(
    df: pd.DataFrame,
    target_col: str,
    model_type: str = "rf",
    tune: bool = True,
    full_train: bool = True,
    n_iter: int = 15,
) -> tuple:

    src_col = TARGET_COLS[target_col]

    # Use only rows with valid labels
    valid_df = df[df[src_col].notna()].copy()
    valid_df = valid_df.sort_values("date").reset_index(drop=True)

    if full_train:
        # Train on ALL valid data
        # Use last 20% for evaluation metrics only (not held out)
        train_df = valid_df
        dates    = valid_df["date"].unique()
        cutoff   = dates[int(len(dates) * 0.8)]
        eval_df  = valid_df[valid_df["date"] >= cutoff]
        logger.info(
            f"  [{target_col}] FULL TRAIN: {len(train_df):,} rows "
            f"({str(valid_df['date'].min().date())} → {str(valid_df['date'].max().date())})"
        )
    else:
        # Standard 80/20 time split
        dates   = valid_df["date"].unique()
        cutoff  = dates[int(len(dates) * 0.8)]
        train_df = valid_df[valid_df["date"] < cutoff]
        eval_df  = valid_df[valid_df["date"] >= cutoff]
        logger.info(
            f"  [{target_col}] SPLIT: train={len(train_df):,} eval={len(eval_df):,} "
            f"UP%={train_df[src_col].mean():.1%}"
        )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[src_col]

    if len(eval_df) < 50:
        logger.warning(f"  [{target_col}] Eval set too small ({len(eval_df)}), using train for eval")
        eval_df = train_df

    X_eval = pd.DataFrame(eval_df[FEATURE_COLS].values, columns=FEATURE_COLS)
    y_eval = eval_df[src_col]

    pipeline = BUILDERS[model_type]()

    if tune:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=PARAM_GRIDS[model_type],
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        logger.info(f"  [{target_col}] Best CV ROC-AUC: {search.best_score_:.4f}")
    else:
        pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_eval)
    y_prob = pipeline.predict_proba(X_eval)[:, 1]

    # Only compute ROC-AUC if both classes present in eval
    eval_classes = y_eval.unique()
    if len(eval_classes) < 2:
        logger.warning(f"  [{target_col}] Eval has only one class — using train metrics")
        y_pred_t = pipeline.predict(pd.DataFrame(X_train.values, columns=FEATURE_COLS))
        y_prob_t = pipeline.predict_proba(pd.DataFrame(X_train.values, columns=FEATURE_COLS))[:, 1]
        roc_auc  = roc_auc_score(y_train, y_prob_t)
        accuracy = accuracy_score(y_train, y_pred_t)
    else:
        roc_auc  = roc_auc_score(y_eval, y_prob)
        accuracy = accuracy_score(y_eval, y_pred)

    metrics = {
        "horizon":              target_col,
        "label":                HORIZON_LABELS[target_col],
        "model_type":           model_type,
        "full_train":           full_train,
        "accuracy":             round(accuracy, 4),
        "roc_auc":              round(roc_auc, 4),
        "precision":            round(precision_score(y_eval, y_pred, zero_division=0), 4),
        "recall":               round(recall_score(y_eval, y_pred, zero_division=0), 4),
        "f1":                   round(f1_score(y_eval, y_pred, zero_division=0), 4),
        "confidence_threshold": CONFIDENCE_THRESHOLD,  # always fixed at 0.52
        "train_rows":           int(len(train_df)),
        "valid_rows":           int(len(valid_df)),
        "train_from":           str(valid_df["date"].min().date()),
        "train_to":             str(valid_df["date"].max().date()),
        "trained_at":           datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        f"  [{target_col}] {model_type.upper()} "
        f"acc={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f} "
        f"threshold={CONFIDENCE_THRESHOLD}"
    )
    return pipeline, metrics


# ---------------------------------------------------------------------------
# Train all horizons
# ---------------------------------------------------------------------------

def train_all_horizons(
    df: pd.DataFrame,
    tune: bool = True,
    full_train: bool = True,
    horizons: list = None,
) -> dict:

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    available = ["rf"]
    if XGBOOST_AVAILABLE:
        available.append("xgboost")
    if LIGHTGBM_AVAILABLE:
        available.append("lgbm")

    logger.info(f"Models available: {available}")
    logger.info(f"Mode: {'FULL TRAIN' if full_train else '80/20 SPLIT'}")

    targets  = horizons or list(TARGET_COLS.keys())
    registry = _load_registry()
    final    = {}

    for target_col in targets:
        label = HORIZON_LABELS[target_col]
        logger.info(f"\n{'='*55}\nHORIZON: {label}\n{'='*55}")

        # Validate before training
        is_valid, reason = _validate_horizon(df, target_col)
        if not is_valid:
            logger.warning(f"  SKIPPING {label}: {reason}")
            continue

        logger.info(f"  Data OK: {reason}")

        best_pipeline = None
        best_auc      = 0
        best_metrics  = None

        for model_type in available:
            try:
                pipeline, metrics = train_one(
                    df, target_col, model_type,
                    tune=tune, full_train=full_train
                )
                if metrics["roc_auc"] > best_auc:
                    best_auc      = metrics["roc_auc"]
                    best_pipeline = pipeline
                    best_metrics  = metrics
            except Exception as e:
                logger.warning(f"  [{target_col}] {model_type} failed: {e}")

        if best_pipeline and best_metrics:
            version   = _next_version(registry, target_col)
            model_key = f"{target_col}_{version}"

            # Save current model (used by predict.py)
            model_path   = MODELS_DIR / f"model_{target_col}.joblib"
            archive_path = MODELS_DIR / f"model_{model_key}.joblib"
            joblib.dump(best_pipeline, model_path)
            joblib.dump(best_pipeline, archive_path)

            # Save metrics sidecar (used by predict.py)
            metrics_path = MODELS_DIR / f"model_{target_col}.json"
            metrics_path.write_text(json.dumps(best_metrics, indent=2))

            # Update registry
            registry[model_key] = {
                **best_metrics,
                "version":    version,
                "is_current": True,
            }
            for k in registry:
                if k.startswith(target_col) and k != model_key:
                    registry[k]["is_current"] = False
            _save_registry(registry)

            mode_str = "FULL TRAIN" if full_train else "80/20"
            logger.info(
                f"\n  BEST [{label}]: {best_metrics['model_type'].upper()} "
                f"AUC={best_auc:.4f} | {mode_str} | version={version}"
            )
            final[target_col] = best_metrics

    return final


def load_model(target_col: str) -> tuple:
    model_path   = MODELS_DIR / f"model_{target_col}.joblib"
    metrics_path = MODELS_DIR / f"model_{target_col}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No model for {target_col}.")
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
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Train on 100%% of data (production mode)"
    )
    parser.add_argument("--no-tune",       action="store_true")
    parser.add_argument("--features-path", default="data/datasets/features.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not LIGHTGBM_AVAILABLE:
        logger.warning("LightGBM missing. Run: pip install lightgbm")
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost missing. Run: pip install xgboost")

    df = pd.read_parquet(args.features_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded: {df.shape} | {df.date.min().date()} → {df.date.max().date()}")

    if args.full_train:
        logger.info("PRODUCTION MODE — training on 100% of data")

    horizons = (
        [f"target_{args.horizon}"] if args.horizon != "all"
        else list(TARGET_COLS.keys())
    )

    results = train_all_horizons(
        df,
        tune=not args.no_tune,
        full_train=args.full_train,
        horizons=horizons,
    )

    if not results:
        print("\n  No models trained. Check data quality.")
        return

    mode = "FULL TRAIN" if args.full_train else "80/20 SPLIT"
    print(f"\n{'='*72}")
    print(f"  FINAL RESULTS — {mode}")
    print(f"{'='*72}")
    print(f"{'Horizon':<14} {'Model':<10} {'Accuracy':>10} {'ROC-AUC':>10} {'Threshold':>10}")
    print(f"{'-'*72}")
    for h, m in results.items():
        print(
            f"{HORIZON_LABELS[h]:<14} {m['model_type'].upper():<10} "
            f"{m['accuracy']:>10.4f} {m['roc_auc']:>10.4f} "
            f"{m['confidence_threshold']:>10.2f}"
        )
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
