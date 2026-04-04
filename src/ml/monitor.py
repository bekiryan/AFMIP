"""
AFMIP — Model Monitoring
=========================
Checks model performance on recent data weekly.
If accuracy degrades below threshold, triggers alert + retraining.

What it checks:
  - Accuracy on last 30 days of actual outcomes
  - Confidence calibration (are high-confidence calls actually right?)
  - Data drift (are features shifting vs training distribution?)

Usage:
    python -m src.ml.monitor               # check all models
    python -m src.ml.monitor --horizon 1d  # check specific horizon
    python -m src.ml.monitor --fix         # retrain any degraded models
"""

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.ml.features import FEATURE_COLS, TARGET_COLS, HORIZON_LABELS
from src.ml.train import load_model, train_all_horizons

logger = logging.getLogger(__name__)

MODELS_DIR  = Path("data/models")
MONITOR_DIR = Path("data/monitoring")

# Thresholds — if performance drops below these, alert fires
MIN_ACCURACY  = 0.50   # must beat random
MIN_ROC_AUC   = 0.50
DRIFT_Z_SCORE = 3.0    # feature mean shifted by 3 std devs = drift


# ---------------------------------------------------------------------------
# Performance check
# ---------------------------------------------------------------------------

def check_recent_performance(
    df: pd.DataFrame,
    target_col: str,
    lookback_days: int = 30,
) -> dict:
    """
    Evaluate model on the most recent `lookback_days` of data.
    Compares against training-time performance.
    """
    df = df.sort_values("date").reset_index(drop=True)
    cutoff = df["date"].max() - timedelta(days=lookback_days)
    recent = df[df["date"] >= cutoff].dropna(subset=FEATURE_COLS + [target_col])

    if len(recent) < 100:
        return {"status": "INSUFFICIENT_DATA", "rows": len(recent)}

    try:
        pipeline, train_metrics = load_model(target_col)
    except FileNotFoundError:
        return {"status": "NO_MODEL"}

    X = recent[FEATURE_COLS]
    y = recent[target_col]

    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    current_acc = accuracy_score(y, y_pred)
    current_auc = roc_auc_score(y, y_prob)

    train_acc = train_metrics.get("accuracy", 0)
    train_auc = train_metrics.get("roc_auc", 0)

    acc_drop = train_acc - current_acc
    auc_drop = train_auc - current_auc

    # Confidence calibration: for predictions above threshold,
    # what fraction were actually correct?
    threshold = train_metrics.get("confidence_threshold", 0.52)
    high_conf_mask = y_prob >= threshold
    if high_conf_mask.sum() > 10:
        calibration_acc = accuracy_score(
            y[high_conf_mask],
            y_pred[high_conf_mask]
        )
    else:
        calibration_acc = None

    degraded = (
        current_acc < MIN_ACCURACY or
        current_auc < MIN_ROC_AUC or
        acc_drop > 0.03 or
        auc_drop > 0.03
    )

    return {
        "status":           "DEGRADED" if degraded else "OK",
        "horizon":          target_col,
        "label":            HORIZON_LABELS[target_col],
        "rows_evaluated":   int(len(recent)),
        "lookback_days":    lookback_days,
        "current_accuracy": round(current_acc, 4),
        "current_roc_auc":  round(current_auc, 4),
        "train_accuracy":   round(train_acc, 4),
        "train_roc_auc":    round(train_auc, 4),
        "accuracy_drop":    round(acc_drop, 4),
        "auc_drop":         round(auc_drop, 4),
        "calibration_acc":  round(calibration_acc, 4) if calibration_acc else None,
        "checked_at":       datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Data drift detection
# ---------------------------------------------------------------------------

def check_data_drift(
    df: pd.DataFrame,
    lookback_days: int = 30,
) -> dict:
    """
    Compare feature distributions of recent data vs training data.
    Uses z-score to detect large shifts.
    """
    df = df.sort_values("date")
    cutoff = df["date"].max() - timedelta(days=lookback_days)

    train_df  = df[df["date"] < cutoff][FEATURE_COLS].dropna()
    recent_df = df[df["date"] >= cutoff][FEATURE_COLS].dropna()

    if len(recent_df) < 50:
        return {"status": "INSUFFICIENT_DATA"}

    drifted = []
    for col in FEATURE_COLS:
        train_mean  = train_df[col].mean()
        train_std   = train_df[col].std()
        recent_mean = recent_df[col].mean()

        if train_std > 0:
            z = abs(recent_mean - train_mean) / train_std
            if z > DRIFT_Z_SCORE:
                drifted.append({"feature": col, "z_score": round(z, 2)})

    return {
        "status":          "DRIFT_DETECTED" if drifted else "OK",
        "drifted_features": drifted,
        "total_features":   len(FEATURE_COLS),
        "checked_at":       datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Save monitoring report
# ---------------------------------------------------------------------------

def save_report(report: dict):
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    path = MONITOR_DIR / f"monitor_{ts}.json"
    path.write_text(json.dumps(report, indent=2))
    logger.info(f"Report saved → {path}")

    # Also keep a latest.json for easy access
    latest = MONITOR_DIR / "latest.json"
    latest.write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(report: dict):
    print(f"\n{'='*60}")
    print(f"  AFMIP Model Monitor  |  {report['checked_at'][:10]}")
    print(f"{'='*60}")

    for horizon, result in report.get("horizons", {}).items():
        status  = result.get("status", "?")
        label   = result.get("label", horizon)
        icon    = "OK" if status == "OK" else "!!"

        print(f"\n  [{icon}] {label} ({horizon})")
        if status == "INSUFFICIENT_DATA":
            print(f"      Not enough recent data to evaluate.")
            continue
        if status == "NO_MODEL":
            print(f"      No trained model found.")
            continue

        print(f"      Accuracy:  {result['current_accuracy']:.4f}  "
              f"(was {result['train_accuracy']:.4f}, "
              f"drop={result['accuracy_drop']:+.4f})")
        print(f"      ROC-AUC:   {result['current_roc_auc']:.4f}  "
              f"(was {result['train_roc_auc']:.4f}, "
              f"drop={result['auc_drop']:+.4f})")
        if result.get("calibration_acc"):
            print(f"      High-conf accuracy: {result['calibration_acc']:.4f}")
        if status == "DEGRADED":
            print(f"      ** MODEL DEGRADED — consider retraining **")

    drift = report.get("drift", {})
    print(f"\n  Data drift: {drift.get('status', '?')}")
    if drift.get("drifted_features"):
        for f in drift["drifted_features"][:5]:
            print(f"    {f['feature']}: z={f['z_score']}")

    print(f"\n{'='*60}\n")


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
    parser.add_argument("--fix",           action="store_true",
                        help="Retrain degraded models automatically")
    parser.add_argument("--lookback",      type=int, default=30)
    parser.add_argument("--features-path", default="data/datasets/features.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_parquet(args.features_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded features: {df.shape}")

    horizons = (
        [f"target_{args.horizon}"] if args.horizon != "all"
        else list(TARGET_COLS.keys())
    )

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "horizons":   {},
        "drift":      check_data_drift(df, lookback_days=args.lookback),
    }

    degraded = []
    for target_col in horizons:
        result = check_recent_performance(df, target_col, args.lookback)
        report["horizons"][target_col] = result
        if result.get("status") == "DEGRADED":
            degraded.append(target_col)

    print_report(report)
    save_report(report)

    if degraded:
        logger.warning(f"Degraded models: {degraded}")
        if args.fix:
            logger.info("Retraining degraded models ...")
            train_all_horizons(df, tune=False, horizons=degraded)
            logger.info("Retraining complete.")
        else:
            print(f"  Run with --fix to retrain automatically.\n")


if __name__ == "__main__":
    main()
