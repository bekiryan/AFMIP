"""
AFMIP — Multi-Horizon Prediction (Startup Grade)
==================================================
- Predicts UP/DOWN across all 5 horizons
- Uses per-model confidence thresholds (from training)
- Exports clean CSV/JSON for Power BI and teammates
- Shows model metadata with each prediction

Usage:
    python -m src.ml.predict --ticker AAPL MSFT
    python -m src.ml.predict --all --top 20
    python -m src.ml.predict --all --export csv
    python -m src.ml.predict --all --export json
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ml.features import FEATURE_COLS, TARGET_COLS, HORIZON_LABELS
from src.ml.train import load_model

logger = logging.getLogger(__name__)
EXPORTS_DIR = Path("data/exports")


def predict_all_horizons(
    tickers=None,
    features_path="data/datasets/features.parquet",
    top_n=None,
    only_confident=True,
):
    """
    Generate predictions for all horizons.

    Parameters
    ----------
    tickers : list or None
        Specific tickers, or None for all.
    only_confident : bool
        If True, show NO_SIGNAL when confidence < model threshold.
    """
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values("date").groupby("ticker").last().reset_index()

    if tickers:
        tickers_upper = [t.upper() for t in tickers]
        missing = set(tickers_upper) - set(latest["ticker"])
        if missing:
            logger.warning(f"Tickers not found: {missing}")
        latest = latest[latest["ticker"].isin(tickers_upper)]

    if latest.empty:
        return pd.DataFrame()

    X = latest[FEATURE_COLS]
    results = latest[["ticker", "date"]].copy()

    model_meta = {}

    for target_col in TARGET_COLS:
        try:
            pipeline, metrics = load_model(target_col)
            threshold = metrics.get("confidence_threshold", 0.52)
            model_meta[target_col] = metrics

            probs = pipeline.predict_proba(X)[:, 1]

            signals = []
            for p in probs:
                if only_confident and p < threshold and p > (1 - threshold):
                    signals.append("HOLD")
                elif p >= 0.5:
                    signals.append("UP")
                else:
                    signals.append("DOWN")

            results[f"signal_{target_col}"]     = signals
            results[f"confidence_{target_col}"] = probs.round(4)
            results[f"threshold_{target_col}"]  = threshold

        except FileNotFoundError:
            logger.warning(f"No model for {target_col} — run train.py first")
            results[f"signal_{target_col}"]     = "NO_MODEL"
            results[f"confidence_{target_col}"] = None
            results[f"threshold_{target_col}"]  = None

    # Sort by 1-day confidence
    if "confidence_target_1d" in results.columns:
        results = results.sort_values("confidence_target_1d", ascending=False)

    if top_n:
        results = results.head(top_n)

    return results.reset_index(drop=True), model_meta


def export_signals(results: pd.DataFrame, fmt: str = "csv") -> Path:
    """Export predictions to CSV or JSON for Power BI / teammates."""
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    # Build clean export DataFrame
    rows = []
    for _, row in results.iterrows():
        base = {
            "ticker": row["ticker"],
            "date":   str(row["date"].date()),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        for target_col, label in HORIZON_LABELS.items():
            base[f"signal_{label.replace(' ', '_')}"]     = row.get(f"signal_{target_col}", "")
            base[f"confidence_{label.replace(' ', '_')}"] = row.get(f"confidence_{target_col}", "")
        rows.append(base)

    export_df = pd.DataFrame(rows)

    if fmt == "csv":
        path = EXPORTS_DIR / f"signals_{ts}.csv"
        export_df.to_csv(path, index=False)
    else:
        path = EXPORTS_DIR / f"signals_{ts}.json"
        path.write_text(export_df.to_json(orient="records", indent=2))

    logger.info(f"Exported {len(export_df)} tickers → {path}")
    return path


def print_predictions(results: pd.DataFrame, model_meta: dict):
    if results.empty:
        print("No predictions.")
        return

    date_str = str(results["date"].iloc[0].date())

    # Header
    print(f"\n{'='*78}")
    print(f"  AFMIP Signal Dashboard  |  {date_str}")
    print(f"{'='*78}")
    print(
        f"{'Ticker':<8}  "
        f"{'Next Day':>12}  "
        f"{'1 Week':>12}  "
        f"{'1 Month':>12}  "
        f"{'3 Months':>12}  "
        f"{'1 Year':>12}"
    )
    print(f"{'-'*78}")

    for _, row in results.iterrows():
        cells = []
        for target_col in TARGET_COLS:
            sig  = row.get(f"signal_{target_col}", "—")
            conf = row.get(f"confidence_{target_col}")
            thr  = row.get(f"threshold_{target_col}", 0.52)

            if conf is None or sig == "NO_MODEL":
                cells.append(f"{'—':>12}")
            elif sig == "HOLD":
                cells.append(f"{'HOLD':>12}")
            else:
                marker = "*" if conf >= thr else " "
                cells.append(f"{sig} {conf:.0%}{marker}".rjust(12))

        print(f"{row['ticker']:<8}  {'  '.join(cells)}")

    print(f"{'='*78}")
    print("  * = above confidence threshold  |  HOLD = low confidence, skip trade")
    print(f"{'='*78}")

    # Model metadata footer
    print(f"\n  Model versions:")
    for target_col, label in HORIZON_LABELS.items():
        meta = model_meta.get(target_col, {})
        if meta:
            print(
                f"    {label:<14} {meta.get('model_type','?').upper():<8} "
                f"AUC={meta.get('roc_auc','?')}  "
                f"threshold={meta.get('confidence_threshold','?')}  "
                f"trained={meta.get('trained_at','?')[:10]}"
            )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",        nargs="+")
    parser.add_argument("--all",           action="store_true")
    parser.add_argument("--top",           type=int, default=None)
    parser.add_argument("--export",        choices=["csv", "json"], default=None)
    parser.add_argument("--all-signals",   action="store_true",
                        help="Show all signals, not just high-confidence")
    parser.add_argument("--features-path", default="data/datasets/features.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tickers = args.ticker if not args.all else None
    results, model_meta = predict_all_horizons(
        tickers=tickers,
        features_path=args.features_path,
        top_n=args.top,
        only_confident=not args.all_signals,
    )

    print_predictions(results, model_meta)

    if args.export:
        path = export_signals(results, fmt=args.export)
        print(f"  Exported → {path}\n")


if __name__ == "__main__":
    main()
