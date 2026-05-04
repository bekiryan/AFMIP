import argparse
import json
import logging
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ml.gold_adapter import ALL_FEATURE_COLS as FEATURE_COLS
from src.ml.gold_adapter import ALL_TARGET_COLS as TARGET_COLS
from src.ml.gold_adapter import HORIZON_LABELS, ensure_prepared_features
from src.ml.train import load_model

logger = logging.getLogger(__name__)
EXPORTS_DIR = Path("data/exports")
MODELS_DIR  = Path("data/models")


def get_trained_horizons() -> list:
    """Return only horizons that have a saved model file."""
    trained = []
    for target_col in TARGET_COLS:
        model_path = MODELS_DIR / f"model_{target_col}.joblib"
        if model_path.exists():
            trained.append(target_col)
    return trained


def predict_all_horizons(
    tickers=None,
    features_path="data/datasets/features.parquet",
    gold_path="data/datasets/gold_dataset.csv",
    top_n=None,
    only_confident=True,
    rebuild_features=False,
):
    features_path = ensure_prepared_features(
        features_path=features_path,
        gold_path=gold_path,
        rebuild=rebuild_features,
    )
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
        return pd.DataFrame(), {}

    X = latest[FEATURE_COLS]
    results = latest[["ticker", "date"]].copy()
    model_meta = {}

    # Only loop over trained horizons
    trained = get_trained_horizons()
    if not trained:
        logger.error("No trained models found. Run: python -m src.ml.train --horizon 1d")
        return pd.DataFrame(), {}

    for target_col in trained:
        pipeline, metrics = load_model(target_col)
        threshold = metrics.get("confidence_threshold", 0.52)
        model_meta[target_col] = metrics

        X_df = pd.DataFrame(X.values, columns=FEATURE_COLS)
        probs = pipeline.predict_proba(X_df)[:, 1]

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

    if "confidence_target_1d" in results.columns:
        results = results.sort_values("confidence_target_1d", ascending=False)

    if top_n:
        results = results.head(top_n)

    return results.reset_index(drop=True), model_meta


def export_signals(results: pd.DataFrame, model_meta: dict, fmt: str = "csv") -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    trained = list(model_meta.keys())
    rows = []
    for _, row in results.iterrows():
        base = {
            "ticker":      row["ticker"],
            "date":        str(row["date"].date()),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        for target_col in trained:
            label = HORIZON_LABELS[target_col].replace(" ", "_")
            base[f"signal_{label}"]     = row.get(f"signal_{target_col}", "")
            base[f"confidence_{label}"] = row.get(f"confidence_{target_col}", "")
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

    trained  = list(model_meta.keys())
    labels   = [HORIZON_LABELS[t] for t in trained]
    date_str = str(results["date"].iloc[0].date())

    # Dynamic width based on number of trained horizons
    col_w   = 14
    total_w = 10 + col_w * len(trained)

    print(f"\n{'='*total_w}")
    print(f"  AFMIP Signal Dashboard  |  {date_str}")
    print(f"{'='*total_w}")

    header = f"{'Ticker':<8}  " + "  ".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print(f"{'-'*total_w}")

    for _, row in results.iterrows():
        cells = []
        for target_col in trained:
            sig  = row.get(f"signal_{target_col}", "—")
            conf = row.get(f"confidence_{target_col}")
            thr  = row.get(f"threshold_{target_col}", 0.52)

            if conf is None:
                cells.append(f"{'—':{col_w}}")
            elif sig == "HOLD":
                cells.append(f"{'HOLD':{col_w}}")
            else:
                marker = "*" if conf >= thr else " "
                cells.append(f"{sig} {conf:.0%}{marker}".rjust(col_w))

        print(f"{row['ticker']:<8}  " + "  ".join(cells))

    print(f"{'='*total_w}")
    print("  * = above confidence threshold  |  HOLD = model uncertain, skip trade")
    print(f"{'='*total_w}")

    print(f"\n  Trained models:")
    for target_col, meta in model_meta.items():
        label = HORIZON_LABELS[target_col]
        print(
            f"    {label:<14} {meta.get('model_type','?'). upper():<8} "
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
    parser.add_argument("--all-signals",   action="store_true")
    parser.add_argument("--features-path", default="data/datasets/features.parquet")
    parser.add_argument("--gold-path",     default="data/datasets/gold_dataset.csv")
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tickers = args.ticker if not args.all else None
    results, model_meta = predict_all_horizons(
        tickers=tickers,
        features_path=args.features_path,
        gold_path=args.gold_path,
        top_n=args.top,
        only_confident=not args.all_signals,
        rebuild_features=args.rebuild_features,
    )

    print_predictions(results, model_meta)

    if args.export:
        path = export_signals(results, model_meta, fmt=args.export)
        print(f"  Exported → {path}\n")


if __name__ == "__main__":
    main()
