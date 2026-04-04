"""
AFMIP — Stage 3: Inference / Prediction
=========================================
Load a trained model and generate UP/DOWN signals for new data.

Usage:
    python -m src.ml.predict --ticker AAPL
    python -m src.ml.predict --ticker AAPL MSFT GOOGL --model xgboost
    python -m src.ml.predict --all --top 20
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.ml.features import FEATURE_COLS, build_features
from src.ml.train import load_model

logger = logging.getLogger(__name__)


def predict_latest(
    tickers: list[str] | None = None,
    model_type: str = "rf",
    features_path: str = "data/datasets/features.parquet",
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Load the most recent feature row for each ticker and predict next-day direction.

    Returns a DataFrame with columns:
        ticker, date, signal (UP/DOWN), confidence (probability of UP)
    """
    pipeline = load_model(model_type)

    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])

    # Most recent row per ticker
    latest = df.sort_values("date").groupby("ticker").last().reset_index()

    if tickers:
        tickers_upper = [t.upper() for t in tickers]
        missing = set(tickers_upper) - set(latest["ticker"])
        if missing:
            logger.warning(f"Tickers not found in features: {missing}")
        latest = latest[latest["ticker"].isin(tickers_upper)]

    if latest.empty:
        logger.error("No matching tickers found.")
        return pd.DataFrame()

    X = latest[FEATURE_COLS]
    latest = latest.copy()
    latest["confidence"]  = pipeline.predict_proba(X)[:, 1]   # P(UP)
    latest["signal"]      = latest["confidence"].apply(lambda p: "UP" if p >= 0.5 else "DOWN")

    out = latest[["ticker", "date", "signal", "confidence"]].sort_values(
        "confidence", ascending=False
    ).reset_index(drop=True)

    if top_n:
        out = out.head(top_n)

    return out


def main():
    parser = argparse.ArgumentParser(description="AFMIP: predict next-day direction")
    parser.add_argument("--ticker", nargs="+", help="Specific ticker(s) e.g. AAPL MSFT")
    parser.add_argument("--all",  action="store_true", help="Predict for all tickers")
    parser.add_argument("--top",  type=int, default=None, help="Show only top N by confidence")
    parser.add_argument("--model", choices=["rf", "xgboost"], default="rf")
    parser.add_argument("--features-path", default="data/datasets/features.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tickers = args.ticker if not args.all else None

    results = predict_latest(
        tickers=tickers,
        model_type=args.model,
        features_path=args.features_path,
        top_n=args.top,
    )

    if results.empty:
        print("No predictions generated.")
        return

    print(f"\n📈  AFMIP — Next-Day Predictions ({args.model.upper()})")
    print(f"{'─'*52}")
    print(f"{'Ticker':<10} {'Date':<14} {'Signal':<8} {'Confidence':>10}")
    print(f"{'─'*52}")
    for _, row in results.iterrows():
        arrow = "↑" if row["signal"] == "UP" else "↓"
        print(
            f"{row['ticker']:<10} {str(row['date'].date()):<14} "
            f"{arrow} {row['signal']:<6} {row['confidence']:>10.1%}"
        )
    print(f"{'─'*52}\n")


if __name__ == "__main__":
    main()
