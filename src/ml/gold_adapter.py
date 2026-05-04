"""
AFMIP — Gold Layer Adapter
===========================
Connects Person 1's gold_dataset.csv to the ML training pipeline.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

GOLD_FEATURE_COLS = [
    "fft_1", "fft_2", "fft_3",
    "taylor_1", "taylor_2",
    "cheb_1", "cheb_2",
    "returns", "rolling_7", "rolling_14", "volatility",
    "article_count", "avg_length",
    "kw_earnings", "kw_growth", "kw_crisis",
]

ALL_FEATURE_COLS = GOLD_FEATURE_COLS + [
    "rsi_14", "macd", "macd_signal",
    "volume_ratio", "high_low_range", "close_open_diff",
]

ALL_TARGET_COLS = {
    "target_1d":   "target_1d",
    "target_5d":   "target_5d",
    "target_21d":  "target_21d",
    "target_63d":  "target_63d",
    "target_252d": "target_252d",
}

TARGET_HORIZON_DAYS = {
    "target_1d":   1,
    "target_5d":   5,
    "target_21d":  21,
    "target_63d":  63,
    "target_252d": 252,
}

HORIZON_LABELS = {
    "target_1d":   "Next day",
    "target_5d":   "1 week",
    "target_21d":  "1 month",
    "target_63d":  "3 months",
    "target_252d": "1 year",
}


def _add_extra_features(df):
    g = df.groupby("ticker")
    delta    = g["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df["ticker"]).transform(lambda x: x.rolling(14).mean())
    avg_loss = loss.groupby(df["ticker"]).transform(lambda x: x.rolling(14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = g["close"].transform(lambda x: x.ewm(span=12).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26).mean())
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda x: x.ewm(span=9).mean())

    vol_sma = g["volume"].transform(lambda x: x.rolling(5).mean())
    df["volume_ratio"]   = (df["volume"] / vol_sma.replace(0, np.nan)).fillna(1.0)
    df["high_low_range"]  = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["close_open_diff"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    return df


def _add_extra_targets(df):
    for col, n in TARGET_HORIZON_DAYS.items():
        if col == "target_1d":
            continue
        future_close = df.groupby("ticker")["close"].shift(-n)
        df[col] = np.where(
            future_close.notna(),
            future_close.gt(df["close"]).astype(float),
            np.nan,
        )
    return df


def validate_gold_columns(df: pd.DataFrame) -> None:
    required = {
        "date", "ticker", "open", "high", "low", "close", "volume",
        *GOLD_FEATURE_COLS, "target",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "gold dataset is missing required columns: "
            + ", ".join(missing)
        )


def adapt_gold_dataset(
    gold_path="data/datasets/gold_dataset.csv",
    output_path="data/datasets/features.parquet",
):
    logger.info(f"Loading {gold_path} ...")
    df = pd.read_csv(gold_path)
    validate_gold_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info(f"Shape: {df.shape} | Tickers: {df['ticker'].nunique()}")

    df["target_1d"] = pd.to_numeric(df["target"], errors="coerce")
    df = _add_extra_features(df)
    df = _add_extra_targets(df)

    required = ALL_FEATURE_COLS
    before = len(df)
    df = df.dropna(subset=required)
    logger.info(f"Dropped {before - len(df):,} NaN rows → {len(df):,} remain")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved → {output_path}")

    print(f"\n{'='*55}")
    print(f"  GOLD DATASET READY FOR TRAINING")
    print(f"{'='*55}")
    print(f"  Rows:     {len(df):,}")
    print(f"  Tickers:  {df['ticker'].nunique()}")
    print(f"  Features: {len(ALL_FEATURE_COLS)}")
    print(f"  Dates:    {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n  Target balance:")
    for col, src in ALL_TARGET_COLS.items():
        label = HORIZON_LABELS[col]
        pct   = df[src].mean() if src in df.columns else 0
        print(f"    {label:<14}: {pct:.1%} UP")
    print(f"{'='*55}\n")
    return df


def ensure_prepared_features(
    features_path: str = "data/datasets/features.parquet",
    gold_path: str = "data/datasets/gold_dataset.csv",
    rebuild: bool = False,
) -> Path:
    """
    Ensure the prepared features parquet exists for the gold dataset workflow.
    """
    features = Path(features_path)
    gold = Path(gold_path)

    if rebuild or not features.exists():
        if not gold.exists():
            raise FileNotFoundError(
                f"Prepared features not found at {features} and gold dataset not found at {gold}."
            )
        logger.info(f"Preparing features from {gold} ...")
        adapt_gold_dataset(gold_path=str(gold), output_path=str(features))

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-path",   default="data/datasets/gold_dataset.csv")
    parser.add_argument("--output-path", default="data/datasets/features.parquet")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    adapt_gold_dataset(args.gold_path, args.output_path)

if __name__ == "__main__":
    main()
