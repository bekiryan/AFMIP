"""
AFMIP — Feature Engineering (Multi-Horizon)
=============================================
Builds feature matrix with targets for 5 time horizons:
  - 1 day, 5 days, 21 days, 63 days, 252 days (1 year)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "return_1d", "return_3d", "return_5d", "return_10d", "return_21d",
    "sma_ratio_5_20", "sma_ratio_5_50", "sma_ratio_20_50",
    "price_vs_52w_high", "price_vs_52w_low",
    "volatility_5d", "volatility_21d", "volatility_ratio",
    "volume_ratio_5d", "volume_ratio_21d",
    "high_low_range", "close_open_diff",
    "rsi_14", "rsi_28",
    "macd", "macd_signal",
    "bb_position", "bb_width",
    "sentiment_mean", "sentiment_std", "news_count", "sentiment_momentum",
]

TARGET_COLS = {
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


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ticker")

    for n in [1, 3, 5, 10, 21]:
        df[f"return_{n}d"] = g["close"].pct_change(n)

    for w in [5, 20, 50]:
        df[f"sma_{w}"] = g["close"].transform(lambda x: x.rolling(w).mean())

    df["sma_ratio_5_20"]  = df["sma_5"]  / df["sma_20"]
    df["sma_ratio_5_50"]  = df["sma_5"]  / df["sma_50"]
    df["sma_ratio_20_50"] = df["sma_20"] / df["sma_50"]

    df["52w_high"] = g["close"].transform(lambda x: x.rolling(252).max())
    df["52w_low"]  = g["close"].transform(lambda x: x.rolling(252).min())
    df["price_vs_52w_high"] = df["close"] / df["52w_high"]
    df["price_vs_52w_low"]  = df["close"] / df["52w_low"]

    df["volatility_5d"]    = g["return_1d"].transform(lambda x: x.rolling(5).std())
    df["volatility_21d"]   = g["return_1d"].transform(lambda x: x.rolling(21).std())
    df["volatility_ratio"] = df["volatility_5d"] / df["volatility_21d"]

    df["volume_sma_5"]     = g["volume"].transform(lambda x: x.rolling(5).mean())
    df["volume_sma_21"]    = g["volume"].transform(lambda x: x.rolling(21).mean())
    df["volume_ratio_5d"]  = df["volume"] / df["volume_sma_5"]
    df["volume_ratio_21d"] = df["volume"] / df["volume_sma_21"]

    df["high_low_range"]  = (df["high"] - df["low"]) / df["close"]
    df["close_open_diff"] = (df["close"] - df["open"]) / df["open"]

    for period in [14, 28]:
        delta    = g["close"].diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.groupby(df["ticker"]).transform(lambda x: x.rolling(period).mean())
        avg_loss = loss.groupby(df["ticker"]).transform(lambda x: x.rolling(period).mean())
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    ema12 = g["close"].transform(lambda x: x.ewm(span=12).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26).mean())
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda x: x.ewm(span=9).mean())

    # Bollinger Bands (20-day)
    sma20 = g["close"].transform(lambda x: x.rolling(20).mean())
    std20 = g["close"].transform(lambda x: x.rolling(20).std())
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-9)
    df["bb_width"]    = (upper - lower) / sma20

    return df


def _add_sentiment_features(stock_df, news_df):
    if "sentiment_score" not in news_df.columns:
        logger.warning("No sentiment_score column — filling with zeros.")
        for col in ["sentiment_mean", "sentiment_std", "news_count", "sentiment_momentum"]:
            stock_df[col] = 0.0
        return stock_df

    daily = (
        news_df.groupby(["date", "ticker"])["sentiment_score"]
        .agg(sentiment_mean="mean", sentiment_std="std", news_count="count")
        .reset_index()
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    merged = stock_df.merge(daily, on=["date", "ticker"], how="left")
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0)
    merged["sentiment_std"]  = merged["sentiment_std"].fillna(0)
    merged["news_count"]     = merged["news_count"].fillna(0)
    merged["sentiment_momentum"] = (
        merged["sentiment_mean"]
        - merged.groupby("ticker")["sentiment_mean"].transform(lambda x: x.rolling(5).mean())
    ).fillna(0)
    return merged


def _add_targets(df):
    for col, n in TARGET_COLS.items():
        df[col] = (
            df.groupby("ticker")["close"].shift(-n).gt(df["close"]).astype(int)
        )
    return df


def build_features(
    stocks_path="data/datasets/stocks.parquet",
    news_path="data/datasets/news.parquet",
    output_path="data/datasets/features.parquet",
):
    logger.info("Loading stocks ...")
    stocks = pd.read_parquet(stocks_path)
    stocks["date"] = pd.to_datetime(stocks["date"])
    stocks = stocks.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info("Loading news ...")
    news = pd.read_parquet(news_path)
    news["date"] = pd.to_datetime(news["date"])

    logger.info("Building price features ...")
    stocks = _add_price_features(stocks)

    logger.info("Building sentiment features ...")
    stocks = _add_sentiment_features(stocks, news)

    logger.info("Adding targets ...")
    stocks = _add_targets(stocks)

    required = FEATURE_COLS + list(TARGET_COLS.keys())
    before = len(stocks)
    stocks = stocks.dropna(subset=required)
    logger.info(f"Dropped {before - len(stocks):,} NaN rows → {len(stocks):,} remain")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    stocks.to_parquet(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return stocks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = build_features()
    print(f"\nShape: {df.shape}  |  Features: {len(FEATURE_COLS)}")
    for col in TARGET_COLS:
        print(f"  {HORIZON_LABELS[col]:>12}: {df[col].mean():.1%} UP")
