"""
AFMIP — Stage 2 & 3: Feature Engineering
=========================================
Combines OHLCV price data + news sentiment scores into a single
feature matrix ready for model training.

Input:  data/datasets/stocks.parquet
        data/datasets/news.parquet  (with sentiment_score column from Stage 2)
Output: data/datasets/features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Technical indicators (computed from OHLCV)
# ---------------------------------------------------------------------------

def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators derived from OHLCV columns."""

    g = df.groupby("ticker")

    # --- Returns ---
    df["return_1d"] = g["close"].pct_change(1)
    df["return_3d"] = g["close"].pct_change(3)
    df["return_5d"] = g["close"].pct_change(5)

    # --- Moving averages ---
    df["sma_5"]  = g["close"].transform(lambda x: x.rolling(5).mean())
    df["sma_20"] = g["close"].transform(lambda x: x.rolling(20).mean())
    df["sma_ratio"] = df["sma_5"] / df["sma_20"]   # > 1 → short MA above long MA

    # --- Volatility ---
    df["volatility_5d"] = g["return_1d"].transform(lambda x: x.rolling(5).std())

    # --- Volume ---
    df["volume_sma_5"] = g["volume"].transform(lambda x: x.rolling(5).mean())
    df["volume_ratio"] = df["volume"] / df["volume_sma_5"]   # spike vs average

    # --- Candle shape ---
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_diff"] = (df["close"] - df["open"]) / df["open"]

    # --- RSI (14-day) ---
    delta = g["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df["ticker"]).transform(lambda x: x.rolling(14).mean())
    avg_loss = loss.groupby(df["ticker"]).transform(lambda x: x.rolling(14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df


# ---------------------------------------------------------------------------
# Sentiment aggregation (from news.parquet, which must have sentiment_score)
# ---------------------------------------------------------------------------

def _add_sentiment_features(
    stock_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate news sentiment per ticker per day and merge onto price data.

    Expects news_df to have columns: date, ticker, sentiment_score
    sentiment_score should be in [-1, +1] (negative → positive).
    If the column is missing, sentiment features are filled with 0.
    """

    if "sentiment_score" not in news_df.columns:
        logger.warning(
            "sentiment_score column not found in news data. "
            "Sentiment features will be zero. Run Stage 2 (FinBERT) first."
        )
        stock_df["sentiment_mean"] = 0.0
        stock_df["sentiment_std"]  = 0.0
        stock_df["news_count"]     = 0
        return stock_df

    daily_sentiment = (
        news_df.groupby(["date", "ticker"])["sentiment_score"]
        .agg(
            sentiment_mean="mean",
            sentiment_std="std",
            news_count="count",
        )
        .reset_index()
    )

    # Fill NaN std (single article days) with 0
    daily_sentiment["sentiment_std"] = daily_sentiment["sentiment_std"].fillna(0)

    merged = stock_df.merge(daily_sentiment, on=["date", "ticker"], how="left")
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0)
    merged["sentiment_std"]  = merged["sentiment_std"].fillna(0)
    merged["news_count"]     = merged["news_count"].fillna(0)

    return merged


# ---------------------------------------------------------------------------
# Target label: did the stock go UP the next day?
# ---------------------------------------------------------------------------

def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary classification target:
        1 = next-day close HIGHER than today's close
        0 = next-day close LOWER or equal
    """
    df["target"] = (
        df.groupby("ticker")["close"]
        .shift(-1)                      # tomorrow's close
        .gt(df["close"])                # > today's close?
        .astype(int)
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "return_1d", "return_3d", "return_5d",
    "sma_ratio",
    "volatility_5d",
    "volume_ratio",
    "high_low_range", "close_open_diff",
    "rsi_14",
    "sentiment_mean", "sentiment_std", "news_count",
]

TARGET_COL = "target"


def build_features(
    stocks_path: str | Path = "data/datasets/stocks.parquet",
    news_path:   str | Path = "data/datasets/news.parquet",
    output_path: str | Path = "data/datasets/features.parquet",
) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.

    Returns the feature DataFrame and saves it to output_path.
    """
    logger.info("Loading stocks …")
    stocks = pd.read_parquet(stocks_path)
    stocks["date"] = pd.to_datetime(stocks["date"])
    stocks = stocks.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info("Loading news …")
    news = pd.read_parquet(news_path)
    news["date"] = pd.to_datetime(news["date"])

    logger.info("Adding price features …")
    stocks = _add_price_features(stocks)

    logger.info("Adding sentiment features …")
    stocks = _add_sentiment_features(stocks, news)

    logger.info("Adding target label …")
    stocks = _add_target(stocks)

    # Drop rows with NaNs in feature columns or target
    before = len(stocks)
    stocks = stocks.dropna(subset=FEATURE_COLS + [TARGET_COL])
    logger.info(f"Dropped {before - len(stocks)} rows with NaNs → {len(stocks)} rows remain")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    stocks.to_parquet(output_path, index=False)
    logger.info(f"Features saved → {output_path}")

    return stocks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = build_features()
    print(df[FEATURE_COLS + [TARGET_COL]].describe())
