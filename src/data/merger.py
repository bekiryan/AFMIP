"""
Dataset Merger — join news + stock datasets on (date, ticker).

This is a convenience utility, not part of the core pipeline.
Both datasets should be built independently first, then joined here
for training.

Usage
─────
    from src.data.merger import merge_news_stocks

    import pandas as pd
    news = pd.read_parquet("data/datasets/news.parquet")
    stocks = pd.read_parquet("data/datasets/stocks.parquet")
    merged = merge_news_stocks(news, stocks)
"""

from __future__ import annotations

import pandas as pd


def merge_news_stocks(
    news_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Join news and stock datasets on (date, ticker).

    Parameters
    ----------
    news_df   : DataFrame with at least (date, ticker, title)
    stocks_df : DataFrame with at least (date, ticker, open, high, low, close, volume)
    how       : join type — 'inner' (default), 'left', 'right', 'outer'

    Returns
    -------
    Merged DataFrame with all columns from both datasets.
    Stock columns get a '_stock' suffix if they conflict with news columns.
    """
    # Normalise date columns
    news_df = news_df.copy()
    stocks_df = stocks_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
    stocks_df["date"] = pd.to_datetime(stocks_df["date"]).dt.normalize()

    # Normalise tickers
    news_df["ticker"] = news_df["ticker"].astype(str).str.upper().str.strip()
    stocks_df["ticker"] = stocks_df["ticker"].astype(str).str.upper().str.strip()

    merged = news_df.merge(
        stocks_df,
        on=["date", "ticker"],
        how=how,
        suffixes=("", "_stock"),
    )

    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


def alignment_report(
    news_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
) -> dict:
    """
    Quick alignment stats between the two datasets.
    Returns a dict with match counts and overlap info.
    """
    news_keys = set(zip(
        pd.to_datetime(news_df["date"]).dt.date,
        news_df["ticker"].str.upper(),
    ))
    stock_keys = set(zip(
        pd.to_datetime(stocks_df["date"]).dt.date,
        stocks_df["ticker"].str.upper(),
    ))

    news_tickers = {t for _, t in news_keys}
    stock_tickers = {t for _, t in stock_keys}

    matched = len(news_keys & stock_keys)

    return {
        "news_rows": len(news_df),
        "stock_rows": len(stocks_df),
        "news_unique_keys": len(news_keys),
        "stock_unique_keys": len(stock_keys),
        "matched_keys": matched,
        "match_rate": f"{matched / len(news_keys):.1%}" if news_keys else "N/A",
        "tickers_in_both": len(news_tickers & stock_tickers),
        "tickers_news_only": len(news_tickers - stock_tickers),
        "tickers_stocks_only": len(stock_tickers - news_tickers),
    }
