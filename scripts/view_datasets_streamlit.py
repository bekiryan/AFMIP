#!/usr/bin/env python3
"""
Streamlit dataset explorer for AFMIP parquet files.

Run directly:
    streamlit run scripts/view_datasets_streamlit.py

Or use launcher:
    python scripts/view_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import DATASETS_DIR  # noqa: E402

STOCKS_PATH = DATASETS_DIR / "stocks.parquet"
NEWS_PATH = DATASETS_DIR / "news.parquet"

st.set_page_config(page_title="AFMIP Dataset Explorer", page_icon="chart_with_upwards_trend", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out


def prep_stocks(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    out = normalize_dates(df)
    if "close" in out.columns and "ticker" in out.columns:
        out = out.sort_values(["ticker", "date"])
        out["_return"] = out.groupby("ticker")["close"].pct_change()
    out = out.tail(max_rows)
    return out.astype(object).where(pd.notna(out), None)


def prep_news(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    out = normalize_dates(df)
    out = out.head(max_rows)
    return out.astype(object).where(pd.notna(out), None)


def render_summary(df: pd.DataFrame, dataset_name: str) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
    c3.metric("Unique Tickers", f"{tickers:,}")
    st.caption(f"Showing sampled view of {dataset_name} data")


def render_table(df: pd.DataFrame, key_prefix: str) -> None:
    if df.empty:
        st.warning("No rows to display after filtering.")
        return

    filter_cols = st.columns(3)

    search_text = filter_cols[0].text_input("Search", key=f"search_{key_prefix}").strip().lower()

    selected_ticker = "__all__"
    if "ticker" in df.columns:
        ticker_options = ["__all__"] + sorted(df["ticker"].dropna().astype(str).unique().tolist())
        selected_ticker = filter_cols[1].selectbox("Ticker", ticker_options, key=f"ticker_{key_prefix}")

    selectable_cols = ["__all__"] + [str(c) for c in df.columns]
    selected_col = filter_cols[2].selectbox("Column", selectable_cols, key=f"col_{key_prefix}")

    view = df
    if selected_ticker != "__all__" and "ticker" in view.columns:
        view = view[view["ticker"].astype(str) == selected_ticker]

    if search_text:
        if selected_col != "__all__":
            view = view[view[selected_col].astype(str).str.lower().str.contains(search_text, na=False)]
        else:
            mask = pd.Series(False, index=view.index)
            for col in view.columns:
                mask = mask | view[col].astype(str).str.lower().str.contains(search_text, na=False)
            view = view[mask]

    st.dataframe(view, use_container_width=True, hide_index=True)


def show_missing(dataset_label: str, build_cmd: str) -> None:
    st.error(f"{dataset_label} dataset not found.")
    st.code(build_cmd, language="bash")


def main() -> None:
    st.title("AFMIP Dataset Explorer")

    with st.sidebar:
        st.header("Display Settings")
        max_stock_rows = st.slider("Stocks rows", min_value=200, max_value=10000, value=1000, step=200)
        max_news_rows = st.slider("News rows", min_value=100, max_value=10000, value=1000, step=100)
        st.caption(f"Datasets directory: {DATASETS_DIR}")

    tab_stocks, tab_news = st.tabs(["Stocks", "News"])

    with tab_stocks:
        if STOCKS_PATH.exists():
            with st.spinner("Loading stocks dataset..."):
                stocks_raw = load_dataset(str(STOCKS_PATH))
                stocks_df = prep_stocks(stocks_raw, max_stock_rows)
            render_summary(stocks_df, "stocks")
            render_table(stocks_df, "stocks")
        else:
            show_missing("Stocks", "python scripts/build_stock_dataset.py")

    with tab_news:
        if NEWS_PATH.exists():
            with st.spinner("Loading news dataset..."):
                news_raw = load_dataset(str(NEWS_PATH))
                news_df = prep_news(news_raw, max_news_rows)
            render_summary(news_df, "news")
            render_table(news_df, "news")
        else:
            show_missing("News", "python scripts/build_news_dataset.py")


if __name__ == "__main__":
    main()
