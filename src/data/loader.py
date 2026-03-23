"""
Data loading layer — FNSPID (historical) + yfinance (2023-present gap fill).

Public interface
────────────────
load_fnspid_news(path)      -> pd.DataFrame
load_fnspid_prices(zip_path)-> pd.DataFrame
load_yfinance_prices(tickers, start, end) -> pd.DataFrame
load_sp500_tickers()        -> list[str]
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockSnapshotRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import (
    FNSPID_NEWS_CSV,
    FNSPID_PRICES_ZIP,
    SP500_TICKERS_PATH,
    YFINANCE_START,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    NEWSAPI_KEY,
)


# ── FNSPID ─────────────────────────────────────────────────────────────────────

def load_fnspid_news(path: Path | str = FNSPID_NEWS_CSV) -> pd.DataFrame:
    """
    Load FNSPID news CSV (either the 1 000-row sample or the full HF export).

    Returns normalised columns:
        date (date), ticker (str), title (str), article (str), publisher (str)
    """
    df = pd.read_csv(path, low_memory=False)

    df = df.rename(columns={
        "Date": "date",
        "Stock_symbol": "ticker",
        "Article_title": "title",
        "Article": "article",
        "Publisher": "publisher",
        "Author": "author",
        "Url": "url",
        "Lsa_summary": "lsa_summary",
        "Luhn_summary": "luhn_summary",
        "Textrank_summary": "textrank_summary",
        "Lexrank_summary": "lexrank_summary",
    })

    # Normalise date to timezone-naive date objects
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["date", "ticker"])
    return df.reset_index(drop=True)


def load_fnspid_prices(zip_path: Path | str = FNSPID_PRICES_ZIP) -> pd.DataFrame:
    """
    Extract and load the FNSPID stock price CSV from the full_history.zip.

    Returns normalised columns:
        date (date), ticker (str), open, high, low, close, volume (float)

    Raises
    ------
    FileNotFoundError  if the zip does not exist
    zipfile.BadZipFile if the zip is corrupted / partially downloaded
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("No CSV found inside the zip archive.")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, low_memory=False)

    # Normalise column names (FNSPID uses lowercase)
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={"symbol": "ticker"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["date", "ticker", "close"])

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


# ── yfinance supplement (2023-present) ─────────────────────────────────────────

def load_yfinance_prices(
    tickers: list[str],
    start: str = YFINANCE_START,
    end: str | None = None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Download OHLCV data for *tickers* from Yahoo Finance.

    Downloads in batches to respect rate limits and stacks results into a
    long-form DataFrame with the same schema as load_fnspid_prices().
    """
    frames: list[pd.DataFrame] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        if raw.empty:
            continue

        # yfinance returns multi-level columns when >1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in batch:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                t_df = raw[ticker].copy()
                t_df = t_df.reset_index().rename(columns={"Date": "date"})
                t_df["ticker"] = ticker
                t_df.columns = t_df.columns.str.lower()
                frames.append(t_df)
        else:
            # Single ticker
            raw = raw.reset_index().rename(columns={"Date": "date"})
            raw["ticker"] = batch[0]
            raw.columns = raw.columns.str.lower()
            frames.append(raw)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "ticker", "close"])
    return df.reset_index(drop=True)


# ── Alpaca (Historical bars) ───────────────────────────────────────────────────

def load_alpaca_prices(
    tickers: list[str],
    start: str = YFINANCE_START,
    end: str | None = None,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Download daily OHLCV bars from Alpaca Markets.

    Returns the same schema as load_yfinance_prices():
        date (datetime), ticker (str), open, high, low, close, volume (float)
    """
    from datetime import datetime

    client = get_alpaca_client()
    if not client:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()

    frames: list[pd.DataFrame] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
            )
            bars = client.get_stock_bars(req)
            df = bars.df.reset_index()

            # Normalise to match our schema
            df = df.rename(columns={"symbol": "ticker", "timestamp": "date"})
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            frames.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])
        except Exception as e:
            print(f"  Alpaca batch {i//batch_size + 1} failed: {e}")
            continue

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["date", "ticker", "close"])
    return result.reset_index(drop=True)


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def load_newsapi_articles(
    tickers: list[str],
    max_requests: int = 90,
) -> pd.DataFrame:
    """
    Fetch financial news from NewsAPI (free tier: last 30 days, 100 req/day).

    Returns the same schema as load_fnspid_news():
        date (datetime), ticker (str), title (str), article (str), publisher (str), url (str)
    """
    import requests as _requests
    from datetime import datetime, timedelta

    if not NEWSAPI_KEY:
        raise ValueError("NEWSAPI_KEY must be set in .env")

    frames: list[pd.DataFrame] = []
    requests_made = 0
    from_date = (datetime.now() - timedelta(days=25)).strftime("%Y-%m-%d")

    for ticker in tickers:
        if requests_made >= max_requests:
            print(f"  NewsAPI: hit request limit ({max_requests}), stopping.")
            break

        try:
            resp = _requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": f'"{ticker}" stock',
                    "from": from_date,
                    "sortBy": "relevancy",
                    "language": "en",
                    "pageSize": 100,
                    "apiKey": NEWSAPI_KEY,
                },
                timeout=15,
            )
            requests_made += 1
            data = resp.json()

            if data.get("status") != "ok" or not data.get("articles"):
                continue

            articles = data["articles"]
            rows = []
            for a in articles:
                rows.append({
                    "date": a.get("publishedAt", ""),
                    "ticker": ticker.upper(),
                    "title": a.get("title", ""),
                    "article": a.get("description") or a.get("content") or "",
                    "publisher": (a.get("source") or {}).get("name", ""),
                    "url": a.get("url", ""),
                })
            frames.append(pd.DataFrame(rows))
        except Exception as e:
            print(f"  NewsAPI ticker {ticker} failed: {e}")
            continue

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "title", "article", "publisher", "url"])

    result = pd.concat(frames, ignore_index=True)
    result["date"] = (
        pd.to_datetime(result["date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    result = result.dropna(subset=["date", "ticker", "title"])
    return result.reset_index(drop=True)


# ── Alpaca (Real-time) ─────────────────────────────────────────────────────────

def get_alpaca_client() -> StockHistoricalDataClient | None:
    """Return an initialized Alpaca Client if credentials exist."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def load_realtime_quotes(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch the latest real-time quotes using App.Alpaca.Markets.
    Returns: DataFrame with columns [ticker, bid_price, bid_size, ask_price, ask_size, timestamp]
    """
    client = get_alpaca_client()
    if not client:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment.")

    req = StockLatestQuoteRequest(symbol_or_symbols=tickers)
    quotes = client.get_stock_latest_quote(req)

    data = []
    for ticker, quote in quotes.items():
        data.append({
            "ticker": ticker,
            "bid_price": quote.bid_price,
            "bid_size": quote.bid_size,
            "ask_price": quote.ask_price,
            "ask_size": quote.ask_size,
            "timestamp": quote.timestamp,
        })
    df = pd.DataFrame(data)
    # Ensure correct types
    for col in ("bid_price", "bid_size", "ask_price", "ask_size"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_realtime_snapshots(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch the latest real-time snapshots (latest trade, quote, daily bar) using Alpaca.
    Returns: DataFrame with snapshot information.
    """
    client = get_alpaca_client()
    if not client:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment.")

    req = StockSnapshotRequest(symbol_or_symbols=tickers)
    snapshots = client.get_stock_snapshot(req)

    data = []
    for ticker, snapshot in snapshots.items():
        daily = snapshot.daily_bar
        prev = snapshot.previous_daily_bar
        trade = snapshot.latest_trade
        data.append({
            "ticker": ticker,
            "latest_trade_price": trade.price if trade else None,
            "latest_trade_size": trade.size if trade else None,
            "today_open": daily.open if daily else None,
            "today_high": daily.high if daily else None,
            "today_low": daily.low if daily else None,
            "today_close": daily.close if daily else None,
            "today_volume": daily.volume if daily else None,
            "prev_close": prev.close if prev else None,
            "timestamp": trade.timestamp if trade else None,
        })
    df = pd.DataFrame(data)
    return df




# ── S&P 500 ticker universe ────────────────────────────────────────────────────

def load_sp500_tickers(cache_path: Path | str = SP500_TICKERS_PATH) -> list[str]:
    """
    Return the current S&P 500 ticker list, using a cached CSV when available.
    Falls back to scraping Wikipedia with a browser User-Agent.
    """
    import urllib.request as _urllib_req

    cache_path = Path(cache_path)

    if cache_path.exists():
        tickers = pd.read_csv(cache_path)["ticker"].tolist()
        return [str(t).upper() for t in tickers]

    # Wikipedia requires a real User-Agent header
    import io as _io

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = _urllib_req.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AFMIP-bot/1.0)"},
    )
    with _urllib_req.urlopen(req) as resp:
        html = resp.read().decode("utf-8")

    tables = pd.read_html(_io.StringIO(html))
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).str.upper().tolist()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
    return tickers
