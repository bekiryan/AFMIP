"""
Build Stock Price Dataset
─────────────────────────
Generates a clean, deduplicated stock price dataset (daily OHLCV) from
FNSPID historical data + yfinance supplement.

Output : data/datasets/stocks.parquet
Key    : (date, ticker)  — joinable with the news dataset

Usage
─────
    # Quick test (3 tickers, 1 year)
    python scripts/build_stock_dataset.py --tickers AAPL,MSFT,GOOG --start 2020-01-01 --end 2020-12-31

    # Full build (all FNSPID tickers, 1999-present)
    python scripts/build_stock_dataset.py

    # Skip corrupt FNSPID zip, use yfinance only
    python scripts/build_stock_dataset.py --skip-fnspid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from rich.console import Console

from config.settings import (
    FNSPID_PRICES_ZIP,
    FNSPID_NEWS_CSV,
    DATASETS_DIR,
    YFINANCE_FULL_START,
    AZURE_CONTAINER_DATASETS,
)
from src.data.loader import load_fnspid_prices, load_yfinance_prices, load_alpaca_prices, load_sp500_tickers
from src.azure.storage import AzureStorageClient

console = Console()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stock price dataset")
    p.add_argument("--skip-fnspid", action="store_true",
                   help="Skip FNSPID zip, use yfinance only")
    p.add_argument("--skip-alpaca", action="store_true",
                   help="Skip Alpaca historical bars")
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated list of tickers (default: all from news)")
    p.add_argument("--start", type=str, default=YFINANCE_FULL_START,
                   help="Start date for yfinance download (default: 1999-01-01)")
    p.add_argument("--end", type=str, default=None,
                   help="End date for yfinance download (default: today)")
    p.add_argument("--output", type=str, default=None,
                   help="Output path (default: data/datasets/stocks.parquet)")
    p.add_argument("--upload", action="store_true",
                   help="Upload dataset to Azure Blob Storage / Data Lake Gen2")
    return p.parse_args()


# ── Ticker resolution ──────────────────────────────────────────────────────────

def resolve_tickers(explicit: str | None) -> list[str]:
    """Determine which tickers to download."""
    if explicit:
        return [t.strip().upper() for t in explicit.split(",")]

    # Try to get tickers from the news CSV (so we cover the same universe)
    news_csv = FNSPID_NEWS_CSV
    if news_csv.exists():
        console.print(f"  Reading tickers from [cyan]{news_csv.name}[/] ...")
        df = pd.read_csv(news_csv, usecols=["Stock_symbol"], low_memory=False)
        tickers = sorted(df["Stock_symbol"].dropna().astype(str).str.upper().str.strip().unique())
        console.print(f"  Found [green]{len(tickers)}[/] unique tickers in news CSV")
        return tickers

    # Fallback: S&P 500
    console.print("  No news CSV found — using S&P 500 ticker list")
    return load_sp500_tickers()


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def load_fnspid_step() -> pd.DataFrame:
    """Try loading FNSPID prices from zip."""
    import zipfile as _zf

    console.rule("[bold blue]Step 1 — Load FNSPID Prices")
    empty = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    if not FNSPID_PRICES_ZIP.exists():
        console.print(f"  [yellow]Skipping[/] — {FNSPID_PRICES_ZIP.name} not found")
        return empty

    try:
        df = load_fnspid_prices(FNSPID_PRICES_ZIP)
        console.print(
            f"  Loaded [green]{len(df):,}[/] rows, "
            f"[green]{df['ticker'].nunique():,}[/] tickers, "
            f"[green]{df['date'].min().date()} → {df['date'].max().date()}[/]"
        )
        return df
    except _zf.BadZipFile:
        size_mb = FNSPID_PRICES_ZIP.stat().st_size / 1e6
        console.print(
            f"  [yellow]Warning[/] — {FNSPID_PRICES_ZIP.name} is corrupt "
            f"({size_mb:.0f} MB). Skipping."
        )
        return empty


def download_yfinance_step(
    tickers: list[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Download full OHLCV history from yfinance."""
    console.rule("[bold blue]Step 2 — Download yfinance Prices")
    console.print(
        f"  Downloading [cyan]{len(tickers)}[/] tickers "
        f"from [cyan]{start}[/] to [cyan]{end or 'today'}[/] ..."
    )

    df = load_yfinance_prices(tickers, start=start, end=end)
    console.print(
        f"  Got [green]{len(df):,}[/] rows, "
        f"[green]{df['ticker'].nunique():,}[/] tickers"
    )
    return df


def download_alpaca_step(
    tickers: list[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Download daily OHLCV bars from Alpaca Markets."""
    from config.settings import ALPACA_API_KEY

    console.rule("[bold blue]Step 2b — Download Alpaca Prices")
    if not ALPACA_API_KEY:
        console.print("  [yellow]Skipping[/] — ALPACA_API_KEY not set")
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    console.print(
        f"  Downloading [cyan]{len(tickers)}[/] tickers "
        f"from [cyan]{start}[/] to [cyan]{end or 'today'}[/] ..."
    )
    df = load_alpaca_prices(tickers, start=start, end=end)
    console.print(
        f"  Got [green]{len(df):,}[/] rows, "
        f"[green]{df['ticker'].nunique():,}[/] tickers"
    )
    return df


def merge_and_clean(fnspid_df: pd.DataFrame, yf_df: pd.DataFrame, alpaca_df: pd.DataFrame = None) -> pd.DataFrame:
    """Merge FNSPID + yfinance + Alpaca, deduplicate, validate."""
    console.rule("[bold blue]Step 3 — Merge & Clean")

    parts = [fnspid_df, yf_df]
    if alpaca_df is not None and not alpaca_df.empty:
        parts.append(alpaca_df)
    combined = pd.concat(parts, ignore_index=True)

    # Normalise dates
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()

    # Deduplicate — keep yfinance (last) when both exist
    before = len(combined)
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    dupes = before - len(combined)
    if dupes:
        console.print(f"  Removed [yellow]{dupes:,}[/] duplicate (date, ticker) rows")

    # Drop invalid rows
    combined = combined.dropna(subset=["date", "ticker", "close"])
    combined = combined[combined["close"] > 0]

    # Ensure column types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["ticker"] = combined["ticker"].astype(str).str.upper().str.strip()
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    console.print(
        f"  Final: [green]{len(combined):,}[/] rows, "
        f"[green]{combined['ticker'].nunique():,}[/] tickers, "
        f"[green]{combined['date'].min().date()} → {combined['date'].max().date()}[/]"
    )
    return combined


def save_dataset(df: pd.DataFrame, output: Path) -> None:
    """Save to Parquet."""
    console.rule("[bold blue]Step 4 — Save")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep only the clean OHLCV columns
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]]

    df.to_parquet(output, index=False, engine="pyarrow")
    size_mb = output.stat().st_size / 1e6
    console.print(f"  Saved [green]{len(df):,}[/] rows → {output}  ({size_mb:.1f} MB)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output else DATASETS_DIR / "stocks.parquet"

    tickers = resolve_tickers(args.tickers)

    # Step 1: FNSPID
    if args.skip_fnspid:
        fnspid_df = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        console.print("  [yellow]Skipping FNSPID (--skip-fnspid)[/]")
    else:
        fnspid_df = load_fnspid_step()

    # Step 2: yfinance
    yf_df = download_yfinance_step(tickers, start=args.start, end=args.end)

    # Step 2b: Alpaca
    if args.skip_alpaca:
        alpaca_df = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        console.print("  [yellow]Skipping Alpaca (--skip-alpaca)[/]")
    else:
        alpaca_df = download_alpaca_step(tickers, start=args.start, end=args.end)

    # Step 3: Merge & clean
    final_df = merge_and_clean(fnspid_df, yf_df, alpaca_df)

    if final_df.empty:
        console.print("\n  [red]No data produced.[/] Check tickers and date range.")
        return

    # Step 4: Save
    save_dataset(final_df, output)

    if args.upload:
        console.rule("[bold blue]Step 5 — Azure Upload")
        try:
            client = AzureStorageClient(container=AZURE_CONTAINER_DATASETS)
            client.ensure_container()
            url = client.upload_file(output, blob_path=output.name)
            console.print(f"  [green]Uploaded {output.name} → {url}[/]")
        except ImportError as e:
            console.print(f"  [red]Azure SDK not installed:[/] {e}")
        except Exception as e:
            console.print(f"  [red]Azure upload failed:[/] {e}\n  Set Connection String/SP env vars.")

    console.rule("[bold green]Stock Dataset Complete")
    console.print(f"  Output: {output}")
    console.print(f"  Rows: {len(final_df):,}  |  Tickers: {final_df['ticker'].nunique()}")
    console.print(f"  Date range: {final_df['date'].min().date()} → {final_df['date'].max().date()}")


if __name__ == "__main__":
    main()
