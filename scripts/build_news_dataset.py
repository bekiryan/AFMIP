"""
Build Financial News Dataset
─────────────────────────────
Generates a clean, deduplicated financial news dataset from FNSPID data.

Output : data/datasets/news.parquet
Key    : (date, ticker)  — joinable with the stock dataset

Usage
─────
    # Quick test with local sample CSV
    python scripts/build_news_dataset.py --sample 100

    # Full build from HuggingFace
    python scripts/build_news_dataset.py

    # Use local CSV only (no HuggingFace download)
    python scripts/build_news_dataset.py --local-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from rich.console import Console

from config.settings import FNSPID_NEWS_CSV, FNSPID_HF_REPO, DATASETS_DIR, AZURE_CONTAINER_DATASETS, NEWSAPI_KEY
from src.data.loader import load_newsapi_articles, load_sp500_tickers
from src.azure.storage import AzureStorageClient

console = Console()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build financial news dataset")
    p.add_argument("--sample", type=int, default=None,
                   help="Limit to N rows (for quick testing)")
    p.add_argument("--local-only", action="store_true",
                   help="Only use local CSV, don't download from HuggingFace")
    p.add_argument("--skip-newsapi", action="store_true",
                   help="Skip NewsAPI (free tier: last 30 days)")
    p.add_argument("--newsapi-only", action="store_true",
                   help="Only use NewsAPI, skip FNSPID/HuggingFace")
    p.add_argument("--newsapi-tickers", type=str, default=None,
                   help="Comma-separated tickers for NewsAPI (default: top 20 S&P 500)")
    p.add_argument("--output", type=str, default=None,
                   help="Output path (default: data/datasets/news.parquet)")
    p.add_argument("--upload", action="store_true",
                   help="Upload dataset to Azure Blob Storage / Data Lake Gen2")
    return p.parse_args()


# ── Loading ────────────────────────────────────────────────────────────────────

def load_from_huggingface() -> pd.DataFrame:
    """Load full FNSPID news from HuggingFace datasets library."""
    console.print("  Loading from HuggingFace [cyan]Zihan1004/FNSPID[/] ...")
    try:
        from datasets import load_dataset

        ds = load_dataset(FNSPID_HF_REPO, split="train")
        df = ds.to_pandas()
        console.print(f"  Loaded [green]{len(df):,}[/] rows from HuggingFace")
        return df
    except Exception as e:
        console.print(f"  [red]HuggingFace load failed:[/] {e}")
        console.print("  Falling back to local CSV ...")
        return pd.DataFrame()


def load_from_local_csv() -> pd.DataFrame:
    """Load FNSPID news from local CSV sample."""
    if not FNSPID_NEWS_CSV.exists():
        console.print(f"  [red]Local CSV not found:[/] {FNSPID_NEWS_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(FNSPID_NEWS_CSV, low_memory=False)
    console.print(f"  Loaded [green]{len(df):,}[/] rows from local CSV")
    return df


def load_from_newsapi(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load recent news from NewsAPI (free tier: last 30 days)."""
    if not NEWSAPI_KEY:
        console.print("  [yellow]Skipping NewsAPI[/] — NEWSAPI_KEY not set")
        return pd.DataFrame()

    if not tickers:
        # Default to top 20 S&P 500 tickers to stay within free tier limits
        all_tickers = load_sp500_tickers()
        tickers = all_tickers[:20]

    console.print(f"  Fetching NewsAPI for [cyan]{len(tickers)}[/] tickers (last 30 days) ...")
    df = load_newsapi_articles(tickers)
    console.print(f"  Got [green]{len(df):,}[/] articles from NewsAPI")
    return df


def load_news(local_only: bool, skip_newsapi: bool = True, newsapi_only: bool = False, newsapi_tickers: list[str] | None = None) -> pd.DataFrame:
    """Load news data from available sources."""
    console.rule("[bold blue]Step 1 — Load News Data")

    if newsapi_only:
        return load_from_newsapi(newsapi_tickers)

    if local_only:
        df = load_from_local_csv()
    else:
        df = load_from_huggingface()
        if df.empty:
            df = load_from_local_csv()

    # Add NewsAPI as supplemental source
    if not skip_newsapi:
        newsapi_df = load_from_newsapi(newsapi_tickers)
        if not newsapi_df.empty:
            df = pd.concat([df, newsapi_df], ignore_index=True) if not df.empty else newsapi_df

    return df


# ── Cleaning ───────────────────────────────────────────────────────────────────

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise FNSPID column names to our standard schema."""
    rename_map = {
        "Date": "date",
        "Article_title": "title",
        "Stock_symbol": "ticker",
        "Url": "url",
        "Publisher": "publisher",
        "Author": "author",
        "Article": "article",
        "Lsa_summary": "lsa_summary",
        "Luhn_summary": "luhn_summary",
        "Textrank_summary": "textrank_summary",
        "Lexrank_summary": "lexrank_summary",
    }
    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Also handle already-lowercase columns (from HuggingFace)
    df.columns = df.columns.str.lower().str.strip()

    return df


def clean_news(df: pd.DataFrame, sample: int | None = None) -> pd.DataFrame:
    """Clean and deduplicate news data."""
    console.rule("[bold blue]Step 2 — Clean & Deduplicate")

    df = normalise_columns(df)

    # Normalise date
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Normalise ticker
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Drop rows without essential fields
    before = len(df)
    df = df.dropna(subset=["date", "ticker", "title"])
    dropped = before - len(df)
    if dropped:
        console.print(f"  Dropped [yellow]{dropped:,}[/] rows missing date/ticker/title")

    # Deduplicate on (date, ticker, title)
    before = len(df)
    df = df.drop_duplicates(subset=["date", "ticker", "title"])
    dupes = before - len(df)
    if dupes:
        console.print(f"  Removed [yellow]{dupes:,}[/] duplicate rows")

    # Sample if requested
    if sample and sample < len(df):
        df = df.head(sample)
        console.print(f"  Sampled to [cyan]{sample:,}[/] rows")

    console.print(
        f"  Clean: [green]{len(df):,}[/] rows, "
        f"[green]{df['ticker'].nunique():,}[/] tickers, "
        f"[green]{df['date'].min().date()} → {df['date'].max().date()}[/]"
    )
    return df


# ── Output ─────────────────────────────────────────────────────────────────────

def save_dataset(df: pd.DataFrame, output: Path) -> None:
    """Select final columns and save to Parquet."""
    console.rule("[bold blue]Step 3 — Save")

    # Keep only the columns we need
    keep_cols = ["date", "ticker", "title", "article", "publisher", "url"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Sort for efficient queries
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False, engine="pyarrow")
    size_mb = output.stat().st_size / 1e6
    console.print(f"  Saved [green]{len(df):,}[/] rows → {output}  ({size_mb:.1f} MB)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output else DATASETS_DIR / "news.parquet"

    # Step 1: Load
    newsapi_tickers = None
    if args.newsapi_tickers:
        newsapi_tickers = [t.strip().upper() for t in args.newsapi_tickers.split(",")]

    raw_df = load_news(
        local_only=args.local_only,
        skip_newsapi=args.skip_newsapi,
        newsapi_only=args.newsapi_only,
        newsapi_tickers=newsapi_tickers,
    )
    if raw_df.empty:
        console.print("\n  [red]No data loaded.[/] Check data sources.")
        return

    # Step 2: Clean
    clean_df = clean_news(raw_df, sample=args.sample)

    if clean_df.empty:
        console.print("\n  [red]No rows after cleaning.[/]")
        return

    # Step 3: Save
    save_dataset(clean_df, output)

    if args.upload:
        console.rule("[bold blue]Step 4 — Azure Upload")
        try:
            client = AzureStorageClient(container=AZURE_CONTAINER_DATASETS)
            client.ensure_container()
            url = client.upload_file(output, blob_path=output.name)
            console.print(f"  [green]Uploaded {output.name} → {url}[/]")
        except ImportError as e:
            console.print(f"  [red]Azure SDK not installed:[/] {e}")
        except Exception as e:
            console.print(f"  [red]Azure upload failed:[/] {e}\n  Set Connection String/SP env vars.")

    console.rule("[bold green]News Dataset Complete")
    console.print(f"  Output: {output}")
    console.print(f"  Rows: {len(clean_df):,}  |  Tickers: {clean_df['ticker'].nunique()}")
    console.print(f"  Date range: {clean_df['date'].min().date()} → {clean_df['date'].max().date()}")


if __name__ == "__main__":
    main()
