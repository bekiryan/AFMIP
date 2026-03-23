"""
Upload Datasets to Azure Blob Storage
──────────────────────────────────────
Uploads stocks.parquet and news.parquet from data/datasets/ to Azure Blob
Storage using the credentials configured in .env.

Output : blobs under the 'datasets/' prefix in the configured container
Prereq : .env with Azure credentials, datasets already built

Usage
─────
    # Upload both datasets
    python scripts/upload_datasets.py

    # Upload only stocks
    python scripts/upload_datasets.py --stocks-only

    # Upload only news
    python scripts/upload_datasets.py --news-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.azure.storage import AzureStorageClient
from config.settings import DATASETS_DIR


def upload_datasets(stocks: bool = True, news: bool = True) -> None:
    client = AzureStorageClient()
    client.ensure_container()

    if stocks:
        stocks_path = DATASETS_DIR / "stocks.parquet"
        if not stocks_path.exists():
            print(f"ERROR: {stocks_path} not found. Run build_stock_dataset.py first.")
            sys.exit(1)
        print(f"Uploading {stocks_path} ...")
        df = pd.read_parquet(stocks_path)
        url = client.upload_dataframe(df, "datasets/stocks.parquet")
        print(f"  -> {url}  ({len(df):,} rows)")

    if news:
        news_path = DATASETS_DIR / "news.parquet"
        if not news_path.exists():
            print(f"ERROR: {news_path} not found. Run build_news_dataset.py first.")
            sys.exit(1)
        print(f"Uploading {news_path} ...")
        df = pd.read_parquet(news_path)
        url = client.upload_dataframe(df, "datasets/news.parquet")
        print(f"  -> {url}  ({len(df):,} rows)")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload datasets to Azure Blob Storage")
    parser.add_argument("--stocks-only", action="store_true", help="Upload only stocks.parquet")
    parser.add_argument("--news-only", action="store_true", help="Upload only news.parquet")
    args = parser.parse_args()

    if args.stocks_only and args.news_only:
        parser.error("Cannot use both --stocks-only and --news-only")

    upload_datasets(
        stocks=not args.news_only,
        news=not args.stocks_only,
    )


if __name__ == "__main__":
    main()
