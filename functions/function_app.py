"""
AFMIP Azure Functions App
──────────────────────────
Timer-triggered and HTTP-triggered functions for running the data pipeline.

Timer trigger: runs daily at 06:00 UTC (after US market close + settlement)
HTTP trigger:  manual on-demand pipeline execution

Data sources:
  - Stocks: yfinance + Alpaca historical bars (merged, deduped)
  - News:   local FNSPID CSV + NewsAPI (last 25 days, free tier)
"""

import azure.functions as func
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import our modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = func.FunctionApp()


def run_pipeline(trigger_source: str) -> dict:
    """
    Core pipeline logic shared by both triggers.

    Builds stock and news datasets using all available sources,
    then uploads to Azure Data Lake.
    Returns a summary dict with status and stats.
    """
    import pandas as pd
    from config.settings import DATASETS_DIR, YFINANCE_FULL_START
    from src.azure.storage import AzureStorageClient

    results = {
        "trigger": trigger_source,
        "timestamp": datetime.utcnow().isoformat(),
        "stocks": {"status": "skipped"},
        "news": {"status": "skipped"},
    }

    # ── Stock dataset ─────────────────────────────────────────────────────────
    try:
        logging.info("Building stock dataset...")
        from scripts.build_stock_dataset import (
            download_yfinance_step,
            download_alpaca_step,
            merge_and_clean,
            resolve_tickers,
            save_dataset,
        )

        tickers = resolve_tickers(None)

        empty = pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "volume"]
        )

        # yfinance
        yf_df = download_yfinance_step(tickers, start=YFINANCE_FULL_START, end=None)

        # Alpaca
        try:
            alpaca_df = download_alpaca_step(tickers, start=YFINANCE_FULL_START, end=None)
        except Exception as e:
            logging.warning(f"Alpaca download failed, continuing without it: {e}")
            alpaca_df = empty.copy()

        final_df = merge_and_clean(empty, yf_df, alpaca_df)

        output_path = DATASETS_DIR / "stocks.parquet"
        save_dataset(final_df, output_path)

        # Upload to Azure
        client = AzureStorageClient(container="afmip-datasets")
        client.ensure_container()
        url = client.upload_file(output_path, blob_path="datasets/stocks.parquet")

        results["stocks"] = {
            "status": "success",
            "rows": len(final_df),
            "tickers": int(final_df["ticker"].nunique()),
            "blob_url": url,
        }
        logging.info(f"Stock dataset: {len(final_df):,} rows uploaded")
    except Exception as e:
        logging.error(f"Stock dataset failed: {e}")
        results["stocks"] = {"status": "error", "error": str(e)}

    # ── News dataset ──────────────────────────────────────────────────────────
    try:
        logging.info("Building news dataset...")
        from scripts.build_news_dataset import clean_news, load_news
        from scripts.build_news_dataset import save_dataset as save_news

        # Load from local CSV + NewsAPI (skip HuggingFace — too heavy for Functions)
        raw_df = load_news(local_only=True, skip_newsapi=False)

        if not raw_df.empty:
            clean_df = clean_news(raw_df, sample=None)

            output_path = DATASETS_DIR / "news.parquet"
            save_news(clean_df, output_path)

            client = AzureStorageClient(container="afmip-datasets")
            url = client.upload_file(output_path, blob_path="datasets/news.parquet")

            results["news"] = {
                "status": "success",
                "rows": len(clean_df),
                "tickers": int(clean_df["ticker"].nunique()),
                "blob_url": url,
            }
            logging.info(f"News dataset: {len(clean_df):,} rows uploaded")
        else:
            results["news"] = {"status": "skipped", "reason": "no data available"}
    except Exception as e:
        logging.error(f"News dataset failed: {e}")
        results["news"] = {"status": "error", "error": str(e)}

    return results


@app.timer_trigger(schedule="0 0 6 * * *", arg_name="timer", run_on_startup=False)
def daily_pipeline(timer: func.TimerRequest) -> None:
    """Run the data pipeline daily at 06:00 UTC."""
    logging.info("Daily pipeline triggered")

    if timer.past_due:
        logging.warning("Timer is past due — running anyway")

    results = run_pipeline(trigger_source="timer")
    logging.info(f"Pipeline complete: {json.dumps(results, indent=2)}")


@app.route(route="run-pipeline", auth_level=func.AuthLevel.FUNCTION)
def manual_pipeline(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger for on-demand pipeline execution."""
    logging.info("Manual pipeline triggered via HTTP")

    results = run_pipeline(trigger_source="http")

    return func.HttpResponse(
        body=json.dumps(results, indent=2),
        mimetype="application/json",
        status_code=200
        if all(
            r.get("status") != "error" for r in [results["stocks"], results["news"]]
        )
        else 500,
    )
