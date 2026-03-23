"""
Central configuration — all paths and Azure settings loaded from environment
or .env file. No hardcoded secrets.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Repository root ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"

# Ensure local dirs exist
for _d in (RAW_DIR, PROCESSED_DIR, DATASETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── FNSPID dataset ─────────────────────────────────────────────────────────────
FNSPID_NEWS_CSV = ROOT_DIR / "fnsipid_samples.csv"        # 1 000-row sample already on disk
FNSPID_PRICES_ZIP = ROOT_DIR / "full_history.zip"
FNSPID_HF_REPO = "Zihan1004/FNSPID"

# ── yfinance supplement ────────────────────────────────────────────────────────
YFINANCE_START = "2023-01-01"          # pick up where FNSPID ends (≤ 2023)
YFINANCE_FULL_START = "1999-01-01"     # full history for standalone dataset

# ── Azure Data Lake Gen2 ───────────────────────────────────────────────────────
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "")
AZURE_CONTAINER_RAW = os.getenv("AZURE_CONTAINER_RAW", "afmip-raw")
AZURE_CONTAINER_DATASETS = os.getenv("AZURE_CONTAINER_DATASETS", "afmip-datasets")
# Default for legacy code, usually points to datasets now
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", AZURE_CONTAINER_DATASETS)
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")
# Alternatively, a SAS / connection string for simpler local testing:
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

# ── App.Alpaca.Markets ────────────────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# ── NewsAPI ────────────────────────────────────────────────────────────────────
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# ── Azure Functions ───────────────────────────────────────────────────────────
AZURE_FUNCTIONS_ENABLED = os.getenv("AZURE_FUNCTIONS_ENABLED", "false").lower() == "true"
DATALAKE_DATASETS_PREFIX = os.getenv("DATALAKE_DATASETS_PREFIX", "")

# ── ML target ─────────────────────────────────────────────────────────────────
PREDICTION_HORIZON_DAYS = 1            # T+1 next-day close
MARKET_SCOPE = "SP500"

# ── S&P 500 ticker list (sourced from Wikipedia at runtime if needed) ──────────
SP500_TICKERS_PATH = DATA_DIR / "sp500_tickers.csv"
