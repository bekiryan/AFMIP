# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AFMIP** (AI-Driven Financial Market Intelligence Platform) predicts stock price movements by combining financial news with OHLCV price data. The project generates two independent, joinable datasets:

1. **Stock prices** — daily OHLCV from FNSPID (1999–2023) + yfinance (1999–2025)
2. **Financial news** — 15.7M articles from FNSPID (1999–2023)

Both share a `(date, ticker)` key for joining at training time.

## Project Structure

```
AFMIP/
├── config/                  # Configuration settings
│   ├── __init__.py
│   └── settings.py          # Paths, Azure env vars, ML parameters
├── data/                    # Local storage for datasets (ignored in git)
│   ├── datasets/            # Built parquet datasets (stocks.parquet, news.parquet)
│   ├── processed/           # Intermediate files
│   ├── raw/                 # Raw downloaded files
│   └── sp500_tickers.csv    # Ticker list
├── fnsipid_samples.csv      # 1,000-row FNSPID news sample
├── full_history.zip         # FNSPID stock prices (may be truncated/corrupt)
├── functions/               # Azure Functions app code
│   ├── function_app.py      # Entry point for Azure Functions triggers
│   ├── host.json            # Host configuration for the Function app
│   └── requirements.txt     # Dependencies for the Azure Functions execution environment
├── infra/                   # Terraform infrastructure-as-code definitions for Azure deployment
│   ├── functions.tf         # Azure Function App and related resources provisioning
│   ├── main.tf              # Main Terraform configuration and provider setup
│   ├── outputs.tf           # Terraform output variables
│   ├── storage.tf           # Azure Storage Account and Blob container provisioning
│   ├── terraform.tfvars.example # Example variable definitions for Terraform
│   └── variables.tf         # Terraform input variable definitions
├── scripts/                 # Utility scripts and dataset building pipelines
│   ├── build_news_dataset.py  # Builds clean news dataset (FNSPID via HuggingFace or local CSV)
│   ├── build_stock_dataset.py # Builds clean stock price dataset (FNSPID + yfinance)
│   └── view_datasets.py       # Browser dashboard for raw stock and news datasets
├── src/                     # Core application source code
│   ├── __init__.py
│   ├── azure/               # Azure integration modules
│   │   └── storage.py       # Azure Blob / Data Lake Gen2 upload/download functions
│   └── data/                # Data ingestion, validation and merging
│       ├── loader.py        # FNSPID news/prices, yfinance data loading
│       └── merger.py        # Joins news + stock datasets on (date, ticker)
├── requirements.txt         # Core dependencies for local execution
├── .env.example             # Azure credential template
├── implementation_plan.md   # Initial project roadmap and implementation stages
└── README.md                # Project documentation
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in Azure credentials if needed
```

## Commands

### Dataset Builders (primary workflow)

```bash
# Stock price dataset — quick test (3 tickers, 1 year)
python scripts/build_stock_dataset.py --tickers AAPL,MSFT,GOOG --start 2020-01-01 --end 2020-12-31

# Stock price dataset — full build (all tickers from news CSV, 1999-present)
python scripts/build_stock_dataset.py

# Stock price dataset — yfinance only (skip corrupt FNSPID zip)
python scripts/build_stock_dataset.py --skip-fnspid

# News dataset — local sample (quick test)
python scripts/build_news_dataset.py --local-only --sample 100

# News dataset — full from HuggingFace (15.7M articles)
python scripts/build_news_dataset.py

# News dataset — local CSV only
python scripts/build_news_dataset.py --local-only
```

### Data Viewer

```bash
python scripts/view_datasets.py          # http://localhost:8502
python scripts/view_datasets.py --port 9000
```

## Architecture

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `config/settings.py` | All paths, Azure env vars, ML constants (single source of truth) |
| `src/data/loader.py` | Ingests FNSPID news/prices, yfinance, S&P 500 ticker list |
| `src/data/merger.py` | Joins news + stock datasets on `(date, ticker)` with alignment reports |
| `src/azure/storage.py` | Upload/download Parquet to Azure Blob / Data Lake Gen2 |
| `scripts/build_stock_dataset.py` | CLI: FNSPID prices + yfinance → `data/datasets/stocks.parquet` |
| `scripts/build_news_dataset.py` | CLI: FNSPID news (HF or local) → `data/datasets/news.parquet` |
| `scripts/view_datasets.py` | Single-page HTML dashboard served via built-in HTTPServer |

### Dataset Schemas

**stocks.parquet** — `(date, ticker, open, high, low, close, volume)`

**news.parquet** — `(date, ticker, title, article, publisher, url)`

Both join on `(date, ticker)`:
```python
from src.data.merger import merge_news_stocks
merged = merge_news_stocks(news_df, stocks_df)  # inner join
```

### Data Sources

| Dataset | Coverage | Source |
|---------|----------|--------|
| FNSPID News | 1999–2023 | HuggingFace `Zihan1004/FNSPID` |
| FNSPID Prices | 1999–2023 | `full_history.zip` (local, may be corrupt) |
| yfinance | 1999–2025 | Yahoo Finance API (free, on-demand) |

### Key Config Values (`config/settings.py`)

- `DATASETS_DIR` → `data/datasets/` (standalone parquet output)
- `YFINANCE_FULL_START` → `1999-01-01` (full history start)
- `YFINANCE_START` → `2023-01-01` (gap-fill start for stage1)
- `FNSPID_HF_REPO` → `Zihan1004/FNSPID`

### Azure Auth Priority

`AzureStorageClient` tries in order:
1. `AZURE_STORAGE_CONNECTION_STRING`
2. Service principal (`AZURE_TENANT_ID` + `AZURE_CLIENT_ID` + `AZURE_CLIENT_SECRET`)
3. `DefaultAzureCredential` (managed identity, CLI login, etc.)

## Planned Stages (not yet implemented)

- **Stage 2**: FinBERT / DistilRoBERTa sentiment scoring on news
- **Stage 3**: Binary classification model (RandomForest / XGBoost)
- **Stage 4**: Backtesting framework and signal evaluation
- **Stage 5**: Azure Event Hubs real-time ingestion + Azure ML deployment
