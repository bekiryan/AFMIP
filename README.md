# AFMIP — AI-Driven Financial Market Intelligence Platform

Predicts next-day stock price direction (UP/DOWN) for S&P 500 companies by combining financial news with OHLCV price data.

## Stage 1 — Data Ingestion, Validation & Feature Extraction

### What it does

1. **Builds Stock Dataset**: Extracts historical OHLCV data from FNSPID (29.7M records) and falls back to Yahoo Finance via `yfinance` to fill gaps up to the present.
2. **Builds News Dataset**: Gathers 15.7M financial news articles natively sourced from the FNSPID project.
3. **Harmonises Data**: Normalises date boundaries and ticker symbols for immediate joining. 
4. **Saves Datasets**: Outputs as portable Parquet files and optimally supports an automated Azure Data Lake Gen2 upload pipeline logic.

### Project structure

```
AFMIP/
├── config/                  # Configuration settings
│   └── settings.py          # Paths, Azure env vars, ML parameters
├── data/                    # Local storage for datasets (ignored in git)
│   ├── datasets/            # Built parquet datasets (stocks.parquet, news.parquet)
│   ├── raw/                 # Raw downloaded files
│   └── sp500_tickers.csv    # Ticker list
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
│   ├── build_news_dataset.py  # Pipeline to gather and process financial news articles
│   ├── build_stock_dataset.py # Pipeline to fetch and process historical stock price data
│   └── view_datasets.py       # Browser dashboard for raw stock and news datasets
├── src/                     # Core application source code
│   ├── azure/               # Azure integration modules
│   │   └── storage.py       # Azure Blob / Data Lake Gen2 upload/download functions
│   ├── data/                # Data ingestion, validation and merging
│   │   ├── loader.py        # FNSPID news/prices, yfinance data loading
│   │   └── merger.py        # Merging datasets (e.g. news with stock data)
├── requirements.txt         # Core dependencies for local execution
├── .env.example             # Azure credential template
├── implementation_plan.md   # Initial project roadmap and implementation stages
└── README.md                # Project documentation
```

### Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the core datasets
python scripts/build_stock_dataset.py
python scripts/build_news_dataset.py --local-only

# 4. View datasets side-by-side in the browser
python scripts/view_datasets.py
```

### Dataset building flags

| Flag | Description |
|------|-------------|
| `--sample <N>` | Process only N rows (useful for quick testing) |
| `--yfinance` | Explicitly update standard datasets with the latest yfinance history |
| `--upload` | Upload successfully constructed Parquet files to Azure Data Lake Gen2 |
| `--local-only` | (News dataset) Force parsing of local CSV dataset and skip HF Hub |

#### Data viewer

```bash
python scripts/view_datasets.py            # opens browser at http://localhost:8502
python scripts/view_datasets.py --port 9000  # custom port
```

Interactive dashboard to inspect DataFrames. It uses a clean tabular format, numeric sorting abilities and sparkline-capable analysis indicators.

### Azure upload

```bash
cp .env.example .env
# Fill in your Azure Storage credentials, then use the upload flag:
python scripts/build_stock_dataset.py --upload
python scripts/build_news_dataset.py --local-only --upload
```

Supports three authentication methods (in priority order):
1. Connection string (`AZURE_STORAGE_CONNECTION_STRING`)
2. Service principal (`AZURE_TENANT_ID` + `AZURE_CLIENT_ID` + `AZURE_CLIENT_SECRET`)
3. `DefaultAzureCredential` (managed identity, CLI login, etc.)

### Built datasets output

The independent Parquet files are stored cleanly under `data/datasets/`:
1. `stocks.parquet` (columns: `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`)
2. `news.parquet` (columns: `date`, `ticker`, `title`, `article`, `publisher`, `url`)

These are joined in real-time iteratively or fully orchestrated by `src/data/merger.py`.

### Datasets

| Dataset | Role | Size |
|---------|------|------|
| [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) | Core historical news + prices (1999–2023) | 29.7M prices, 15.7M news |
| [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) | Sentiment label fine-tuning (future stage) | ~4,840 sentences |
| [Yahoo Finance (yfinance)](https://github.com/ranaroussi/yfinance) | Price data gap-fill (2023–present) | On-demand via API |

### Azure Deployment (Free Tier)

The platform can be deployed to Azure using only free tier resources ($0/month).

#### Resources Provisioned

| Resource | Tier | Purpose |
|----------|------|---------|
| Azure Blob Storage (Data Lake Gen2) | Free (5 GB LRS) | Store Parquet datasets |
| Azure Functions (Consumption) | Free (1M exec/mo) | Run pipeline on schedule |

#### Deploy with Terraform

```bash
# One-time setup
cd infra
cp terraform.tfvars.example terraform.tfvars  # adjust values if needed
terraform init

# Review and deploy
terraform plan
terraform apply

# Get connection string for local .env
terraform output -raw storage_connection_string
```

#### Upload Datasets

```bash
# Build and upload stock dataset
python scripts/build_stock_dataset.py --upload

# Build and upload news dataset
python scripts/build_news_dataset.py --local-only --upload
```

#### Azure Functions

The Function App runs the pipeline daily at 06:00 UTC. You can also trigger it manually:

```bash
# Via Azure Functions URL (after deployment)
curl https://<function-app-name>.azurewebsites.net/api/run-pipeline?code=<function-key>
```

### Next stages

- **Stage 2**: FinBERT / DistilRoBERTa sentiment scoring on article text
- **Stage 3**: Binary classification model (RandomForest / XGBoost)
- **Stage 4**: Backtesting framework and signal evaluation
- **Stage 5**: Azure Event Hubs real-time ingestion + Azure ML deployment
