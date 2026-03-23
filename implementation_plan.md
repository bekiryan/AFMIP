# Azure Deployment with Terraform — AFMIP (Free Tier Only)

Deploy the AFMIP data pipeline on Azure using **only free tier resources** with Terraform IaC.

## Architecture — Free Tier Resources

| Resource | Free Tier Allowance | Our Use |
|----------|-------------------|---------|
| **Azure Blob Storage (Data Lake Gen2)** | 5 GB LRS, 20K read, 10K write ops/mo (always free) | Store Parquet datasets as the "database" |
| **Azure Functions** (Consumption Plan) | 1M executions + 400K GB-s/mo (always free) | Run pipeline scripts on schedule/trigger |

> [!TIP]
> **No separate database needed** — your existing Parquet-on-Data-Lake approach already works perfectly. The [AzureStorageClient](file:///Users/bekiryan/Projects/AFMIP/src/azure/storage.py#48-154) you built uploads/downloads Parquet to Blob Storage. We'll enhance it to serve as a full data layer (read/write/query).

**Total cost: $0/month** (within free tier limits)

---

## Proposed Changes

### Terraform Infrastructure — `infra/`

#### [NEW] [main.tf](file:///Users/bekiryan/Projects/AFMIP/infra/main.tf)
- `azurerm` provider with `required_providers`
- `azurerm_resource_group.afmip`
- Tags and naming conventions

#### [NEW] [variables.tf](file:///Users/bekiryan/Projects/AFMIP/infra/variables.tf)
- `project_name` (default: `"afmip"`), `location` (default: `"eastus"`), `environment` (default: `"dev"`)

#### [NEW] [outputs.tf](file:///Users/bekiryan/Projects/AFMIP/infra/outputs.tf)
- Storage account name, connection string (sensitive), container URLs
- Function app URL

#### [NEW] [storage.tf](file:///Users/bekiryan/Projects/AFMIP/infra/storage.tf)
- `azurerm_storage_account` — Standard LRS, `is_hns_enabled = true` (Data Lake Gen2)
- Containers: `afmip-raw`, `afmip-datasets`, `afmip-features`

#### [NEW] [functions.tf](file:///Users/bekiryan/Projects/AFMIP/infra/functions.tf)
- `azurerm_service_plan` — Consumption tier (Y1, free)
- `azurerm_linux_function_app` — Python 3.11 runtime
- App settings: storage connection string, function-specific config
- Timer trigger for daily pipeline runs

#### [NEW] [terraform.tfvars.example](file:///Users/bekiryan/Projects/AFMIP/infra/terraform.tfvars.example)
- Example values for all variables

---

### Azure Functions App — `functions/`

#### [NEW] [functions/host.json](file:///Users/bekiryan/Projects/AFMIP/functions/host.json)
- Azure Functions v4 host config, Python worker

#### [NEW] [functions/requirements.txt](file:///Users/bekiryan/Projects/AFMIP/functions/requirements.txt)
- Subset of main requirements for the Function App runtime

#### [NEW] [functions/function_app.py](file:///Users/bekiryan/Projects/AFMIP/functions/function_app.py)
- **Timer-triggered function**: runs `build_stock_dataset` + `build_news_dataset` daily
- **HTTP-triggered function**: manual trigger for on-demand pipeline runs
- Both save Parquet results to Data Lake via [AzureStorageClient](file:///Users/bekiryan/Projects/AFMIP/src/azure/storage.py#48-154)

---

### Application Code Updates

#### [MODIFY] [settings.py](file:///Users/bekiryan/Projects/AFMIP/config/settings.py)
- Add `AZURE_FUNCTIONS_ENABLED` flag
- Add `DATALAKE_DATASETS_PREFIX` and `DATALAKE_FEATURES_PREFIX` for blob path config

#### [MODIFY] [storage.py](file:///Users/bekiryan/Projects/AFMIP/src/azure/storage.py)
- Add `list_datasets()` — list available Parquet files in the datasets container
- Add `download_file()` — download any file from blob storage to local path

#### [MODIFY] [build_stock_dataset.py](file:///Users/bekiryan/Projects/AFMIP/scripts/build_stock_dataset.py)
- Add `--upload` flag to push [stocks.parquet](file:///Users/bekiryan/Projects/AFMIP/data/datasets/stocks.parquet) to Data Lake after building

#### [MODIFY] [build_news_dataset.py](file:///Users/bekiryan/Projects/AFMIP/scripts/build_news_dataset.py)
- Add `--upload` flag to push [news.parquet](file:///Users/bekiryan/Projects/AFMIP/data/datasets/news.parquet) to Data Lake after building

#### [MODIFY] [.env.example](file:///Users/bekiryan/Projects/AFMIP/.env.example)
- Add new settings documentation

#### [MODIFY] [README.md](file:///Users/bekiryan/Projects/AFMIP/README.md)
- Add "Azure Deployment (Free Tier)" section with Terraform commands

---

## Verification Plan

### Automated Tests

1. **Terraform validation**:
   ```bash
   cd infra && terraform init && terraform validate
   terraform plan
   ```

2. **Python import check** — verify updated modules import cleanly:
   ```bash
   python -c "from src.azure.storage import AzureStorageClient; print('OK')"
   ```

### Manual Verification
1. Review `terraform plan` output to confirm all resources are free tier
2. After `terraform apply` — verify resources in Azure Portal
3. Run pipeline with `--upload` flag and verify data appears in Data Lake
