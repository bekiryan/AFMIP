# ── Data Factory Workspace ───────────────────────────────────────────────────

resource "azurerm_data_factory" "afmip" {
  count               = var.enable_data_factory ? 1 : 0
  name                = "${var.project_name}-${var.environment}-adf"
  location            = azurerm_resource_group.afmip.location
  resource_group_name = azurerm_resource_group.afmip.name

  tags = azurerm_resource_group.afmip.tags
}

# ── Linked Services ──────────────────────────────────────────────────────────

# Destination: Azure Blob Storage (Data Lake Gen2)
resource "azurerm_data_factory_linked_service_azure_blob_storage" "datalake" {
  count             = var.enable_data_factory ? 1 : 0
  name              = "DataLakeStorage"
  data_factory_id   = azurerm_data_factory.afmip[0].id
  connection_string = azurerm_storage_account.afmip.primary_connection_string
}

# Source 1: Azure SQL Database (Placeholder for "these databases")
resource "azurerm_data_factory_linked_service_azure_sql_database" "source_db" {
  count             = var.enable_data_factory ? 1 : 0
  name              = "SourceSQLDatabase"
  data_factory_id   = azurerm_data_factory.afmip[0].id
  connection_string = var.source_db_connection_string
}

# Source 2: REST API (For Stock Data via Alpaca)
resource "azurerm_data_factory_linked_service_web" "alpaca_api" {
  count               = var.enable_data_factory ? 1 : 0
  name                = "AlpacaWebAPI"
  data_factory_id     = azurerm_data_factory.afmip[0].id
  authentication_type = "Anonymous"
  url                 = "https://data.alpaca.markets"
}

# Source 3: REST API (For News via NewsAPI)
resource "azurerm_data_factory_linked_service_web" "newsapi" {
  count               = var.enable_data_factory ? 1 : 0
  name                = "NewsWebAPI"
  data_factory_id     = azurerm_data_factory.afmip[0].id
  authentication_type = "Anonymous"
  url                 = "https://newsapi.org"
}

# ── ADF Pipeline: Trigger Function App ────────────────────────────────────────

resource "azurerm_data_factory_pipeline" "daily_ingestion" {
  count           = var.enable_data_factory && var.enable_functions ? 1 : 0
  name            = "DailyDataIngestion"
  data_factory_id = azurerm_data_factory.afmip[0].id

  activities_json = jsonencode([
    {
      name = "RunDataPipeline"
      type = "WebActivity"
      typeProperties = {
        url    = "https://${azurerm_linux_function_app.afmip[0].default_hostname}/api/run-pipeline?code=${data.azurerm_function_app_host_keys.afmip[0].default_function_key}"
        method = "POST"
        body   = jsonencode({ source = "adf" })
        connectVia = {
          referenceName = "AutoResolveIntegrationRuntime"
          type          = "IntegrationRuntimeReference"
        }
      }
      policy = {
        timeout = "01:00:00"
        retry   = 1
      }
    }
  ])
}

# ── Get Function App host keys for ADF trigger ───────────────────────────────

data "azurerm_function_app_host_keys" "afmip" {
  count               = var.enable_data_factory && var.enable_functions ? 1 : 0
  name                = azurerm_linux_function_app.afmip[0].name
  resource_group_name = azurerm_resource_group.afmip.name
}

# ── ADF Trigger: Daily schedule ───────────────────────────────────────────────

resource "azurerm_data_factory_trigger_schedule" "daily" {
  count           = var.enable_data_factory && var.enable_functions ? 1 : 0
  name            = "DailyTrigger"
  data_factory_id = azurerm_data_factory.afmip[0].id
  pipeline_name   = azurerm_data_factory_pipeline.daily_ingestion[0].name

  frequency = "Day"
  interval  = 1

  # Run at 06:00 UTC (after US market close + settlement)
  schedule {
    hours   = [6]
    minutes = [0]
  }
}
