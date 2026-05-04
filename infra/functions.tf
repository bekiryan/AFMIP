# ── App Service Plan (Consumption / Free Tier) ────────────────────────────────

resource "azurerm_service_plan" "afmip" {
  count               = var.enable_functions ? 1 : 0
  name                = "${var.project_name}-${var.environment}-plan"
  resource_group_name = azurerm_resource_group.afmip.name
  location            = azurerm_resource_group.afmip.location
  os_type             = "Linux"
  sku_name            = "Y1" # Consumption tier (free: 1M executions/mo)

  tags = azurerm_resource_group.afmip.tags
}

# ── Linux Function App (Python 3.11) ─────────────────────────────────────────

resource "azurerm_linux_function_app" "afmip" {
  count               = var.enable_functions ? 1 : 0
  name                = "${var.project_name}-${var.environment}-func"
  resource_group_name = azurerm_resource_group.afmip.name
  location            = azurerm_resource_group.afmip.location
  service_plan_id     = azurerm_service_plan.afmip[0].id

  storage_account_name       = azurerm_storage_account.afmip.name
  storage_account_access_key = azurerm_storage_account.afmip.primary_access_key

  site_config {
    application_stack {
      python_version = "3.11"
    }
  }

  app_settings = {
    AZURE_STORAGE_CONNECTION_STRING = azurerm_storage_account.afmip.primary_connection_string
    AZURE_CONTAINER                 = "afmip-datasets"
    FUNCTIONS_WORKER_RUNTIME        = "python"
    AzureWebJobsFeatureFlags        = "EnableWorkerIndexing"
    ALPACA_API_KEY                  = var.alpaca_api_key
    ALPACA_SECRET_KEY               = var.alpaca_secret_key
    NEWSAPI_KEY                     = var.newsapi_key
  }

  tags = azurerm_resource_group.afmip.tags
}
