# ── Outputs ────────────────────────────────────────────────────────────────────

output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.afmip.name
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.afmip.name
}

output "storage_connection_string" {
  description = "Primary connection string for the storage account"
  value       = azurerm_storage_account.afmip.primary_connection_string
  sensitive   = true
}

output "datalake_raw_url" {
  description = "URL for the raw data container"
  value       = "https://${azurerm_storage_account.afmip.name}.blob.core.windows.net/${azurerm_storage_container.raw.name}"
}

output "datalake_datasets_url" {
  description = "URL for the datasets container"
  value       = "https://${azurerm_storage_account.afmip.name}.blob.core.windows.net/${azurerm_storage_container.datasets.name}"
}

output "datalake_features_url" {
  description = "URL for the features container"
  value       = "https://${azurerm_storage_account.afmip.name}.blob.core.windows.net/${azurerm_storage_container.features.name}"
}

output "function_app_url" {
  description = "URL of the Function App"
  value       = "https://${azurerm_linux_function_app.afmip.default_hostname}"
}

output "function_app_name" {
  description = "Name of the Function App"
  value       = azurerm_linux_function_app.afmip.name
}

output "pipeline_trigger_url" {
  description = "Manual trigger URL for the data pipeline"
  value       = "https://${azurerm_linux_function_app.afmip.default_hostname}/api/run-pipeline"
}

output "data_factory_name" {
  description = "Name of the Azure Data Factory"
  value       = azurerm_data_factory.afmip.name
}

output "data_factory_id" {
  description = "ID of the Azure Data Factory"
  value       = azurerm_data_factory.afmip.id
}
