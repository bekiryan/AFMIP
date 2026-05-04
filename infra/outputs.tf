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

output "datalake_models_url" {
  description = "URL for the models container"
  value       = "https://${azurerm_storage_account.afmip.name}.blob.core.windows.net/${azurerm_storage_container.models.name}"
}

output "datalake_exports_url" {
  description = "URL for the exports container"
  value       = "https://${azurerm_storage_account.afmip.name}.blob.core.windows.net/${azurerm_storage_container.exports.name}"
}

output "function_app_url" {
  description = "URL of the Function App"
  value       = var.enable_functions ? "https://${azurerm_linux_function_app.afmip[0].default_hostname}" : null
}

output "function_app_name" {
  description = "Name of the Function App"
  value       = var.enable_functions ? azurerm_linux_function_app.afmip[0].name : null
}

output "pipeline_trigger_url" {
  description = "Manual trigger URL for the data pipeline"
  value       = var.enable_functions ? "https://${azurerm_linux_function_app.afmip[0].default_hostname}/api/run-pipeline" : null
}

output "data_factory_name" {
  description = "Name of the Azure Data Factory"
  value       = var.enable_data_factory ? azurerm_data_factory.afmip[0].name : null
}

output "data_factory_id" {
  description = "ID of the Azure Data Factory"
  value       = var.enable_data_factory ? azurerm_data_factory.afmip[0].id : null
}

output "azure_ml_workspace_name" {
  description = "Name of the Azure ML workspace"
  value       = var.enable_azure_ml ? azurerm_machine_learning_workspace.afmip[0].name : null
}

output "azure_ml_compute_name" {
  description = "Name of the Azure ML compute cluster"
  value       = var.enable_azure_ml && var.enable_azure_ml_compute ? azurerm_machine_learning_compute_cluster.cpu[0].name : null
}

output "azure_ml_storage_account_name" {
  description = "Name of the non-HNS Storage Account used by Azure ML"
  value       = var.enable_azure_ml ? azurerm_storage_account.ml[0].name : null
}
