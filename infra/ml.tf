# ── Azure ML Workspace ────────────────────────────────────────────────────────

data "azurerm_client_config" "current" {}

resource "azurerm_application_insights" "afmip" {
  count               = var.enable_azure_ml ? 1 : 0
  name                = "${var.project_name}-${var.environment}-appi"
  location            = azurerm_resource_group.afmip.location
  resource_group_name = azurerm_resource_group.afmip.name
  application_type    = "web"

  tags = azurerm_resource_group.afmip.tags

  lifecycle {
    ignore_changes = [workspace_id]
  }
}

resource "azurerm_key_vault" "afmip" {
  count                      = var.enable_azure_ml ? 1 : 0
  name                       = "${var.project_name}-${var.environment}-kv-${random_string.storage_suffix.result}"
  location                   = azurerm_resource_group.afmip.location
  resource_group_name        = azurerm_resource_group.afmip.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false
  enable_rbac_authorization  = true

  tags = azurerm_resource_group.afmip.tags
}

resource "azurerm_storage_account" "ml" {
  count                    = var.enable_azure_ml ? 1 : 0
  name                     = local.ml_storage_account_name
  resource_group_name      = azurerm_resource_group.afmip.name
  location                 = azurerm_resource_group.afmip.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = false

  tags = azurerm_resource_group.afmip.tags
}

resource "azurerm_machine_learning_workspace" "afmip" {
  count                   = var.enable_azure_ml ? 1 : 0
  name                    = var.azure_ml_workspace_name
  location                = azurerm_resource_group.afmip.location
  resource_group_name     = azurerm_resource_group.afmip.name
  application_insights_id = azurerm_application_insights.afmip[0].id
  key_vault_id            = azurerm_key_vault.afmip[0].id
  storage_account_id      = azurerm_storage_account.ml[0].id

  identity {
    type = "SystemAssigned"
  }

  tags = azurerm_resource_group.afmip.tags
}

resource "azurerm_machine_learning_compute_cluster" "cpu" {
  count                         = var.enable_azure_ml && var.enable_azure_ml_compute ? 1 : 0
  name                          = "afmip-cpu-cluster"
  location                      = azurerm_resource_group.afmip.location
  vm_priority                   = "Dedicated"
  vm_size                       = var.azure_ml_compute_size
  machine_learning_workspace_id = azurerm_machine_learning_workspace.afmip[0].id

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = var.azure_ml_compute_max_nodes
    scale_down_nodes_after_idle_duration = "PT120S"
  }
}
