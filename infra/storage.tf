# ── Storage Account (Data Lake Gen2) ──────────────────────────────────────────

resource "azurerm_storage_account" "afmip" {
  name                     = "${var.project_name}${var.environment}store"
  resource_group_name      = azurerm_resource_group.afmip.name
  location                 = azurerm_resource_group.afmip.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = true # Enables Data Lake Gen2

  tags = azurerm_resource_group.afmip.tags
}

# ── Blob Containers ──────────────────────────────────────────────────────────

resource "azurerm_storage_container" "raw" {
  name                  = "afmip-raw"
  storage_account_name  = azurerm_storage_account.afmip.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "datasets" {
  name                  = "afmip-datasets"
  storage_account_name  = azurerm_storage_account.afmip.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "features" {
  name                  = "afmip-features"
  storage_account_name  = azurerm_storage_account.afmip.name
  container_access_type = "private"
}
