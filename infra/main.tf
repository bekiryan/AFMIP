# ── Terraform Configuration ───────────────────────────────────────────────────

terraform {
  required_version = ">= 1.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# ── Resource Group ────────────────────────────────────────────────────────────

resource "azurerm_resource_group" "afmip" {
  name     = "${var.project_name}-${var.environment}-rg"
  location = var.location

  tags = {
    project     = var.project_name
    environment = var.environment
    managed_by  = "terraform"
  }
}

resource "random_string" "storage_suffix" {
  length  = 6
  upper   = false
  special = false
}

locals {
  storage_account_name    = var.storage_account_name != "" ? var.storage_account_name : "${var.project_name}${var.environment}${random_string.storage_suffix.result}"
  ml_storage_account_name = var.ml_storage_account_name != "" ? var.ml_storage_account_name : "${var.project_name}${var.environment}ml${random_string.storage_suffix.result}"
}
