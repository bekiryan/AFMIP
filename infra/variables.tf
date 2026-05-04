# ── Input Variables ────────────────────────────────────────────────────────────

variable "project_name" {
  description = "Project name used as prefix for all resources"
  type        = string
  default     = "afmip"
}

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "eastus"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "storage_account_name" {
  description = "Globally unique Storage Account name. Leave empty to generate one."
  type        = string
  default     = ""
}

variable "ml_storage_account_name" {
  description = "Globally unique non-HNS Storage Account name for Azure ML. Leave empty to generate one."
  type        = string
  default     = ""
}

variable "enable_functions" {
  description = "Create Azure Functions resources. Some subscriptions have Dynamic VM quota 0 and cannot create Consumption Functions."
  type        = bool
  default     = false
}

variable "enable_data_factory" {
  description = "Create Azure Data Factory resources. The current pipeline depends on the Function App trigger."
  type        = bool
  default     = false
}

variable "enable_azure_ml" {
  description = "Create Azure ML workspace dependencies and workspace"
  type        = bool
  default     = true
}

variable "enable_azure_ml_compute" {
  description = "Create Azure ML CPU compute cluster with min nodes set to 0"
  type        = bool
  default     = true
}

variable "azure_ml_workspace_name" {
  description = "Azure ML workspace name"
  type        = string
  default     = "afmip-ml-workspace"
}

variable "azure_ml_compute_size" {
  description = "VM size for Azure ML compute jobs"
  type        = string
  default     = "Standard_DS3_v2"
}

variable "azure_ml_compute_max_nodes" {
  description = "Maximum Azure ML compute nodes"
  type        = number
  default     = 1
}

variable "alpaca_api_key" {
  description = "Alpaca API Key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "alpaca_secret_key" {
  description = "Alpaca Secret Key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "newsapi_key" {
  description = "NewsAPI API Key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "source_db_connection_string" {
  description = "Connection string for the source database to be extracted via Data Factory"
  type        = string
  sensitive   = true
  default     = "Integrated Security=False;Encrypt=True;Connection Timeout=30;Data Source=tcp:example.database.windows.net,1433;Initial Catalog=example_db;User ID=example_user;Password=example_password;"
}
