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
