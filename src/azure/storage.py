"""
Azure Data Lake Gen2 / Blob Storage upload helpers.

Authentication priority
───────────────────────
1. Connection string  (AZURE_STORAGE_CONNECTION_STRING env var)  — easiest for local dev
2. Service principal  (AZURE_TENANT_ID + AZURE_CLIENT_ID + AZURE_CLIENT_SECRET)
3. DefaultAzureCredential                                         — works inside Azure VMs / ACI

Usage
─────
from src.azure.storage import AzureStorageClient

client = AzureStorageClient()
client.upload_dataframe(features_df, blob_path="features/sp500_features.parquet")
client.upload_file(local_path, blob_path="raw/full_history.zip")
path = client.download_dataframe("features/sp500_features.parquet")
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    from azure.identity import (
        ClientSecretCredential,
        DefaultAzureCredential,
    )
    _AZURE_AVAILABLE = True
except ImportError:
    _AZURE_AVAILABLE = False

from config.settings import (
    AZURE_CONTAINER,
    AZURE_STORAGE_ACCOUNT,
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_TENANT_ID,
    AZURE_CLIENT_ID,
    AZURE_CLIENT_SECRET,
)


class AzureStorageClient:
    """
    Thin wrapper around BlobServiceClient for AFMIP feature storage.

    All uploads go to the container configured in AZURE_CONTAINER
    (default: 'afmip-features').
    """

    def __init__(self, container: str = AZURE_CONTAINER) -> None:
        if not _AZURE_AVAILABLE:
            raise ImportError(
                "azure-storage-blob and azure-identity are not installed. "
                "Run: pip install azure-storage-blob azure-identity"
            )
        self.container = container
        self._client = self._build_service_client()

    # ── Upload ─────────────────────────────────────────────────────────────────

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        blob_path: str,
        overwrite: bool = True,
    ) -> str:
        """
        Serialise *df* as Parquet and upload to Azure Blob / ADLS Gen2.

        Returns the full blob URL.
        """
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        buf.seek(0)

        blob_client = self._client.get_blob_client(
            container=self.container, blob=blob_path
        )
        blob_client.upload_blob(
            buf,
            overwrite=overwrite,
            content_settings=ContentSettings(content_type="application/octet-stream"),
        )
        return blob_client.url

    def upload_file(
        self,
        local_path: str | Path,
        blob_path: str,
        overwrite: bool = True,
    ) -> str:
        """Upload a local file (any format) to Azure Blob Storage."""
        local_path = Path(local_path)
        blob_client = self._client.get_blob_client(
            container=self.container, blob=blob_path
        )
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=overwrite)
        return blob_client.url

    # ── Download ───────────────────────────────────────────────────────────────

    def download_dataframe(self, blob_path: str) -> pd.DataFrame:
        """Download a Parquet blob and return a DataFrame."""
        blob_client = self._client.get_blob_client(
            container=self.container, blob=blob_path
        )
        stream = blob_client.download_blob()
        buf = io.BytesIO(stream.readall())
        return pd.read_parquet(buf, engine="pyarrow")

    def download_file(self, blob_path: str, local_path: str | Path) -> Path:
        """Download any blob to a local file path."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        blob_client = self._client.get_blob_client(
            container=self.container, blob=blob_path
        )
        with open(local_path, "wb") as f:
            stream = blob_client.download_blob()
            stream.readinto(f)
            
        return local_path

    # ── List ───────────────────────────────────────────────────────────────────

    def list_blobs(self, prefix: str = "") -> list[str]:
        """Return all blob names under the given prefix."""
        container_client = self._client.get_container_client(self.container)
        return [b.name for b in container_client.list_blobs(name_starts_with=prefix)]
        
    def list_datasets(self, prefix: str = "") -> list[str]:
        """List available Parquet files in the datasets container."""
        return [
            name for name in self.list_blobs(prefix=prefix)
            if name.endswith(".parquet")
        ]

    # ── Container setup ────────────────────────────────────────────────────────

    def ensure_container(self) -> None:
        """Create the container if it does not exist."""
        container_client = self._client.get_container_client(self.container)
        try:
            container_client.create_container()
        except Exception:
            pass  # already exists

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_service_client(self) -> "BlobServiceClient":
        if AZURE_STORAGE_CONNECTION_STRING:
            return BlobServiceClient.from_connection_string(
                AZURE_STORAGE_CONNECTION_STRING
            )

        if AZURE_TENANT_ID and AZURE_CLIENT_ID and AZURE_CLIENT_SECRET:
            credential = ClientSecretCredential(
                tenant_id=AZURE_TENANT_ID,
                client_id=AZURE_CLIENT_ID,
                client_secret=AZURE_CLIENT_SECRET,
            )
        else:
            credential = DefaultAzureCredential()

        account_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
        return BlobServiceClient(account_url=account_url, credential=credential)
