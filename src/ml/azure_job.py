"""
AFMIP — Azure ML Job Submission
=================================
Submits training pipeline to Azure ML cloud.

Usage:
    # Train next-day model on Azure (full train)
    python -m src.ml.azure_job --pipeline train --horizon 1d --wait

    # Train 1-week model on Azure (full train)
    python -m src.ml.azure_job --pipeline train --horizon 5d --wait

    # Run full pipeline: adapter + train both + predict + export
    python -m src.ml.azure_job --pipeline full --wait

    # Check job status
    python -m src.ml.azure_job --status
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import AmlCompute, Environment
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential
from dotenv import load_dotenv


def ensure_azure_cli_on_path() -> None:
    """Make AzureCliCredential find az when Python is launched outside the shell."""
    candidate_dirs = ["/opt/homebrew/bin", "/usr/local/bin"]
    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []

    for directory in reversed(candidate_dirs):
        if directory not in parts:
            parts.insert(0, directory)

    os.environ["PATH"] = os.pathsep.join(parts)

    if not shutil.which("az"):
        raise RuntimeError(
            "Azure CLI executable 'az' was not found on PATH. "
            "Install Azure CLI or add its directory to PATH."
        )


ensure_azure_cli_on_path()
load_dotenv()
logger = logging.getLogger(__name__)

# ── Config — reads from .env ──────────────────────────────────────────────────
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP  = os.getenv("AZURE_RESOURCE_GROUP",  "afmip-dev-rg")
WORKSPACE       = os.getenv("AZURE_ML_WORKSPACE",    "afmip-ml-workspace")
STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "afmipdevdp9fba")
STORAGE_CONN    = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
COMPUTE_NAME    = "afmip-cpu-cluster"
EXPERIMENT_NAME = "afmip-ml-training"
CONTAINER       = "afmip-datasets"


# ── Connect to Azure ML ───────────────────────────────────────────────────────

def get_client() -> MLClient:
    if not SUBSCRIPTION_ID:
        raise ValueError(
            "AZURE_SUBSCRIPTION_ID not set in .env\n"
            f"Current .env SUBSCRIPTION_ID: {SUBSCRIPTION_ID}"
        )
    client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE,
    )
    ws = client.workspaces.get(WORKSPACE)
    logger.info(f"Connected: {ws.name} ({ws.location})")
    return client


# ── Compute cluster ───────────────────────────────────────────────────────────

def ensure_compute(client: MLClient) -> str:
    try:
        client.compute.get(COMPUTE_NAME)
        logger.info(f"Compute '{COMPUTE_NAME}' exists — OK")
    except Exception:
        logger.info(f"Creating compute cluster '{COMPUTE_NAME}'...")
        client.compute.begin_create_or_update(AmlCompute(
            name=COMPUTE_NAME,
            type="amlcompute",
            size="Standard_DS3_v2",      # 4 cores, 14 GB RAM
            min_instances=0,              # FREE when idle
            max_instances=2,
            idle_time_before_scale_down=120,
        )).result()
        logger.info(f"Cluster '{COMPUTE_NAME}' created. Cost: ~$0.10/hr ONLY when running.")
    return COMPUTE_NAME


# ── ML Environment ────────────────────────────────────────────────────────────

def get_environment(client: MLClient) -> Environment:
    env = Environment(
        name="afmip-ml-env",
        description="AFMIP ML training environment",
        conda_file={
            "name": "afmip-ml",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.11", "pip",
                {"pip": [
                    "scikit-learn>=1.3",
                    "xgboost>=2.0",
                    "lightgbm>=4.0",
                    "pandas>=2.0",
                    "numpy>=1.24",
                    "pyarrow>=14.0",
                    "joblib>=1.3",
                    "matplotlib>=3.7",
                    "azure-ai-ml",
                    "azure-storage-blob",
                    "python-dotenv",
                ]},
            ],
        },
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
    )
    env = client.environments.create_or_update(env)
    logger.info(f"Environment: {env.name} v{env.version}")
    return env


# ── Upload data to Azure Blob ─────────────────────────────────────────────────

def upload_data():
    """Upload gold_dataset.csv and features.parquet to Azure Blob."""
    if not STORAGE_CONN:
        logger.warning("AZURE_STORAGE_CONNECTION_STRING not set — skipping upload")
        return

    try:
        from azure.storage.blob import BlobServiceClient
        blob_client = BlobServiceClient.from_connection_string(STORAGE_CONN)

        # Create container
        try:
            blob_client.create_container(CONTAINER)
        except Exception:
            pass

        files = {
            "data/datasets/gold_dataset.csv":   "gold/gold_dataset.csv",
            "data/datasets/features.parquet":   "features/features.parquet",
        }

        for local, remote in files.items():
            path = Path(local)
            if path.exists():
                bc = blob_client.get_blob_client(container=CONTAINER, blob=remote)
                with open(path, "rb") as f:
                    bc.upload_blob(f, overwrite=True)
                logger.info(f"Uploaded: {local} → {CONTAINER}/{remote}")
            else:
                logger.warning(f"File not found: {local}")

    except Exception as e:
        logger.error(f"Upload failed: {e}")


# ── Build job command ─────────────────────────────────────────────────────────

def build_command(pipeline: str, horizon: str, no_tune: bool) -> str:
    """Build the shell command that runs inside Azure ML."""

    tune_flag = "--no-tune" if no_tune else ""
    gold_input = "${{inputs.gold_data}}"
    model_output = "${{outputs.model_output}}"
    publish_outputs = (
        f"mkdir -p {model_output}/models {model_output}/exports && "
        f"cp -R data/models/. {model_output}/models/ && "
        f"cp -R data/exports/. {model_output}/exports/"
    )

    if pipeline == "train":
        # Single horizon train
        return (
            f"python -m src.ml.gold_adapter "
            f"--gold-path {gold_input} "
            f"--output-path data/datasets/features.parquet && "
            f"python -m src.ml.train "
            f"--horizon {horizon} "
            f"--full-train {tune_flag} "
            f"--features-path data/datasets/features.parquet && "
            f"{publish_outputs}"
        )

    elif pipeline == "full":
        # Full pipeline: both horizons + predict + export
        return (
            f"python -m src.ml.gold_adapter "
            f"--gold-path {gold_input} "
            f"--output-path data/datasets/features.parquet && "
            f"python -m src.ml.train --horizon 1d --full-train {tune_flag} && "
            f"python -m src.ml.train --horizon 5d --full-train {tune_flag} && "
            f"python -m src.ml.predict --all --export csv && "
            f"{publish_outputs} && "
            f"echo 'Pipeline complete'"
        )

    raise ValueError(f"Unknown pipeline: {pipeline}")


# ── Submit job ────────────────────────────────────────────────────────────────

def submit_job(
    client: MLClient,
    pipeline: str,
    horizon: str,
    no_tune: bool,
    gold_path: str,
    wait: bool,
) -> str:

    compute = ensure_compute(client)
    env     = get_environment(client)
    cmd     = build_command(pipeline, horizon, no_tune)

    ts   = datetime.now().strftime("%Y%m%d-%H%M")
    name = f"afmip-{pipeline}-{horizon}-{ts}"

    job = command(
        code="./",
        command=cmd,
        environment=env,
        compute=compute,
        experiment_name=EXPERIMENT_NAME,
        display_name=name,
        description=f"AFMIP {pipeline} pipeline — horizon={horizon} full_train=True",
        inputs={
            "gold_data": Input(
                type=AssetTypes.URI_FILE,
                path=gold_path,
            ),
        },
        outputs={
            "model_output": Output(
                type=AssetTypes.URI_FOLDER,
                path=f"azureml://datastores/workspaceblobstore/paths/afmip-models/",
            ),
        },
        tags={
            "project":    "AFMIP",
            "pipeline":   pipeline,
            "horizon":    horizon,
            "full_train": "true",
        },
    )

    submitted = client.jobs.create_or_update(job)

    print(f"\n{'='*60}")
    print(f"  JOB SUBMITTED")
    print(f"{'='*60}")
    print(f"  Name:     {submitted.name}")
    print(f"  Pipeline: {pipeline} | Horizon: {horizon}")
    print(f"  Status:   {submitted.status}")
    print(f"  URL:      {submitted.studio_url}")
    print(f"{'='*60}\n")
    print("  Open the URL above to watch logs in real time.")

    if wait:
        logger.info("Waiting for job to complete...")
        client.jobs.stream(submitted.name)
        final = client.jobs.get(submitted.name)
        print(f"\n  Job finished: {final.status}")

    return submitted.name


# ── Show job status ───────────────────────────────────────────────────────────

def show_status(client: MLClient):
    jobs = list(client.jobs.list(max_results=5))
    print(f"\n{'='*60}")
    print(f"  Recent AFMIP Jobs")
    print(f"{'='*60}")
    print(f"{'Name':<40} {'Status':<12} {'Created'}")
    print(f"{'-'*60}")
    for j in jobs:
        created = str(j.creation_context.created_at)[:16] if j.creation_context else "—"
        print(f"{j.display_name or j.name:<40} {j.status:<12} {created}")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Submit AFMIP job to Azure ML")
    parser.add_argument(
        "--pipeline",
        choices=["train", "full"],
        default="full",
        help="train=single horizon | full=both horizons + predict + export",
    )
    parser.add_argument(
        "--horizon",
        choices=["1d", "5d"],
        default="1d",
        help="Which horizon to train (only used with --pipeline train)",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning (faster, good for testing)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for job to finish and stream logs",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show recent job statuses",
    )
    parser.add_argument(
        "--gold-path",
        default="data/datasets/gold_dataset.csv",
        help="Gold dataset path. Local files are uploaded by Azure ML as job inputs.",
    )
    parser.add_argument(
        "--upload-data",
        action="store_true",
        help="Upload data to Azure Blob before submitting job",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for noisy_logger in [
        "azure",
        "azure.core.pipeline.policies.http_logging_policy",
        "azure.identity",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    if args.upload_data:
        logger.info("Uploading data to Azure Blob...")
        upload_data()

    client = get_client()

    if args.status:
        show_status(client)
        return

    submit_job(
        client,
        pipeline=args.pipeline,
        horizon=args.horizon,
        no_tune=args.no_tune,
        gold_path=args.gold_path,
        wait=args.wait,
    )


if __name__ == "__main__":
    main()
