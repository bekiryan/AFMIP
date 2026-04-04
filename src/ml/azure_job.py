"""
AFMIP — Azure ML Job Submission
=================================
Submits your train.py as a job to Azure ML cloud.

What this does:
  1. Connects to your Azure ML workspace
  2. Uploads your src/ml/ code
  3. Downloads features.parquet from Data Lake into the job
  4. Runs train.py on a cloud compute cluster
  5. Saves the trained model to Azure ML Model Registry

Usage:
    python -m src.ml.azure_job                  # submit training job
    python -m src.ml.azure_job --wait           # submit and wait for completion
    python -m src.ml.azure_job --model xgboost  # train xgboost instead
"""

import argparse
import logging
import os
from pathlib import Path

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import (
    AmlCompute,
    Environment,
    Model,
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, AzureCliCredential
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — loaded from .env
# ---------------------------------------------------------------------------

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "72150dfc-44bc-4625-a1db-340c42d59f06")
RESOURCE_GROUP  = os.getenv("AZURE_RESOURCE_GROUP",  "afmip-dev-rg")
WORKSPACE       = os.getenv("AZURE_ML_WORKSPACE",    "afmip-ml-workspace")

COMPUTE_NAME    = "afmip-cpu-cluster"   # will be created if it doesn't exist
EXPERIMENT_NAME = "afmip-ml-training"


# ---------------------------------------------------------------------------
# Connect to Azure ML
# ---------------------------------------------------------------------------

def get_ml_client() -> MLClient:
    """Connect to Azure ML workspace using Azure CLI credentials."""
    try:
        credential = AzureCliCredential()
        client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE,
        )
        # Test connection
        ws = client.workspaces.get(WORKSPACE)
        logger.info(f"Connected to workspace: {ws.name} ({ws.location})")
        return client
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Azure ML.\n"
            f"Make sure you ran: az login\n"
            f"Error: {e}"
        )


# ---------------------------------------------------------------------------
# Create compute cluster (free tier — serverless or small CPU)
# ---------------------------------------------------------------------------

def ensure_compute(client: MLClient) -> str:
    """Create a CPU compute cluster if it doesn't exist yet."""
    try:
        client.compute.get(COMPUTE_NAME)
        logger.info(f"Compute cluster '{COMPUTE_NAME}' already exists.")
    except Exception:
        logger.info(f"Creating compute cluster '{COMPUTE_NAME}' ...")
        cluster = AmlCompute(
            name=COMPUTE_NAME,
            type="amlcompute",
            size="Standard_DS2_v2",   # 2 cores, 7 GB RAM — cheapest option
            min_instances=0,           # scales to 0 when idle (no cost)
            max_instances=1,
            idle_time_before_scale_down=120,
        )
        client.compute.begin_create_or_update(cluster).result()
        logger.info(f"Compute cluster '{COMPUTE_NAME}' created.")
    return COMPUTE_NAME


# ---------------------------------------------------------------------------
# Build the Python environment for the job
# ---------------------------------------------------------------------------

def get_environment(client: MLClient) -> Environment:
    """Define the Python environment the training job runs in."""
    env = Environment(
        name="afmip-ml-env",
        description="AFMIP ML training environment",
        conda_file={
            "name": "afmip-ml",
            "channels": ["conda-forge", "defaults"],
            "dependencies": [
                "python=3.11",
                "pip",
                {
                    "pip": [
                        "scikit-learn>=1.3",
                        "xgboost>=2.0",
                        "pandas>=2.0",
                        "numpy>=1.24",
                        "pyarrow>=14.0",
                        "joblib>=1.3",
                        "matplotlib>=3.7",
                        "azure-ai-ml",
                        "python-dotenv",
                    ]
                },
            ],
        },
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
    )
    # Register / update the environment
    env = client.environments.create_or_update(env)
    logger.info(f"Environment ready: {env.name} v{env.version}")
    return env


# ---------------------------------------------------------------------------
# Submit the training job
# ---------------------------------------------------------------------------

def submit_training_job(
    client: MLClient,
    model_type: str = "rf",
    no_tune: bool = False,
    wait: bool = False,
) -> str:
    """
    Submit train.py as an Azure ML command job.

    The job:
      - Takes features.parquet as input (from Data Lake or local upload)
      - Runs train.py
      - Outputs the trained model file
      - Registers the model in Azure ML Model Registry
    """
    compute = ensure_compute(client)
    env     = get_environment(client)

    # Build the command to run inside Azure ML
    tune_flag = "--no-tune" if no_tune else ""
    cmd = (
        f"python -m src.ml.train "
        f"--model {model_type} "
        f"--features-path ${{inputs.features_path}} "
        f"{tune_flag}"
    )

    # Define the job
    job = command(
        code="./",                          # upload the whole project folder
        command=cmd,
        environment=env,
        compute=compute,
        experiment_name=EXPERIMENT_NAME,
        display_name=f"afmip-train-{model_type}",
        description=f"Train AFMIP {model_type.upper()} model",
        inputs={
            "features_path": Input(
                type=AssetTypes.URI_FILE,
                path="data/datasets/features.parquet",  # local file → uploaded automatically
            ),
        },
        outputs={
            "model_output": Output(
                type=AssetTypes.URI_FOLDER,
                path="azureml://datastores/workspaceblobstore/paths/afmip/models/",
            ),
        },
        tags={
            "model_type": model_type,
            "project": "AFMIP",
        },
    )

    # Submit
    submitted = client.jobs.create_or_update(job)
    job_url   = submitted.studio_url

    logger.info(f"\n{'='*55}")
    logger.info(f"  Job submitted!")
    logger.info(f"  Name:   {submitted.name}")
    logger.info(f"  Status: {submitted.status}")
    logger.info(f"  URL:    {job_url}")
    logger.info(f"{'='*55}\n")
    logger.info("Open the URL above in your browser to watch the job run.")

    if wait:
        logger.info("Waiting for job to complete (this will take a while) ...")
        client.jobs.stream(submitted.name)   # streams logs to terminal
        final = client.jobs.get(submitted.name)
        logger.info(f"Job finished with status: {final.status}")

        if final.status == "Completed":
            _register_model(client, submitted.name, model_type)

    return submitted.name


# ---------------------------------------------------------------------------
# Register the model in Azure ML Model Registry
# ---------------------------------------------------------------------------

def _register_model(client: MLClient, job_name: str, model_type: str):
    """After a successful job, register the model so teammates can use it."""
    model = Model(
        path=f"azureml://jobs/{job_name}/outputs/model_output",
        name=f"afmip-{model_type}-model",
        description=f"AFMIP next-day direction classifier ({model_type.upper()})",
        type=AssetTypes.CUSTOM_MODEL,
        tags={"model_type": model_type, "project": "AFMIP"},
    )
    registered = client.models.create_or_update(model)
    logger.info(f"Model registered: {registered.name} v{registered.version}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Submit AFMIP training job to Azure ML")
    parser.add_argument("--model",    choices=["rf", "xgboost", "both"], default="rf")
    parser.add_argument("--no-tune",  action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--wait",     action="store_true", help="Wait for job to finish")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    client = get_ml_client()

    models = ["rf", "xgboost"] if args.model == "both" else [args.model]
    for m in models:
        submit_training_job(
            client,
            model_type=m,
            no_tune=args.no_tune,
            wait=args.wait,
        )


if __name__ == "__main__":
    main()
