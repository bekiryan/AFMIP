"""
AFMIP — Azure ML Job Submission (Startup Grade)
================================================
Submits training + monitoring jobs to Azure ML.
Supports full pipeline: features → train → evaluate → monitor.

Usage:
    python -m src.ml.azure_job                     # train all horizons
    python -m src.ml.azure_job --pipeline full     # features + train + evaluate
    python -m src.ml.azure_job --horizon 1d        # single horizon
    python -m src.ml.azure_job --monitor           # run monitoring job
"""

import argparse
import logging
import os
from pathlib import Path

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import AmlCompute, Environment, Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "72150dfc-44bc-4625-a1db-340c42d59f06")
RESOURCE_GROUP  = os.getenv("AZURE_RESOURCE_GROUP",  "afmip-dev-rg")
WORKSPACE       = os.getenv("AZURE_ML_WORKSPACE",    "afmip-ml-workspace")
COMPUTE_NAME    = "afmip-cpu-cluster"
EXPERIMENT_NAME = "afmip-ml"


def get_client() -> MLClient:
    client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE,
    )
    logger.info(f"Connected: {WORKSPACE}")
    return client


def ensure_compute(client: MLClient) -> str:
    try:
        client.compute.get(COMPUTE_NAME)
        logger.info(f"Compute '{COMPUTE_NAME}' exists.")
    except Exception:
        logger.info(f"Creating compute '{COMPUTE_NAME}' ...")
        client.compute.begin_create_or_update(AmlCompute(
            name=COMPUTE_NAME,
            type="amlcompute",
            size="Standard_DS3_v2",   # 4 cores, 14 GB — better for LightGBM
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=120,
        )).result()
    return COMPUTE_NAME


def get_environment(client: MLClient) -> Environment:
    env = Environment(
        name="afmip-ml-env",
        description="AFMIP ML environment",
        conda_file={
            "name": "afmip-ml",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.11", "pip",
                {"pip": [
                    "scikit-learn>=1.3", "xgboost>=2.0", "lightgbm>=4.0",
                    "pandas>=2.0", "numpy>=1.24", "pyarrow>=14.0",
                    "joblib>=1.3", "matplotlib>=3.7",
                    "azure-ai-ml", "python-dotenv",
                ]},
            ],
        },
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
    )
    return client.environments.create_or_update(env)


def submit_job(
    client: MLClient,
    cmd: str,
    display_name: str,
    features_path: str = "data/datasets/features.parquet",
    wait: bool = False,
):
    compute = ensure_compute(client)
    env     = get_environment(client)

    job = command(
        code="./",
        command=cmd,
        environment=env,
        compute=compute,
        experiment_name=EXPERIMENT_NAME,
        display_name=display_name,
        inputs={
            "features_path": Input(
                type=AssetTypes.URI_FILE,
                path=features_path,
            ),
        },
        outputs={
            "models_output": Output(
                type=AssetTypes.URI_FOLDER,
                path="azureml://datastores/workspaceblobstore/paths/afmip/models/",
            ),
            "reports_output": Output(
                type=AssetTypes.URI_FOLDER,
                path="azureml://datastores/workspaceblobstore/paths/afmip/reports/",
            ),
        },
        tags={"project": "AFMIP"},
    )

    submitted = client.jobs.create_or_update(job)
    logger.info(f"\nJob submitted: {submitted.name}")
    logger.info(f"URL: {submitted.studio_url}\n")

    if wait:
        client.jobs.stream(submitted.name)

    return submitted.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        choices=["train", "full", "monitor"],
        default="train",
        help="train=train only | full=features+train+evaluate | monitor=monitoring check"
    )
    parser.add_argument("--horizon",       choices=["1d","5d","21d","63d","252d","all"], default="all")
    parser.add_argument("--no-tune",       action="store_true")
    parser.add_argument("--wait",          action="store_true")
    parser.add_argument("--features-path", default="data/datasets/features.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    client = get_client()

    tune_flag    = "--no-tune" if args.no_tune else ""
    horizon_flag = f"--horizon {args.horizon}"

    if args.pipeline == "train":
        cmd = f"python -m src.ml.train {horizon_flag} {tune_flag}"
        name = f"afmip-train-{args.horizon}"

    elif args.pipeline == "full":
        cmd = (
            f"python -m src.ml.features && "
            f"python -m src.ml.train {horizon_flag} {tune_flag} && "
            f"python -m src.ml.evaluate {horizon_flag}"
        )
        name = f"afmip-full-pipeline"

    elif args.pipeline == "monitor":
        cmd  = "python -m src.ml.monitor --fix"
        name = "afmip-monitor"

    submit_job(client, cmd, name, args.features_path, wait=args.wait)


if __name__ == "__main__":
    main()
