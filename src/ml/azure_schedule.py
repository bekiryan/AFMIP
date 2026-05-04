"""
AFMIP — Azure ML Daily Schedule
=================================
Sets up a daily schedule that automatically retrains the model
every day at 07:00 UTC (after the Azure Function fetches new data).

Run once to register the schedule:
    python -m src.ml.azure_schedule

Check schedule status:
    python -m src.ml.azure_schedule --status

Disable schedule:
    python -m src.ml.azure_schedule --disable
"""

import argparse
import logging

from azure.ai.ml import MLClient
from azure.ai.ml.entities import JobSchedule, RecurrenceTrigger, RecurrencePattern
from azure.ai.ml.constants import TimeZone
from dotenv import load_dotenv

from src.ml.azure_job import (
    get_client,
    ensure_compute,
    get_environment,
    EXPERIMENT_NAME,
)

load_dotenv()
logger = logging.getLogger(__name__)

SCHEDULE_NAME = "afmip-daily-retrain"


def create_schedule(client: MLClient):
    """Register a daily retraining schedule at 07:00 UTC."""

    ensure_compute(client)
    env = get_environment(client)

    # The job to run on schedule (same as azure_job.py but always with tuning)
    from azure.ai.ml import command, Input, Output
    from azure.ai.ml.constants import AssetTypes

    job = command(
        code="./",
        command=(
            "python -m src.ml.train --horizon all "
            "--gold-path ${inputs.gold_path} --rebuild-features"
        ),
        environment=env,
        compute=ensure_compute(client),
        experiment_name=EXPERIMENT_NAME,
        display_name="afmip-daily-retrain",
        inputs={
            "gold_path": Input(
                type=AssetTypes.URI_FILE,
                path="azureml://datastores/workspaceblobstore/paths/afmip/datasets/gold_dataset.csv",
            ),
        },
        outputs={
            "model_output": Output(
                type=AssetTypes.URI_FOLDER,
                path="azureml://datastores/workspaceblobstore/paths/afmip/models/",
            ),
        },
    )

    schedule = JobSchedule(
        name=SCHEDULE_NAME,
        display_name="AFMIP Daily Model Retraining",
        description="Retrains the ML model daily at 07:00 UTC after new data arrives",
        trigger=RecurrenceTrigger(
            frequency="day",
            interval=1,
            schedule=RecurrencePattern(hours=7, minutes=0),
            time_zone=TimeZone.UTC,
        ),
        create_job=job,
    )

    created = client.schedules.begin_create_or_update(schedule).result()
    logger.info(f"Schedule created: {created.name}")
    logger.info(f"Runs daily at: 07:00 UTC")
    logger.info(f"Status: {created.is_enabled}")


def show_status(client: MLClient):
    try:
        schedule = client.schedules.get(SCHEDULE_NAME)
        print(f"\nSchedule: {schedule.name}")
        print(f"Enabled:  {schedule.is_enabled}")
        print(f"Trigger:  daily at 07:00 UTC")
    except Exception:
        print(f"Schedule '{SCHEDULE_NAME}' not found. Run without --status to create it.")


def disable_schedule(client: MLClient):
    schedule = client.schedules.get(SCHEDULE_NAME)
    schedule.is_enabled = False
    client.schedules.begin_create_or_update(schedule).result()
    logger.info(f"Schedule '{SCHEDULE_NAME}' disabled.")


def main():
    parser = argparse.ArgumentParser(description="Manage AFMIP daily retraining schedule")
    parser.add_argument("--status",  action="store_true", help="Show schedule status")
    parser.add_argument("--disable", action="store_true", help="Disable the schedule")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    client = get_client()

    if args.status:
        show_status(client)
    elif args.disable:
        disable_schedule(client)
    else:
        create_schedule(client)


if __name__ == "__main__":
    main()
