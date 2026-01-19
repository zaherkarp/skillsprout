"""Celery application configuration."""
import logging
from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "skillsprout",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.tasks"],
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3300,  # 55 minutes
)

# Periodic tasks schedule
celery_app.conf.beat_schedule = {
    "train-calibration-model-daily": {
        "task": "app.tasks.tasks.train_calibration_model_task",
        "schedule": crontab(hour=2, minute=0),  # 2 AM daily
    },
}

logger.info("Celery app configured")
