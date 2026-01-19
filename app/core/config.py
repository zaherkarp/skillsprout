"""Application configuration."""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "SkillSprout"
    env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database
    database_url: str
    database_url_sync: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # O*NET Web Services
    onet_username: str = ""
    onet_password: str = ""
    onet_base_url: str = "https://services.onetcenter.org/ws"
    onet_timeout: int = 30
    onet_max_retries: int = 3

    # Demo Mode
    demo_mode: bool = False

    # Model Configuration
    model_version: str = "v1_baseline"
    model_training_min_samples: int = 50
    exploration_epsilon: float = 0.1

    # Scoring Thresholds
    ready_now_match_threshold: float = 75.0
    ready_now_gap_threshold: float = 25.0
    trainable_match_min: float = 50.0
    trainable_match_max: float = 74.0
    trainable_gap_min: float = 26.0
    trainable_gap_max: float = 55.0

    # Background Tasks
    cache_warm_batch_size: int = 50
    periodic_training_cron: str = "0 2 * * *"

    @property
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode."""
        return self.demo_mode or not self.onet_username or not self.onet_password


# Global settings instance
settings = Settings()
