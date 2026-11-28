"""Application configuration."""
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
    # Loopback by default, so running the app directly (`python -m app.main`)
    # does not expose it to the network. Docker and docker-compose pass
    # `--host 0.0.0.0` explicitly because the container needs it.
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Cross-origin access. Comma-separated list of browser origins allowed to
    # call the API from another site, e.g.
    #   CORS_ALLOW_ORIGINS=https://app.example.com,http://localhost:5173
    # Empty (the default) allows none. The bundled UI is served by this app on
    # the same origin, so it needs no CORS at all. The old default (`*` with
    # credentials) let any web page the user visited read, change and delete
    # profiles on an instance running on their machine, because the browser
    # was told every origin was allowed.
    cors_allow_origins: str = ""

    # Per-profile ownership (app/core/profile_ownership.py). When true, every
    # API call that names a user id must carry that profile's owner token in
    # the X-Profile-Token header. Turn off only for single-user local use or
    # while migrating old profiles.
    profile_tokens_required: bool = True

    # Database
    database_url: str = "sqlite+aiosqlite:///:memory:"
    database_url_sync: str = "sqlite:///:memory:"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # O*NET Web Services (optional — leave credentials blank to run fully offline)
    onet_username: str = ""
    onet_password: str = ""
    onet_base_url: str = "https://services.onetcenter.org/ws"
    onet_timeout: int = 30
    onet_max_retries: int = 3

    # Offline O*NET source (used whenever credentials are blank / demo mode):
    #   "file" -> OfflineONetClient, served from the vendored O*NET snapshot
    #             (onet_offline_data_path, ~1,000 occupations). This is the default.
    #   "mock" -> MockONetClient, the small deterministic fixture set (used by tests).
    onet_offline_source: str = "file"
    onet_offline_data_path: str = "client/data/onet-full.json"

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

    # Authentication
    api_key: str = ""  # Set via API_KEY env var; empty = auth disabled (dev only)
    auth_enabled: bool = False  # Set to True in production

    # Feature Flags
    enable_private_mode: bool = True
    enable_bias_mitigations: bool = True
    enable_lite_mode: bool = True
    enable_session_export: bool = True

    # Background Tasks
    cache_warm_batch_size: int = 50
    periodic_training_cron: str = "0 2 * * *"

    # Run the data-enrichment pipeline on app startup. Off by default so a normal
    # boot is fast and never rewrites the committed occupation_registry.json.
    # Enrichment still runs via Celery beat, the /enrichment/run endpoint, or the
    # CLI (python -m app.services.enrichment_pipeline).
    run_startup_enrichment: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS_ALLOW_ORIGINS split into a clean list (blank entries dropped)."""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode."""
        return self.demo_mode or not self.onet_username or not self.onet_password


# Global settings instance
settings = Settings()
