# ============================================================
# FILE: src/api/config.py
# ============================================================
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Priority (highest to lowest):
    1. Environment variables
    2. .env file
    3. Default values
    """

    # Environment
    environment: str = "development"
    debug: bool = True
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ecg"
    # Redis
    redis_url: str = "redis://localhost:6379"
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    # Model
    model_version: str = "latest"
    model_path: Path = Path("./models")
    # Data
    ptb_xl_path: Path = Path("~/data/ptb-xl/physionet.org/files/ptb-xl/1.0.3")
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # DATABASE_URL or database_url both work
        extra="ignore",  # Ignore extra env vars not in model
    )

    @property
    def ptb_xl_path_resolved(self) -> Path:
        """Return expanded path (handles ~)."""
        return self.ptb_xl_path.expanduser()


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Using lru_cache ensures settings are only loaded once.
    """
    return Settings()


# For convenience, create a default instance
settings = get_settings()
