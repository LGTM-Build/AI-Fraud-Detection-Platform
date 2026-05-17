from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Fradara Fraud Detection Service"
    APP_VERSION: str = "1.0.0"

    PYTHON_API_KEY: str = "sepakati-dengan-backend"
    INTERNAL_API_KEY: str = Field(
        "sepakati-dengan-backend",
        validation_alias=AliasChoices("INTERNAL_API_KEY", "NODEJS_API_KEY"),
    )
    BACKEND_BASE_URL: str = Field(
        "http://localhost:8080",
        validation_alias=AliasChoices("BACKEND_BASE_URL", "NODEJS_BASE_URL"),
    )

    USE_REAL_MODEL: bool = True
    MODELS_DIR: str = "models"
    MAX_JOBS_IN_MEMORY: int = 1000

    HTTP_TIMEOUT: float = 30.0
    HTTP_MAX_RETRIES: int = 3
    HTTP_RETRY_DELAY: float = 2.0

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
