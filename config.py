from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Fradara — AI Fraud Detection Service"
    APP_VERSION: str = "1.0.0"

    PYTHON_API_KEY: str = "sepakati-dengan-backend"
    NODEJS_API_KEY: str = "sepakati-dengan-backend"
    NODEJS_BASE_URL: str = "http://localhost:3000"

    USE_REAL_MODEL: bool = True
    MODELS_DIR: str = "models"
    MAX_JOBS_IN_MEMORY: int = 1000

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()