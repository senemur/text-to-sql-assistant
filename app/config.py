from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "chinook"

    # Credentials
    DB_ADMIN_USER: str = "postgres"
    DB_ADMIN_PASSWORD: str = "postgres"
    DATABASE_URL_ADMIN: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/chinook"
    )

    # Read-only user (used for LLM queries)
    DB_READONLY_USER: str = "llm_readonly"
    DB_READONLY_PASSWORD: str = "readonly_secret"
    DATABASE_URL_READONLY: str = (
        "postgresql+psycopg2://llm_readonly:readonly_secret@localhost:5432/chinook"
    )

    # Constraints
    STATEMENT_TIMEOUT_MS: int = 5000
    ROW_CAP: int = 100
    MAX_RETRIES: int = 2

    # LLM Settings
    LLM_PROVIDER: Literal["gemini", "openai", "mock"] = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # API and UI
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501


settings = Settings()
