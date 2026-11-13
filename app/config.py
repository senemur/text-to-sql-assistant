from typing import Literal
from pydantic import Field
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

    # Read-only user (Strictly used for LLM queries & schema reflection)
    DB_READONLY_USER: str = "llm_readonly"
    DB_READONLY_PASSWORD: str = "readonly_secret"
    DATABASE_URL_READONLY: str = (
        "postgresql+psycopg2://llm_readonly:readonly_secret@localhost:5432/chinook"
    )

    # Constraints
    STATEMENT_TIMEOUT_MS: int = 5000
    ROW_CAP: int = 100
    MAX_RETRIES: int = 2

    # LLM Settings (Gemini)
    LLM_PROVIDER: Literal["gemini", "mock"] = "gemini"
    LLM_MODEL: str = Field(default="gemini-2.5-flash", alias="LLM_MODEL")
    GEMINI_API_KEY: str = Field(default="", alias="GEMINI_API_KEY")
    GOOGLE_API_KEY: str = Field(default="", alias="GOOGLE_API_KEY")

    # Sensitive/blocked columns config:
    # customer: email, phone, fax, address, postal_code
    # employee: email, phone, fax, address, birth_date, postal_code
    BLOCKED_COLUMNS: dict[str, set[str]] = {
        "customer": {"email", "phone", "fax", "address", "postal_code"},
        "employee": {"email", "phone", "fax", "address", "birth_date", "postal_code"},
    }

    # Chinook database table allowlist
    CHINOOK_TABLES: set[str] = {
        "album",
        "artist",
        "customer",
        "employee",
        "genre",
        "invoice",
        "invoice_line",
        "media_type",
        "playlist",
        "playlist_track",
        "track",
    }

    # Logging & Eval
    ATTEMPTS_LOG_PATH: str = "eval/attempts.jsonl"

    # API and UI
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501
    STREAMLIT_ORIGIN: str = "http://localhost:8501"
    CORS_ORIGINS: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    RATE_LIMIT_PER_MINUTE: int = 10

    @property
    def effective_api_key(self) -> str:
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY


settings = Settings()
