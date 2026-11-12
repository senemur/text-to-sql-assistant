import os
from pathlib import Path
import yaml
from app.config import settings


def test_settings_loaded() -> None:
    """Verify all required configurations are loaded via environment/pydantic settings."""
    assert settings.DB_NAME == "chinook"
    assert settings.DB_READONLY_USER == "llm_readonly"
    assert settings.STATEMENT_TIMEOUT_MS == 5000
    assert settings.ROW_CAP == 100
    assert "llm_readonly" in settings.DATABASE_URL_READONLY
    # Ensure admin connection is separate from readonly
    assert settings.DATABASE_URL_ADMIN != settings.DATABASE_URL_READONLY
    assert settings.DB_ADMIN_USER != settings.DB_READONLY_USER


def test_docker_compose_configuration() -> None:
    """Verify docker-compose.yml defines postgres service with Chinook init scripts and healthcheck."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml must exist"

    with compose_path.open("r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    assert "services" in compose
    assert "postgres" in compose["services"]
    postgres_svc = compose["services"]["postgres"]

    assert "ports" in postgres_svc
    assert "5432:5432" in postgres_svc["ports"]

    env = postgres_svc.get("environment", {})
    assert env.get("POSTGRES_DB") == "chinook"

    volumes = postgres_svc.get("volumes", [])
    assert any("docker-entrypoint-initdb.d" in vol for vol in volumes)

    healthcheck = postgres_svc.get("healthcheck", {})
    assert "test" in healthcheck


def test_chinook_sql_initialization_scripts() -> None:
    """Verify SQL init scripts exist with Chinook schema and readonly role permissions."""
    chinook_sql = Path("docker/init/01_chinook.sql")
    assert chinook_sql.exists(), "Chinook DB script must exist"
    assert chinook_sql.stat().st_size > 100000, "Chinook SQL file is too small or incomplete"

    readonly_sql = Path("docker/init/02_create_readonly_role.sql")
    assert readonly_sql.exists(), "Readonly role script must exist"

    sql_content = readonly_sql.read_text(encoding="utf-8")
    assert "llm_readonly" in sql_content
    assert "GRANT SELECT" in sql_content
    assert "statement_timeout" in sql_content
    assert "5s" in sql_content
    assert "REVOKE CREATE" in sql_content
