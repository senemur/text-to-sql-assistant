import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine, text
from app.config import settings
from app.executor import (
    ExecutionResult,
    PipelineResult,
    execute_query,
    run_pipeline,
    sanitize_error_message,
    verify_readonly_connection,
)
from app.validator import ValidationError


# ==============================================================================
# 1. DIRECT EXECUTOR SECURITY & READ-ONLY ROLE TESTS
# ==============================================================================

def test_executor_rejects_handcrafted_drop() -> None:
    """Verify execute_query rejects DROP statements directly passed to it."""
    with pytest.raises(ValidationError):
        execute_query("DROP TABLE artist")


def test_executor_rejects_handcrafted_injections() -> None:
    """Verify execute_query rejects multi-statement and injection attacks directly."""
    with pytest.raises(ValidationError):
        execute_query("SELECT 1; DROP TABLE artist;")

    with pytest.raises(ValidationError):
        execute_query("SELECT * FROM customer")


@pytest.mark.integration
def test_executor_runs_valid_query() -> None:
    """Verify execute_query executes valid query and returns ExecutionResult."""
    res = execute_query("SELECT artist_id, name FROM artist LIMIT 3")
    assert isinstance(res, ExecutionResult)
    assert res.row_count <= 3
    assert "artist_id" in res.columns
    assert "name" in res.columns
    assert res.elapsed_ms > 0
    assert "LIMIT 3" in res.sql_executed


@pytest.mark.integration
def test_readonly_transaction_blocks_writes_even_if_validator_bypassed() -> None:
    """
    Defense-in-depth: Even if validator is somehow bypassed or mocked out to return a DDL/DML,
    PostgreSQL's SET TRANSACTION READ ONLY and the llm_readonly role guarantee writes fail.
    """
    with patch("app.executor.validate_and_format_sql") as mock_val:
        # Mock validator to maliciously return a write statement
        mock_val.return_value = "DELETE FROM artist WHERE artist_id = 99999"

        with pytest.raises(Exception) as exc_info:
            execute_query("DELETE FROM artist WHERE artist_id = 99999")

        err_str = str(exc_info.value).lower()
        # Must fail at database level due to read-only transaction or permission denial
        assert "read-only" in err_str or "permission denied" in err_str or "transaction" in err_str


def test_verify_readonly_connection_fails_on_superuser() -> None:
    """Verify startup check raises RuntimeError if connection role is superuser."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # Mock superuser setting as "on"
    mock_conn.execute.return_value.scalar.side_effect = ["on", False]

    with pytest.raises(RuntimeError, match="superuser"):
        verify_readonly_connection(mock_engine)


def test_verify_readonly_connection_fails_on_create_privilege() -> None:
    """Verify startup check raises RuntimeError if role has CREATE privileges."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # Not superuser ("off"), but has CREATE privilege (True)
    mock_conn.execute.return_value.scalar.side_effect = ["off", True]

    with pytest.raises(RuntimeError, match="CREATE privileges"):
        verify_readonly_connection(mock_engine)


def test_sanitize_error_message() -> None:
    """Verify sanitize_error_message strips URLs and credentials."""
    raw_error = Exception(
        "relation 'fake_table' does not exist at character 15\n"
        "postgresql+psycopg2://llm_readonly:secret123@localhost:5432/chinook\n"
        "[SQL: SELECT * FROM fake_table]"
    )
    sanitized = sanitize_error_message(raw_error)
    assert "secret123" not in sanitized
    assert "llm_readonly" not in sanitized
    assert "[DATABASE_URL]" in sanitized
    assert "fake_table" in sanitized


# ==============================================================================
# 2. SELF-CORRECTION LOOP & RETRY TESTS
# ==============================================================================

@patch("app.generator.call_llm")
def test_retry_loop_stops_after_max_retries(mock_call: MagicMock, tmp_path: Path) -> None:
    """Verify pipeline stops after max 2 retries (total 3 attempts) on persistent errors."""
    # Always return a query with forbidden column to trigger ValidationError
    mock_call.return_value = json.dumps(
        {
            "sql": "SELECT email FROM customer;",
            "explanation": "Trying to get emails",
            "needs_clarification": False,
        }
    )

    log_file = tmp_path / "test_attempts.jsonl"
    result = run_pipeline(
        question="Get all emails",
        schema_prompt="Table: customer",
        max_retries=2,
        log_path=log_file,
    )

    assert result.status == "error"
    assert result.attempts == 3  # Initial attempt (1) + Retry 1 (2) + Retry 2 (3)
    assert "blocked sensitive column" in result.error.lower()
    assert mock_call.call_count == 3


@patch("app.generator.call_llm")
@pytest.mark.integration
def test_retry_loop_recovers_on_second_attempt(mock_call: MagicMock, tmp_path: Path) -> None:
    """Verify pipeline self-corrects: attempt 1 fails, LLM receives error feedback, attempt 2 succeeds."""
    attempt_responses = [
        # Attempt 1: Generates invalid SQL accessing blocked email
        json.dumps(
            {
                "sql": "SELECT email FROM customer;",
                "explanation": "Trying email",
                "needs_clarification": False,
            }
        ),
        # Attempt 2: Self-corrects to allowed column customer_id
        json.dumps(
            {
                "sql": "SELECT customer_id, first_name FROM customer LIMIT 2;",
                "explanation": "Self-corrected to allowed columns",
                "needs_clarification": False,
            }
        ),
    ]
    mock_call.side_effect = attempt_responses

    log_file = tmp_path / "test_attempts.jsonl"
    result = run_pipeline(
        question="Show customer info",
        schema_prompt="Table: customer",
        max_retries=2,
        log_path=log_file,
    )

    assert result.status == "success"
    assert result.attempts == 2
    assert result.result is not None
    assert result.result.row_count <= 2

    # Verify that the second call received error feedback in prompt
    second_call_args = mock_call.call_args_list[1][1]
    assert "blocked sensitive column" in second_call_args["user"].lower()


@patch("app.generator.call_llm")
def test_needs_clarification_skips_execution(mock_call: MagicMock, tmp_path: Path) -> None:
    """Verify if LLM flags needs_clarification=True, pipeline returns without executing anything."""
    mock_call.return_value = json.dumps(
        {
            "sql": "",
            "explanation": "The term 'sales trend' is ambiguous without a date range.",
            "needs_clarification": True,
        }
    )

    log_file = tmp_path / "test_attempts.jsonl"
    result = run_pipeline(
        question="Give me sales trends",
        schema_prompt="Table: invoice",
        max_retries=2,
        log_path=log_file,
    )

    assert result.status == "needs_clarification"
    assert result.attempts == 1
    assert result.result is None
    assert "ambiguous" in result.explanation.lower()


# ==============================================================================
# 3. ATTEMPTS LOGGING (JSONL) TESTS
# ==============================================================================

@patch("app.generator.call_llm")
def test_attempts_logged_to_jsonl(mock_call: MagicMock, tmp_path: Path) -> None:
    """Verify every attempt is logged to the JSONL file with correct schema."""
    mock_call.return_value = json.dumps(
        {
            "sql": "SELECT * FROM artist FOR UPDATE;",
            "explanation": "Testing lock",
            "needs_clarification": False,
        }
    )

    log_file = tmp_path / "attempts.jsonl"
    run_pipeline(
        question="Lock artists",
        schema_prompt="Table: artist",
        max_retries=1,
        log_path=log_file,
    )

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # Attempt 1 + Attempt 2

    first_entry = json.loads(lines[0])
    assert first_entry["attempt"] == 1
    assert first_entry["question"] == "Lock artists"
    assert first_entry["outcome"] == "validation_error"
    assert "Lock" in first_entry["error"]
    assert "latency_ms" in first_entry
    assert "timestamp" in first_entry
