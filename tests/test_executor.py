from decimal import Decimal
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
from app.generator import LLMResult
from app.validator import PolicyViolationError, ValidationError


# ==============================================================================
# 1. DIRECT EXECUTOR SECURITY & READ-ONLY ROLE TESTS
# ==============================================================================

def test_executor_rejects_handcrafted_drop() -> None:
    """Verify execute_query rejects DROP statements directly passed to it."""
    with pytest.raises(PolicyViolationError):
        execute_query("DROP TABLE artist")


def test_executor_rejects_handcrafted_injections() -> None:
    """Verify execute_query rejects multi-statement and injection attacks directly."""
    with pytest.raises(PolicyViolationError):
        execute_query("SELECT 1; DROP TABLE artist;")

    with pytest.raises(PolicyViolationError):
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
    assert res.truncated is False


@pytest.mark.integration
def test_executor_detects_truncation() -> None:
    """Verify execute_query marks truncated=True when row limit is reached."""
    # Chinook has 275 artists, ask with max_limit=5
    res = execute_query("SELECT artist_id, name FROM artist", max_limit=5)
    assert res.row_count == 5
    assert res.truncated is True


@pytest.mark.integration
def test_decimal_serialized_as_string() -> None:
    """Verify Decimal values (like invoice.total) are serialized as string."""
    res = execute_query("SELECT invoice_id, total FROM invoice LIMIT 1")
    assert isinstance(res.rows[0]["total"], str)


@pytest.mark.integration
def test_readonly_transaction_blocks_writes_even_if_validator_bypassed() -> None:
    """
    Defense-in-depth: Even if validator is somehow bypassed or mocked out to return a DDL/DML,
    PostgreSQL's SET TRANSACTION READ ONLY and the llm_readonly role guarantee writes fail.
    """
    with patch("app.executor.validate_and_format_sql") as mock_val:
        mock_val.return_value = "DELETE FROM artist WHERE artist_id = 99999"

        with pytest.raises(Exception) as exc_info:
            execute_query("DELETE FROM artist WHERE artist_id = 99999")

        err_str = str(exc_info.value).lower()
        assert "read-only" in err_str or "permission denied" in err_str or "transaction" in err_str


def test_verify_readonly_connection_fails_on_superuser() -> None:
    """Verify startup check raises RuntimeError if connection role is superuser."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # is_superuser = 'on'
    mock_conn.execute.return_value.scalar.side_effect = ["on", False]

    with pytest.raises(RuntimeError, match="superuser"):
        verify_readonly_connection(mock_engine)


def test_verify_readonly_connection_fails_on_create_privilege() -> None:
    """Verify startup check raises RuntimeError if role has CREATE privileges."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # is_superuser = 'off', has_schema_privilege = True
    mock_conn.execute.return_value.scalar.side_effect = ["off", True]

    with pytest.raises(RuntimeError, match="CREATE privileges"):
        verify_readonly_connection(mock_engine)


def test_verify_readonly_connection_fails_on_createrole() -> None:
    """Verify startup check raises RuntimeError if role has rolcreaterole or rolcreatedb."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # is_superuser = 'off', has_schema_privilege = False
    mock_conn.execute.return_value.scalar.side_effect = ["off", False]
    # rolcreaterole = True, rolcreatedb = False
    mock_conn.execute.return_value.fetchone.return_value = (True, False)

    with pytest.raises(RuntimeError, match="elevated privileges"):
        verify_readonly_connection(mock_engine)


def test_verify_readonly_connection_fails_on_table_write_privilege() -> None:
    """Verify startup check raises RuntimeError if role has table-level write privileges."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    mock_conn.execute.return_value.scalar.side_effect = ["off", False]
    mock_conn.execute.return_value.fetchone.return_value = (False, False)
    # Write grants present
    mock_conn.execute.return_value.fetchall.return_value = [("artist", "DELETE")]

    with pytest.raises(RuntimeError, match="write privileges"):
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
# 2. SELF-CORRECTION LOOP & POLICY VIOLATION TESTS
# ==============================================================================

@patch("app.generator.call_llm")
def test_policy_violation_stops_immediately_without_retry(
    mock_call: MagicMock, tmp_path: Path
) -> None:
    """Verify that a security policy violation ends with status='blocked' and NEVER triggers a retry."""
    mock_call.return_value = LLMResult(
        text=json.dumps(
            {
                "sql": "SELECT email FROM customer;",
                "explanation": "Attempting sensitive access",
                "needs_clarification": False,
            }
        ),
        input_tokens=50,
        output_tokens=20,
        model="gemini-2.5-flash",
    )

    log_file = tmp_path / "test_attempts.jsonl"
    result = run_pipeline(
        question="Get all emails",
        schema_prompt="Table: customer",
        max_retries=2,
        log_path=log_file,
    )

    # Must be status="blocked", exactly 1 attempt, call_llm called ONLY ONCE
    assert result.status == "blocked"
    assert result.attempts == 1
    assert "blocked sensitive column" in result.error.lower()
    assert mock_call.call_count == 1


@patch("app.generator.call_llm")
@pytest.mark.integration
def test_retry_loop_recovers_on_retryable_syntax_error(
    mock_call: MagicMock, tmp_path: Path
) -> None:
    """Verify retryable errors trigger self-correction retry and succeed when fixed."""
    attempt_responses = [
        # Attempt 1: Malformed SQL (syntax error)
        LLMResult(
            text=json.dumps(
                {
                    "sql": "SELECT artist_id FROM FROM artist;",
                    "explanation": "Typo in query",
                    "needs_clarification": False,
                }
            ),
            input_tokens=40,
            output_tokens=15,
            model="gemini-2.5-flash",
        ),
        # Attempt 2: Self-corrected valid query
        LLMResult(
            text=json.dumps(
                {
                    "sql": "SELECT artist_id, name FROM artist LIMIT 2;",
                    "explanation": "Corrected query",
                    "needs_clarification": False,
                }
            ),
            input_tokens=60,
            output_tokens=25,
            model="gemini-2.5-flash",
        ),
    ]
    mock_call.side_effect = attempt_responses

    log_file = tmp_path / "test_attempts.jsonl"
    result = run_pipeline(
        question="Show artists",
        schema_prompt="Table: artist",
        max_retries=2,
        log_path=log_file,
    )

    assert result.status == "success"
    assert result.attempts == 2
    assert result.result is not None
    assert result.result.row_count <= 2

    # Second call must include error feedback
    second_call_user = mock_call.call_args_list[1][1]["user"].lower()
    assert "syntax error" in second_call_user


@patch("app.generator.call_llm")
def test_retry_loop_stops_after_max_retries_on_persistent_syntax_error(
    mock_call: MagicMock, tmp_path: Path
) -> None:
    """Verify retryable errors stop after max_retries and return status='failed'."""
    mock_call.return_value = LLMResult(
        text=json.dumps(
            {
                "sql": "SELECT * FROM FROM artist;",
                "explanation": "Bad syntax",
                "needs_clarification": False,
            }
        ),
        input_tokens=40,
        output_tokens=15,
        model="gemini-2.5-flash",
    )

    log_file = tmp_path / "test_attempts.jsonl"
    result = run_pipeline(
        question="Show artists",
        schema_prompt="Table: artist",
        max_retries=2,
        log_path=log_file,
    )

    assert result.status == "failed"
    assert result.attempts == 3  # Initial + 2 retries
    assert mock_call.call_count == 3


@patch("app.generator.call_llm")
def test_needs_clarification_skips_execution(
    mock_call: MagicMock, tmp_path: Path
) -> None:
    """Verify if LLM flags needs_clarification=True, pipeline returns without executing anything."""
    mock_call.return_value = LLMResult(
        text=json.dumps(
            {
                "sql": "",
                "explanation": "The term 'sales trend' is ambiguous without a date range.",
                "needs_clarification": True,
            }
        ),
        input_tokens=40,
        output_tokens=20,
        model="gemini-2.5-flash",
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
# 3. ATTEMPTS LOGGING (JSONL) TESTS WITH TOKENS & LATENCY
# ==============================================================================

@patch("app.generator.call_llm")
def test_attempts_logged_to_jsonl_with_token_breakdown(
    mock_call: MagicMock, tmp_path: Path
) -> None:
    """Verify every attempt is logged to the JSONL file with tokens and latency breakdown."""
    mock_call.return_value = LLMResult(
        text=json.dumps(
            {
                "sql": "SELECT * FROM artist FOR UPDATE;",
                "explanation": "Testing lock",
                "needs_clarification": False,
            }
        ),
        input_tokens=125,
        output_tokens=35,
        model="gemini-2.5-flash",
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
    assert len(lines) == 1  # Policy violation ends immediately

    first_entry = json.loads(lines[0])
    assert first_entry["attempt"] == 1
    assert first_entry["question"] == "Lock artists"
    assert first_entry["outcome"] == "blocked"
    assert "Lock" in first_entry["error"]
    assert first_entry["input_tokens"] == 125
    assert first_entry["output_tokens"] == 35
    assert first_entry["model"] == "gemini-2.5-flash"
    assert "llm_ms" in first_entry
    assert "db_ms" in first_entry
    assert "total_ms" in first_entry
