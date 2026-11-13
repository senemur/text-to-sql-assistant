from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
import time
from typing import Any, Literal
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from app.config import settings
from app.generator import SQLResponse, generate_sql
from app.validator import ValidationError, validate_and_format_sql


class ExecutionResult(BaseModel):
    columns: list[str] = Field(description="Names of columns returned by the query.")
    rows: list[dict[str, Any]] = Field(description="List of returned row dictionaries.")
    row_count: int = Field(description="Number of rows returned.")
    elapsed_ms: float = Field(description="Execution latency in milliseconds.")
    sql_executed: str = Field(description="The sanitized, validated SQL query that was executed.")


class PipelineResult(BaseModel):
    status: Literal["success", "needs_clarification", "error"]
    question: str
    final_sql: str | None = None
    explanation: str | None = None
    result: ExecutionResult | None = None
    error: str | None = None
    attempts: int = Field(description="Total number of attempts made.")
    total_latency_ms: float = Field(description="Total latency across all retries in milliseconds.")


_executor_engine: Engine | None = None


def verify_readonly_connection(engine: Engine) -> None:
    """
    Verify that the database connection is strictly read-only:
    1. Rejects superuser connections.
    2. Rejects connections with CREATE privileges on the public schema.
    """
    with engine.connect() as conn:
        # Check superuser status
        is_superuser = conn.execute(
            text("SELECT current_setting('is_superuser')")
        ).scalar()
        if str(is_superuser).lower() in ("on", "true", "yes", "1"):
            raise RuntimeError(
                "Security violation: Configured database connection is a superuser! "
                "Only read-only non-superuser accounts are permitted."
            )

        # Check CREATE privileges on public schema
        has_create = conn.execute(
            text("SELECT has_schema_privilege(current_user, 'public', 'CREATE')")
        ).scalar()
        if has_create:
            raise RuntimeError(
                "Security violation: Configured database role has CREATE privileges on schema 'public'! "
                "Must be restricted to SELECT only."
            )


def get_executor_engine() -> Engine:
    """
    Get or create the singleton SQLAlchemy engine bound strictly to DATABASE_URL_READONLY.
    Verifies that the role is not superuser and has no CREATE privileges.
    """
    global _executor_engine
    if _executor_engine is None:
        engine = create_engine(
            settings.DATABASE_URL_READONLY,
            pool_pre_ping=True,
        )
        verify_readonly_connection(engine)
        _executor_engine = engine
    return _executor_engine


def sanitize_error_message(exc: Exception) -> str:
    """
    Sanitize error message for LLM error feedback.
    Strips raw database credentials, connection strings, hostnames, and internal row dumps.
    """
    if isinstance(exc, ValidationError):
        return f"SQL Validation Error: {str(exc)}"

    raw_msg = str(exc)

    # If it is a SQLAlchemy DBAPI error, extract the driver message
    if isinstance(exc, DBAPIError) and hasattr(exc, "orig") and exc.orig is not None:
        raw_msg = str(exc.orig)

    # Remove connection URLs or passwords if accidentally included
    raw_msg = re.sub(r"postgresql(\+\w+)?://[^\s]+", "[DATABASE_URL]", raw_msg)
    # Remove internal IP or host details
    raw_msg = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}:\d+\b", "[HOST:PORT]", raw_msg)

    # Clean leading/trailing noise
    clean_lines = [
        line.strip()
        for line in raw_msg.splitlines()
        if line.strip() and not line.strip().startswith("[SQL:")
    ]
    cleaned = " ".join(clean_lines[:3])
    return f"Database Execution Error: {cleaned}"


def _format_cell_value(val: Any) -> Any:
    """Convert database values into JSON-serializable primitives."""
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, memoryview):
        return str(bytes(val))
    return val


def execute_query(
    sql: str,
    engine: Engine | None = None,
    max_limit: int = 100,
) -> ExecutionResult:
    """
    Single public entry point for SQL query execution:
    1. Calls validate_and_format_sql(sql) and executes ONLY the returned rewritten SQL.
    2. Runs inside an explicit READ ONLY transaction with statement timeout.
    3. Fetches up to max_limit rows using fetchmany.
    4. Returns ExecutionResult with columns, rows, and elapsed_ms.
    """
    # Step 1: Validate & rewrite query (impossible to bypass)
    validated_sql = validate_and_format_sql(sql, max_limit=max_limit)

    eng = engine or get_executor_engine()
    start_time = time.perf_counter()

    with eng.connect() as conn:
        with conn.begin():
            # Enforce read-only transaction and statement timeout
            conn.execute(text("SET TRANSACTION READ ONLY;"))
            conn.execute(
                text(f"SET LOCAL statement_timeout = '{settings.STATEMENT_TIMEOUT_MS}ms';")
            )

            result_proxy = conn.execute(text(validated_sql))

            columns: list[str] = (
                list(result_proxy.keys()) if result_proxy.returns_rows else []
            )
            raw_rows = (
                result_proxy.fetchmany(max_limit) if result_proxy.returns_rows else []
            )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        row_dict = {
            col: _format_cell_value(val) for col, val in zip(columns, row)
        }
        rows.append(row_dict)

    return ExecutionResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        elapsed_ms=round(elapsed_ms, 2),
        sql_executed=validated_sql,
    )


def log_attempt(
    log_path: str | Path | None,
    attempt: int,
    question: str,
    generated_sql: str | None,
    validated_sql: str | None,
    outcome: Literal["success", "validation_error", "db_error", "needs_clarification"],
    error: str | None,
    latency_ms: float,
) -> None:
    """Log an execution attempt to a JSONL file for audit and evaluation."""
    target_path = Path(log_path or settings.ATTEMPTS_LOG_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "question": question,
        "generated_sql": generated_sql,
        "validated_sql": validated_sql,
        "outcome": outcome,
        "error": error,
        "latency_ms": round(latency_ms, 2),
    }

    with target_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_pipeline(
    question: str,
    schema_prompt: str,
    max_retries: int = 2,
    engine: Engine | None = None,
    log_path: str | Path | None = None,
) -> PipelineResult:
    """
    Self-correction loop pipeline:
    1. Invokes generate_sql to produce candidate SQLResponse.
    2. If needs_clarification is True, stops immediately without executing anything.
    3. Executes query via execute_query.
    4. On ValidationError or DB error, retries up to max_retries with sanitized error_feedback.
    5. Logs every attempt to JSONL.
    """
    pipeline_start = time.perf_counter()
    last_error: str | None = None
    last_sql_response: SQLResponse | None = None
    last_validated_sql: str | None = None

    for attempt in range(max_retries + 1):
        attempt_num = attempt + 1
        attempt_start = time.perf_counter()

        # Step 1: Generate SQL (with optional error feedback from previous attempt)
        sql_response = generate_sql(
            question=question,
            schema_prompt=schema_prompt,
            error_feedback=last_error,
        )
        last_sql_response = sql_response

        # Step 2: Handle clarification request
        if sql_response.needs_clarification:
            latency = (time.perf_counter() - attempt_start) * 1000.0
            total_latency = (time.perf_counter() - pipeline_start) * 1000.0
            log_attempt(
                log_path=log_path,
                attempt=attempt_num,
                question=question,
                generated_sql=sql_response.sql,
                validated_sql=None,
                outcome="needs_clarification",
                error=None,
                latency_ms=latency,
            )
            return PipelineResult(
                status="needs_clarification",
                question=question,
                final_sql=sql_response.sql,
                explanation=sql_response.explanation,
                result=None,
                error=None,
                attempts=attempt_num,
                total_latency_ms=round(total_latency, 2),
            )

        # Step 3: Execute query
        try:
            exec_result = execute_query(
                sql=sql_response.sql,
                engine=engine,
                max_limit=settings.ROW_CAP,
            )
            latency = (time.perf_counter() - attempt_start) * 1000.0
            total_latency = (time.perf_counter() - pipeline_start) * 1000.0

            log_attempt(
                log_path=log_path,
                attempt=attempt_num,
                question=question,
                generated_sql=sql_response.sql,
                validated_sql=exec_result.sql_executed,
                outcome="success",
                error=None,
                latency_ms=latency,
            )

            return PipelineResult(
                status="success",
                question=question,
                final_sql=exec_result.sql_executed,
                explanation=sql_response.explanation,
                result=exec_result,
                error=None,
                attempts=attempt_num,
                total_latency_ms=round(total_latency, 2),
            )

        except ValidationError as val_err:
            latency = (time.perf_counter() - attempt_start) * 1000.0
            sanitized_err = sanitize_error_message(val_err)
            last_error = sanitized_err
            log_attempt(
                log_path=log_path,
                attempt=attempt_num,
                question=question,
                generated_sql=sql_response.sql,
                validated_sql=None,
                outcome="validation_error",
                error=sanitized_err,
                latency_ms=latency,
            )

        except Exception as db_err:
            latency = (time.perf_counter() - attempt_start) * 1000.0
            sanitized_err = sanitize_error_message(db_err)
            last_error = sanitized_err
            log_attempt(
                log_path=log_path,
                attempt=attempt_num,
                question=question,
                generated_sql=sql_response.sql,
                validated_sql=last_validated_sql,
                outcome="db_error",
                error=sanitized_err,
                latency_ms=latency,
            )

    # If retries exhausted
    total_latency = (time.perf_counter() - pipeline_start) * 1000.0
    return PipelineResult(
        status="error",
        question=question,
        final_sql=last_sql_response.sql if last_sql_response else None,
        explanation=last_sql_response.explanation if last_sql_response else None,
        result=None,
        error=last_error or "Maximum retry attempts exhausted.",
        attempts=max_retries + 1,
        total_latency_ms=round(total_latency, 2),
    )
