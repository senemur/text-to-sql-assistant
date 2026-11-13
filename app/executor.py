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
from app.generator import LLMResult, SQLResponse, generate_sql
from app.validator import (
    PolicyViolationError,
    QuerySyntaxError,
    ValidationError,
    validate_and_format_sql,
)


class ExecutionResult(BaseModel):
    columns: list[str] = Field(description="Names of columns returned by the query.")
    rows: list[dict[str, Any]] = Field(description="List of returned row dictionaries.")
    row_count: int = Field(description="Number of rows returned.")
    truncated: bool = Field(
        default=False,
        description="True if query results exceeded max_limit and were truncated.",
    )
    elapsed_ms: float = Field(description="Database execution latency in milliseconds.")
    sql_executed: str = Field(description="The sanitized, validated SQL query that was executed.")


class PipelineResult(BaseModel):
    status: Literal["success", "needs_clarification", "blocked", "failed"]
    question: str
    final_sql: str | None = None
    explanation: str | None = None
    result: ExecutionResult | None = None
    error: str | None = None
    attempts: int = Field(description="Total number of attempts made.")
    total_latency_ms: float = Field(description="Total pipeline latency in milliseconds.")


_executor_engine: Engine | None = None


def verify_readonly_connection(engine: Engine) -> None:
    """
    Verify that the database connection is strictly read-only:
    1. Rejects superuser connections.
    2. Rejects connections with CREATE privileges on the public schema.
    3. Rejects roles with rolcreaterole or rolcreatedb.
    4. Rejects ANY INSERT/UPDATE/DELETE/TRUNCATE privilege on any table in the public schema.
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

        # Check rolcreaterole and rolcreatedb
        role_privs = conn.execute(
            text("SELECT rolcreaterole, rolcreatedb FROM pg_roles WHERE rolname = current_user")
        ).fetchone()
        if role_privs:
            createrole, createdb = role_privs
            if createrole or createdb:
                raise RuntimeError(
                    f"Security violation: Configured role has elevated privileges "
                    f"(createrole={createrole}, createdb={createdb})!"
                )

        # Check for any table-level write privileges (INSERT, UPDATE, DELETE, TRUNCATE) on public schema
        write_grants = conn.execute(
            text(
                """
                SELECT table_name, privilege_type
                FROM information_schema.table_privileges
                WHERE grantee IN (current_user, 'PUBLIC')
                  AND table_schema = 'public'
                  AND privilege_type IN ('INSERT', 'UPDATE', 'DELETE', 'TRUNCATE')
                LIMIT 5
                """
            )
        ).fetchall()
        if write_grants:
            violating_tables = ", ".join(f"{row[0]}:{row[1]}" for row in write_grants)
            raise RuntimeError(
                f"Security violation: Configured role has write privileges on public tables ({violating_tables})!"
            )


def get_executor_engine() -> Engine:
    """
    Get or create the singleton SQLAlchemy engine bound strictly to DATABASE_URL_READONLY.
    Verifies that the role is not superuser and has no CREATE or write privileges.
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
    Sanitize error message for LLM error feedback and user presentation.
    Strips raw database credentials, connection strings, hostnames, and internal row dumps.
    """
    if isinstance(exc, PolicyViolationError):
        return f"Policy Violation: {str(exc)}"

    if is_insufficient_privilege(exc):
        return "Security Policy Violation: Database access denied due to insufficient privileges (SQLSTATE 42501)."

    if isinstance(exc, QuerySyntaxError):
        return f"SQL Syntax Error: {str(exc)}"

    if isinstance(exc, ValidationError):
        return f"Validation Error: {str(exc)}"

    raw_msg = str(exc)
    if isinstance(exc, DBAPIError) and hasattr(exc, "orig") and exc.orig is not None:
        raw_msg = str(exc.orig)

    raw_msg = re.sub(r"postgresql(\+\w+)?://[^\s]+", "[DATABASE_URL]", raw_msg)
    raw_msg = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}:\d+\b", "[HOST:PORT]", raw_msg)

    clean_lines = [
        line.strip()
        for line in raw_msg.splitlines()
        if line.strip() and not line.strip().startswith("[SQL:")
    ]
    cleaned = " ".join(clean_lines[:3])
    return f"Database Execution Error: {cleaned}"


def _format_cell_value(val: Any) -> Any:
    """
    Convert database values into serializable primitives:
    - Decimal -> str
    - date/datetime -> ISO 8601 string
    - memoryview/bytes -> str
    """
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return str(val)
    if isinstance(val, (memoryview, bytes)):
        return str(bytes(val))
    return val


def is_insufficient_privilege(exc: Exception) -> bool:
    """Check if exception represents PostgreSQL SQLSTATE 42501 (insufficient_privilege)."""
    if isinstance(exc, DBAPIError) and hasattr(exc, "orig") and exc.orig is not None:
        orig = exc.orig
        pgcode = getattr(orig, "pgcode", None) or getattr(orig, "sqlstate", None)
        if str(pgcode) == "42501":
            return True
    exc_str = str(exc).lower()
    return "42501" in exc_str or "insufficient_privilege" in exc_str or "permission denied" in exc_str


def execute_query(
    sql: str,
    engine: Engine | None = None,
    max_limit: int = 100,
) -> ExecutionResult:
    """
    Single public entry point for SQL query execution:
    1. Calls validate_and_format_sql(sql) and executes ONLY the returned rewritten SQL.
    2. Runs inside an explicit READ ONLY transaction with statement timeout.
    3. Fetches up to max_limit + 1 rows using fetchmany; truncated is True only if the extra row exists.
    4. Maps SQLSTATE 42501 (insufficient_privilege) to PolicyViolationError.
    5. Serializes Decimal as str, datetime as ISO 8601 string.
    6. Returns ExecutionResult with columns, rows, truncated, and elapsed_ms.
    """
    # Step 1: Validate & rewrite query (impossible to bypass)
    # Validate with max_limit + 1 so we can retrieve the extra probe row to detect truncation
    validated_sql = validate_and_format_sql(sql, max_limit=max_limit + 1)

    eng = engine or get_executor_engine()
    start_time = time.perf_counter()

    try:
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
                    result_proxy.fetchmany(max_limit + 1) if result_proxy.returns_rows else []
                )
    except Exception as exc:
        if is_insufficient_privilege(exc):
            raise PolicyViolationError(
                "Database access denied: insufficient privileges (SQLSTATE 42501)."
            ) from exc
        raise

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Truncated only if the extra (max_limit + 1)th row exists
    truncated = len(raw_rows) > max_limit
    if truncated:
        raw_rows = raw_rows[:max_limit]

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
        truncated=truncated,
        elapsed_ms=round(elapsed_ms, 2),
        sql_executed=validated_sql,
    )


def log_attempt(
    log_path: str | Path | None,
    attempt: int,
    question: str,
    generated_sql: str | None,
    validated_sql: str | None,
    outcome: Literal["success", "blocked", "validation_error", "db_error", "needs_clarification", "failed"],
    error: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "",
    llm_ms: float = 0.0,
    db_ms: float = 0.0,
    total_ms: float = 0.0,
) -> None:
    """Log an execution attempt with full token and latency breakdown to attempts.jsonl."""
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
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
        "llm_ms": round(llm_ms, 2),
        "db_ms": round(db_ms, 2),
        "total_ms": round(total_ms, 2),
    }

    with target_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_pipeline(
    question: str,
    schema_prompt: str,
    max_retries: int = 2,
    engine: Engine | None = None,
    log_path: str | Path | None = None,
    few_shot: bool = True,
    self_correction: bool = True,
) -> PipelineResult:
    """
    Self-correction loop pipeline:
    1. Invokes generate_sql to produce candidate SQLResponse (supports few_shot flag).
    2. If needs_clarification is True, stops immediately without executing anything.
    3. Executes query via execute_query.
    4. If PolicyViolationError or SQLSTATE 42501 (insufficient_privilege):
       ends IMMEDIATELY with status="blocked", NOT sent back to LLM.
    5. On retryable errors (QuerySyntaxError or DB error), retries up to max_retries with sanitized error_feedback.
       If self_correction is False, retries are disabled (max_retries=0).
    6. If retries exhausted, returns status="failed".
    7. Logs tokens, llm_ms, db_ms, and outcome to attempts.jsonl.
    """
    pipeline_start = time.perf_counter()
    last_error: str | None = None
    last_sql_response: SQLResponse | None = None
    effective_retries = max_retries if self_correction else 0

    for attempt in range(effective_retries + 1):
        attempt_num = attempt + 1
        attempt_start = time.perf_counter()

        # Step 1: Generate SQL via LLM
        llm_start = time.perf_counter()
        sql_response = generate_sql(
            question=question,
            schema_prompt=schema_prompt,
            error_feedback=last_error,
            few_shot=few_shot,
        )
        llm_ms = (time.perf_counter() - llm_start) * 1000.0
        last_sql_response = sql_response

        # Token usage
        input_tokens = sql_response.llm_result.input_tokens if sql_response.llm_result else 0
        output_tokens = sql_response.llm_result.output_tokens if sql_response.llm_result else 0
        model_name = sql_response.llm_result.model if sql_response.llm_result else ""

        # Step 2: Handle clarification request
        if sql_response.needs_clarification:
            total_elapsed = (time.perf_counter() - attempt_start) * 1000.0
            overall_elapsed = (time.perf_counter() - pipeline_start) * 1000.0
            log_attempt(
                log_path=log_path,
                attempt=attempt_num,
                question=question,
                generated_sql=sql_response.sql,
                validated_sql=None,
                outcome="needs_clarification",
                error=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name,
                llm_ms=llm_ms,
                db_ms=0.0,
                total_ms=total_elapsed,
            )
            return PipelineResult(
                status="needs_clarification",
                question=question,
                final_sql=sql_response.sql,
                explanation=sql_response.explanation,
                result=None,
                error=None,
                attempts=attempt_num,
                total_latency_ms=round(overall_elapsed, 2),
            )

        # Step 3: Execute query
        try:
            db_start = time.perf_counter()
            exec_result = execute_query(
                sql=sql_response.sql,
                engine=engine,
                max_limit=settings.ROW_CAP,
            )
            db_ms = (time.perf_counter() - db_start) * 1000.0
            attempt_elapsed = (time.perf_counter() - attempt_start) * 1000.0
            overall_elapsed = (time.perf_counter() - pipeline_start) * 1000.0

            log_attempt(
                log_path=log_path,
                attempt=attempt_num,
                question=question,
                generated_sql=sql_response.sql,
                validated_sql=exec_result.sql_executed,
                outcome="success",
                error=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name,
                llm_ms=llm_ms,
                db_ms=db_ms,
                total_ms=attempt_elapsed,
            )

            return PipelineResult(
                status="success",
                question=question,
                final_sql=exec_result.sql_executed,
                explanation=sql_response.explanation,
                result=exec_result,
                error=None,
                attempts=attempt_num,
                total_latency_ms=round(overall_elapsed, 2),
            )

        except PolicyViolationError as pol_err:
            # Policy violations terminate IMMEDIATELY with status="blocked".
            # They are NOT sent back to the LLM for self-correction retries.
            sanitized_err = sanitize_error_message(pol_err)
            attempt_elapsed = (time.perf_counter() - attempt_start) * 1000.0
            overall_elapsed = (time.perf_counter() - pipeline_start) * 1000.0

            log_attempt(
                log_path=log_path,
                attempt=attempt_num,
                question=question,
                generated_sql=sql_response.sql,
                validated_sql=None,
                outcome="blocked",
                error=sanitized_err,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name,
                llm_ms=llm_ms,
                db_ms=0.0,
                total_ms=attempt_elapsed,
            )

            return PipelineResult(
                status="blocked",
                question=question,
                final_sql=sql_response.sql,
                explanation=sql_response.explanation,
                result=None,
                error=sanitized_err,
                attempts=attempt_num,
                total_latency_ms=round(overall_elapsed, 2),
            )

        except (QuerySyntaxError, ValidationError) as syn_err:
            # Retryable validation / syntax error
            sanitized_err = sanitize_error_message(syn_err)
            last_error = sanitized_err
            attempt_elapsed = (time.perf_counter() - attempt_start) * 1000.0
            log_attempt(
                log_path=log_path,
                attempt=attempt_num,
                question=question,
                generated_sql=sql_response.sql,
                validated_sql=None,
                outcome="validation_error",
                error=sanitized_err,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name,
                llm_ms=llm_ms,
                db_ms=0.0,
                total_ms=attempt_elapsed,
            )

        except Exception as db_err:
            if is_insufficient_privilege(db_err):
                # SQLSTATE 42501 (insufficient_privilege) terminates IMMEDIATELY with status="blocked".
                # Never retried, never sent to the LLM.
                sanitized_err = sanitize_error_message(db_err)
                attempt_elapsed = (time.perf_counter() - attempt_start) * 1000.0
                overall_elapsed = (time.perf_counter() - pipeline_start) * 1000.0

                log_attempt(
                    log_path=log_path,
                    attempt=attempt_num,
                    question=question,
                    generated_sql=sql_response.sql,
                    validated_sql=None,
                    outcome="blocked",
                    error=sanitized_err,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model_name,
                    llm_ms=llm_ms,
                    db_ms=0.0,
                    total_ms=attempt_elapsed,
                )

                return PipelineResult(
                    status="blocked",
                    question=question,
                    final_sql=sql_response.sql,
                    explanation=sql_response.explanation,
                    result=None,
                    error=sanitized_err,
                    attempts=attempt_num,
                    total_latency_ms=round(overall_elapsed, 2),
                )

            # Retryable database execution error (unknown column, timeout, etc.)
            sanitized_err = sanitize_error_message(db_err)
            last_error = sanitized_err
            attempt_elapsed = (time.perf_counter() - attempt_start) * 1000.0
            log_attempt(
                log_path=log_path,
                attempt=attempt_num,
                question=question,
                generated_sql=sql_response.sql,
                validated_sql=None,
                outcome="db_error",
                error=sanitized_err,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name,
                llm_ms=llm_ms,
                db_ms=0.0,
                total_ms=attempt_elapsed,
            )

    # If retries exhausted without success
    overall_elapsed = (time.perf_counter() - pipeline_start) * 1000.0
    log_attempt(
        log_path=log_path,
        attempt=effective_retries + 1,
        question=question,
        generated_sql=last_sql_response.sql if last_sql_response else None,
        validated_sql=None,
        outcome="failed",
        error=last_error or "Maximum retry attempts exhausted.",
        total_ms=overall_elapsed,
    )

    return PipelineResult(
        status="failed",
        question=question,
        final_sql=last_sql_response.sql if last_sql_response else None,
        explanation=last_sql_response.explanation if last_sql_response else None,
        result=None,
        error=last_error or "Maximum retry attempts exhausted.",
        attempts=effective_retries + 1,
        total_latency_ms=round(overall_elapsed, 2),
    )
