from contextlib import asynccontextmanager
import time
from typing import Any, Literal
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from app.config import settings
from app.executor import (
    PipelineResult,
    get_executor_engine,
    run_pipeline,
    verify_readonly_connection,
)
from app.schema import extract_schema, schema_to_prompt

from datetime import datetime, timezone

cached_schema_prompt: str = ""
rate_limit_store: dict[str, list[float]] = {}
daily_request_counts: dict[str, int] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan:
    1. Single engine initialization with pool_pre_ping.
    2. Startup verification that database role is read-only.
    3. Extract schema once and cache in-memory.
    """
    global cached_schema_prompt
    engine = get_executor_engine()
    verify_readonly_connection(engine)
    schema = extract_schema(engine=engine)
    cached_schema_prompt = schema_to_prompt(schema)
    yield


app = FastAPI(
    title="Text-to-SQL Assistant API",
    description="Secure, production-ready Text-to-SQL assistant over Chinook database.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration restricted to Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language question about Chinook database (1-500 chars).",
    )


class AskResponse(BaseModel):
    status: Literal["success", "needs_clarification", "blocked", "failed"]
    question: str
    sql: str | None = None
    explanation: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    elapsed_ms: float = 0.0
    attempts: int = 1
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "healthy"
    database: str = "connected"
    version: str = "1.0.0"


def check_rate_limit(client_ip: str) -> None:
    """
    In-memory rate limiter:
    1. Global daily request cap from DAILY_REQUEST_CAP (evicts past days).
    2. Per-client-IP window limiter (evicts expired timestamps and stale client IPs).
    """
    now = time.time()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Enforce global daily request cap and evict past days
    stale_days = [d for d in daily_request_counts if d != today_str]
    for d in stale_days:
        daily_request_counts.pop(d, None)

    current_daily_count = daily_request_counts.get(today_str, 0)
    if current_daily_count >= settings.DAILY_REQUEST_CAP:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Global daily request cap exceeded ({settings.DAILY_REQUEST_CAP} requests/day).",
        )

    # 2. Evict expired entries across all IPs to prevent memory leaks
    window = 60.0
    stale_ips: list[str] = []
    for ip, ts_list in list(rate_limit_store.items()):
        active_ts = [t for t in ts_list if now - t < window]
        if not active_ts:
            stale_ips.append(ip)
        else:
            rate_limit_store[ip] = active_ts

    for ip in stale_ips:
        rate_limit_store.pop(ip, None)

    # 3. Check current client's window
    client_timestamps = rate_limit_store.setdefault(client_ip, [])
    if len(client_timestamps) >= settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute.",
        )

    client_timestamps.append(now)
    daily_request_counts[today_str] = current_daily_count + 1


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint: verifies database connectivity."""
    try:
        engine = get_executor_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return HealthResponse(status="healthy", database="connected")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    """
    Main endpoint for natural language query execution:
    1. Enforces per-IP rate limiting.
    2. Runs query through self-correcting security pipeline.
    3. Returns clean, sanitized response without stack traces or internal DB errors.
    """
    client_ip = request.client.host if request.client else "127.0.0.1"
    check_rate_limit(client_ip)

    global cached_schema_prompt
    schema_prompt = cached_schema_prompt
    if not schema_prompt:
        schema = extract_schema()
        schema_prompt = schema_to_prompt(schema)

    try:
        pipeline_res: PipelineResult = run_pipeline(
            question=payload.question,
            schema_prompt=schema_prompt,
            max_retries=settings.MAX_RETRIES,
        )
    except Exception as exc:
        # Fallback safeguard: never leak internal exceptions or stack traces
        return AskResponse(
            status="failed",
            question=payload.question,
            sql=None,
            explanation=None,
            columns=[],
            rows=[],
            truncated=False,
            elapsed_ms=0.0,
            attempts=1,
            message="Internal processing error encountered.",
        )

    # Construct clean, user-friendly message
    if pipeline_res.status == "success":
        message = "Query executed successfully."
        columns = pipeline_res.result.columns if pipeline_res.result else []
        rows = pipeline_res.result.rows if pipeline_res.result else []
        truncated = pipeline_res.result.truncated if pipeline_res.result else False
        elapsed = pipeline_res.result.elapsed_ms if pipeline_res.result else pipeline_res.total_latency_ms
    elif pipeline_res.status == "blocked":
        message = "Request was blocked due to security or access policy violation."
        columns, rows, truncated, elapsed = [], [], False, pipeline_res.total_latency_ms
    elif pipeline_res.status == "needs_clarification":
        message = pipeline_res.explanation or "The question requires clarification before SQL can be generated."
        columns, rows, truncated, elapsed = [], [], False, pipeline_res.total_latency_ms
    else:  # failed
        message = "Unable to generate a valid SQL query after retry attempts."
        columns, rows, truncated, elapsed = [], [], False, pipeline_res.total_latency_ms

    return AskResponse(
        status=pipeline_res.status,
        question=payload.question,
        sql=pipeline_res.final_sql,
        explanation=pipeline_res.explanation,
        columns=columns,
        rows=rows,
        truncated=truncated,
        elapsed_ms=elapsed,
        attempts=pipeline_res.attempts,
        message=message,
    )
