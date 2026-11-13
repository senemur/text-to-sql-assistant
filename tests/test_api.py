from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from app.api import app, daily_request_counts, rate_limit_store
from app.executor import ExecutionResult, PipelineResult


@pytest.fixture
def client() -> TestClient:
    """Create test client with cleared rate limiter."""
    rate_limit_store.clear()
    daily_request_counts.clear()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Verify /health returns 200 OK and connected status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


@patch("app.api.run_pipeline")
def test_ask_valid_request(mock_pipeline: MagicMock, client: TestClient) -> None:
    """Verify valid request returns 200 OK and proper AskResponse shape."""
    mock_pipeline.return_value = PipelineResult(
        status="success",
        question="How many artists?",
        final_sql="SELECT COUNT(*) AS total FROM artist LIMIT 100",
        explanation="Counts total artists.",
        result=ExecutionResult(
            columns=["total"],
            rows=[{"total": 275}],
            row_count=1,
            truncated=False,
            elapsed_ms=5.4,
            sql_executed="SELECT COUNT(*) AS total FROM artist LIMIT 100",
        ),
        attempts=1,
        total_latency_ms=120.5,
    )

    resp = client.post("/ask", json={"question": "How many artists?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["question"] == "How many artists?"
    assert data["columns"] == ["total"]
    assert data["rows"] == [{"total": 275}]
    assert data["truncated"] is False
    assert data["message"] == "Query executed successfully."


def test_ask_too_long_question_returns_422(client: TestClient) -> None:
    """Verify questions exceeding 500 characters are rejected with 422 Unprocessable Entity."""
    long_q = "a" * 501
    resp = client.post("/ask", json={"question": long_q})
    assert resp.status_code == 422


def test_ask_empty_question_returns_422(client: TestClient) -> None:
    """Verify empty questions are rejected with 422 Unprocessable Entity."""
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


@patch("app.api.run_pipeline")
def test_ask_blocked_response_shape(mock_pipeline: MagicMock, client: TestClient) -> None:
    """Verify policy violations return sanitized status='blocked' without leaking row data."""
    mock_pipeline.return_value = PipelineResult(
        status="blocked",
        question="Show all emails",
        final_sql="SELECT email FROM customer",
        explanation=None,
        result=None,
        error="Access to blocked sensitive column 'email' on table 'customer' is prohibited.",
        attempts=1,
        total_latency_ms=15.0,
    )

    resp = client.post("/ask", json={"question": "Show all emails"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "blocked"
    assert data["rows"] == []
    assert data["columns"] == []
    assert data["message"] == "Request was blocked due to security or access policy violation."


@patch("app.api.run_pipeline")
def test_no_stack_trace_leaks_on_internal_exception(
    mock_pipeline: MagicMock, client: TestClient
) -> None:
    """Verify internal server errors return generic message and never leak stack traces."""
    mock_pipeline.side_effect = RuntimeError(
        "Traceback (most recent call last): Database connection failed with password=super_secret_123"
    )

    resp = client.post("/ask", json={"question": "Show artists"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert data["message"] == "Internal processing error encountered."
    assert "super_secret_123" not in resp.text
    assert "Traceback" not in resp.text


@patch("app.api.run_pipeline")
def test_rate_limiter_blocks_excessive_requests(
    mock_pipeline: MagicMock, client: TestClient
) -> None:
    """Verify rate limiter triggers HTTP 429 after 10 requests per minute."""
    mock_pipeline.return_value = PipelineResult(
        status="success",
        question="q",
        final_sql="SELECT 1",
        explanation="test",
        result=None,
        attempts=1,
        total_latency_ms=1.0,
    )

    # 10 allowed requests
    for i in range(10):
        resp = client.post("/ask", json={"question": f"Question {i}"})
        assert resp.status_code == 200

    # 11th request must trigger 429
    resp_blocked = client.post("/ask", json={"question": "Question 11"})
    assert resp_blocked.status_code == 429
    assert "Rate limit exceeded" in resp_blocked.text


@patch("app.api.run_pipeline")
def test_global_daily_request_cap_triggers_429(
    mock_pipeline: MagicMock, client: TestClient
) -> None:
    """Verify exceeding DAILY_REQUEST_CAP raises 429 Too Many Requests."""
    from datetime import datetime, timezone
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Simulate hitting the daily request limit
    daily_request_counts[today_str] = 1000

    resp = client.post("/ask", json={"question": "Another query"})
    assert resp.status_code == 429
    assert "Global daily request cap exceeded" in resp.text


def test_rate_limiter_evicts_stale_ips() -> None:
    """Verify check_rate_limit purges entries older than 60 seconds and cleans up stale IPs."""
    import time
    from app.api import check_rate_limit, daily_request_counts

    daily_request_counts.clear()
    rate_limit_store.clear()

    # Insert a stale entry (timestamp 100 seconds ago)
    old_time = time.time() - 100.0
    rate_limit_store["192.168.1.50"] = [old_time]

    # Calling check_rate_limit from a different IP should evict the stale IP
    check_rate_limit("127.0.0.1")

    assert "192.168.1.50" not in rate_limit_store
    assert "127.0.0.1" in rate_limit_store
