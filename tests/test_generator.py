import pytest
from unittest.mock import MagicMock, patch
from app.generator import (
    SQLResponse,
    build_user_prompt,
    generate_sql,
    parse_sql_response,
)


def test_parse_valid_json() -> None:
    """Verify parsing clean JSON string into SQLResponse."""
    raw = (
        '{"sql": "SELECT COUNT(*) AS customer_count FROM customer LIMIT 100;", '
        '"explanation": "Counts total customers.", "needs_clarification": false}'
    )
    res = parse_sql_response(raw)
    assert isinstance(res, SQLResponse)
    assert res.sql == "SELECT COUNT(*) AS customer_count FROM customer LIMIT 100;"
    assert res.explanation == "Counts total customers."
    assert res.needs_clarification is False


def test_parse_markdown_fenced_json() -> None:
    """Verify parsing JSON wrapped in ```json code fences."""
    raw = """```json
{
  "sql": "SELECT ar.name, COUNT(al.album_id) AS albums FROM artist ar JOIN album al ON al.artist_id = ar.artist_id GROUP BY ar.name ORDER BY albums DESC LIMIT 5;",
  "explanation": "Top artists by album count.",
  "needs_clarification": false
}
```"""
    res = parse_sql_response(raw)
    assert isinstance(res, SQLResponse)
    assert "JOIN album" in res.sql
    assert res.explanation == "Top artists by album count."
    assert res.needs_clarification is False


def test_parse_json_with_surrounding_text() -> None:
    """Verify parsing JSON with extraneous explanation text outside braces."""
    raw = """Here is the SQL query you asked for:
{
  "sql": "SELECT * FROM genre LIMIT 10;",
  "explanation": "Returns genres.",
  "needs_clarification": false
}
Hope this helps!"""
    res = parse_sql_response(raw)
    assert isinstance(res, SQLResponse)
    assert res.sql == "SELECT * FROM genre LIMIT 10;"


def test_parse_invalid_json_raises_value_error() -> None:
    """Verify invalid JSON raises ValueError with informative message."""
    invalid_inputs = [
        "This is not json at all.",
        '{"sql": "SELECT 1", "unclosed": true',
        '["not", "a", "dict"]',
    ]
    for inp in invalid_inputs:
        with pytest.raises(ValueError):
            parse_sql_response(inp)


def test_build_user_prompt_includes_error_feedback() -> None:
    """Verify build_user_prompt appends error feedback when present."""
    prompt_without_feedback = build_user_prompt(
        question="How many tracks?",
        schema_prompt="Table: track",
    )
    assert "Previous attempt failed" not in prompt_without_feedback
    assert "How many tracks?" in prompt_without_feedback

    prompt_with_feedback = build_user_prompt(
        question="How many tracks?",
        schema_prompt="Table: track",
        error_feedback="relation 'tracks' does not exist",
    )
    assert "Previous attempt failed" in prompt_with_feedback
    assert "relation 'tracks' does not exist" in prompt_with_feedback


@patch("app.generator.call_llm")
def test_generate_sql_with_mocked_llm(mock_call: MagicMock) -> None:
    """Verify generate_sql properly coordinates call_llm and parses response without calling real API."""
    mock_call.return_value = """{
      "sql": "SELECT billing_country, SUM(total) AS total_sales FROM invoice GROUP BY billing_country ORDER BY total_sales DESC LIMIT 5;",
      "explanation": "Top countries by sales.",
      "needs_clarification": false
    }"""

    response = generate_sql(
        question="What are top countries by sales?",
        schema_prompt="Table: invoice ...",
    )

    assert mock_call.called
    assert isinstance(response, SQLResponse)
    assert "billing_country" in response.sql
    assert response.needs_clarification is False


@patch("app.generator.call_llm")
def test_generate_sql_passes_error_feedback(mock_call: MagicMock) -> None:
    """Verify generate_sql passes error_feedback into user message sent to LLM."""
    mock_call.return_value = '{"sql": "SELECT 1 LIMIT 1;", "explanation": "fixed", "needs_clarification": false}'

    generate_sql(
        question="Which artists?",
        schema_prompt="schema",
        error_feedback="column not found: artists",
    )

    call_args = mock_call.call_args[1]
    assert "column not found: artists" in call_args["user"]
