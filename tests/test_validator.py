import pytest
from app.validator import ValidationError, validate_and_format_sql, validate_sql
from app.config import settings


# ==============================================================================
# 1. PARAMETRIZED ATTACK & VIOLATION TESTS (Must ALL raise ValidationError)
# ==============================================================================

BLOCKED_TEST_CASES = [
    # --- The 11 Attack Queries from Stage 3 Review ---
    ("SELECT c FROM customer c", "Whole-row reference"),
    ("SELECT row_to_json(c) FROM customer c", "Forbidden function"),
    ("SELECT COUNT(*) FROM customer WHERE email LIKE 'a%'", "blocked sensitive column"),
    ("SELECT first_name FROM customer ORDER BY email", "blocked sensitive column"),
    ("SELECT * FROM invoice JOIN customer USING (customer_id)", "Wildcard '*' projection"),
    ("SELECT * FROM pg_ls_dir('/')", "Forbidden function"),
    ("SELECT query_to_xml('select 1', true, true, '')", "Forbidden function"),
    ("SELECT * FROM public.customer", "Wildcard '*' projection"),
    ("SELECT * FROM artist FOR UPDATE", "Forbidden statement or clause 'Lock'"),
    ("SELECT * INTO newt FROM artist", "Forbidden statement or clause 'Into'"),
    ("WITH d AS (DELETE FROM artist RETURNING *) SELECT * FROM d", "Forbidden statement or clause 'Delete'"),

    # --- Blocked Columns in Other AST Clauses ---
    ("SELECT customer_id FROM customer GROUP BY customer_id, email", "blocked sensitive column"),
    ("SELECT customer_id FROM customer GROUP BY customer_id HAVING email IS NOT NULL", "blocked sensitive column"),
    ("SELECT i.invoice_id FROM invoice i JOIN customer c ON c.email = 'test@example.com'", "blocked sensitive column"),
    ("SELECT name FROM artist WHERE artist_id IN (SELECT customer_id FROM customer WHERE email IS NOT NULL)", "blocked sensitive column"),
    ("SELECT c.phone FROM customer c", "blocked sensitive column"),
    ("SELECT c.fax FROM customer c", "blocked sensitive column"),
    ("SELECT c.postal_code FROM customer c", "blocked sensitive column"),
    ("SELECT c.address FROM customer c", "blocked sensitive column"),

    # --- Employee Blocked Columns ---
    ("SELECT e.email FROM employee e", "blocked sensitive column"),
    ("SELECT e.phone FROM employee e", "blocked sensitive column"),
    ("SELECT e.fax FROM employee e", "blocked sensitive column"),
    ("SELECT e.address FROM employee e", "blocked sensitive column"),
    ("SELECT e.birth_date FROM employee e", "blocked sensitive column"),
    ("SELECT first_name FROM employee ORDER BY birth_date", "blocked sensitive column"),
    ("SELECT e FROM employee e", "Whole-row reference"),
    ("SELECT employee FROM employee", "Whole-row reference"),
    ("SELECT * FROM employee", "Wildcard '*' projection"),
    ("SELECT e.* FROM employee e", "Wildcard '*' projection"),

    # --- Whole-Row & Unsafe JSON Serialization ---
    ("SELECT customer FROM customer", "Whole-row reference"),
    ("SELECT to_json(c) FROM customer c", "Forbidden function"),

    # --- Forbidden Functions ---
    ("SELECT pg_sleep(10)", "Forbidden function"),
    ("SELECT * FROM artist WHERE pg_sleep(5) IS NULL", "Forbidden function"),
    ("SELECT pg_read_file('/etc/passwd')", "Forbidden function"),
    ("SELECT pg_terminate_backend(1234)", "Forbidden function"),
    ("SELECT dblink('host=evil dbname=db', 'select 1')", "Forbidden function"),
    ("SELECT current_user()", "Forbidden function"),
    ("SELECT version()", "Forbidden function"),

    # --- Table-Valued Functions in FROM/JOIN ---
    ("SELECT * FROM generate_series(1, 10)", "Forbidden function"),

    # --- DDL / DML Attacks ---
    ("DROP TABLE artist", "Only read-only"),
    ("DROP DATABASE chinook", "Only read-only"),
    ("TRUNCATE TABLE track", "Only read-only"),
    ("DELETE FROM artist WHERE artist_id = 1", "Only read-only"),
    ("INSERT INTO artist (artist_id, name) VALUES (999, 'Hacker')", "Only read-only"),
    ("UPDATE artist SET name = 'Owned'", "Only read-only"),
    ("ALTER TABLE artist DROP COLUMN name", "Only read-only"),
    ("CREATE TABLE test (id INT)", "Only read-only"),

    # --- Multi-Statement Injections & Comment Tricks ---
    ("SELECT 1; DROP TABLE artist;", "Multi-statement"),
    ("SELECT 1; /* safe comment */ DELETE FROM album;", "Multi-statement"),
    ("SELECT 1; SELECT 2;", "Multi-statement"),
    ("SELECT * FROM artist; -- comment", "Multi-statement"),

    # --- Unauthorized Tables & Non-Public Schemas ---
    ("SELECT * FROM users", "not in the allowed tables list"),
    ("SELECT * FROM passwords", "not in the allowed tables list"),
    ("SELECT * FROM pg_catalog.pg_user", "Access to schema 'pg_catalog' is forbidden"),
    ("SELECT * FROM information_schema.tables", "Access to schema 'information_schema' is forbidden"),
    ("SELECT * FROM secret.artist", "Access to schema 'secret' is forbidden"),
]


@pytest.mark.parametrize("query, expected_match", BLOCKED_TEST_CASES)
def test_blocked_queries(query: str, expected_match: str) -> None:
    """Verify that every security attack and policy violation raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        validate_and_format_sql(query)
    assert expected_match.lower() in str(exc_info.value).lower(), (
        f"Expected '{expected_match}' in error message, got '{exc_info.value}' for query '{query}'"
    )


# ==============================================================================
# 2. VALID QUERIES TESTS (Must ALL pass cleanly and format safely)
# ==============================================================================

VALID_QUERIES = [
    # Simple select
    ("SELECT artist_id, name FROM artist", "artist"),
    # Query with trailing semicolon
    ("SELECT artist_id, name FROM artist;", "artist"),
    # Query with block comment
    ("SELECT artist_id, name FROM artist /* comment */ WHERE artist_id = 1", "artist_id = 1"),
    # Query with line comment
    ("SELECT artist_id, name FROM artist -- line comment\nWHERE artist_id = 1", "artist_id = 1"),
    # JOIN query
    (
        "SELECT ar.name, al.title FROM artist ar JOIN album al ON al.artist_id = ar.artist_id",
        "JOIN album",
    ),
    # Safe aggregate on sensitive table (COUNT(*) does not leak row data)
    ("SELECT COUNT(*) AS total_customers FROM customer", "COUNT(*)"),
    ("SELECT COUNT(customer_id) AS total FROM customer", "COUNT(customer_id)"),
    # Group by and order by
    (
        "SELECT billing_country, SUM(total) AS total_sales FROM invoice GROUP BY billing_country ORDER BY total_sales DESC",
        "GROUP BY billing_country",
    ),
    # Safe functions: LOWER, UPPER, LENGTH, SUBSTRING, ROUND, COALESCE
    (
        "SELECT LOWER(name), UPPER(name), LENGTH(name), SUBSTRING(name, 1, 3) FROM artist",
        "LOWER(name)",
    ),
    (
        "SELECT ROUND(total, 2), COALESCE(billing_state, 'N/A') FROM invoice",
        "ROUND(total, 2)",
    ),
    # Safe date extraction
    (
        "SELECT EXTRACT(year FROM invoice_date) AS yr, DATE_TRUNC('month', invoice_date) FROM invoice",
        "EXTRACT(year FROM invoice_date)",
    ),
    # Public schema qualification is allowed for Chinook tables
    ("SELECT name FROM public.artist", "FROM public.artist"),
    # CTE query
    (
        """
        WITH top_artists AS (
            SELECT ar.name, COUNT(al.album_id) AS albums
            FROM artist ar
            JOIN album al ON al.artist_id = ar.artist_id
            GROUP BY ar.name
        )
        SELECT name, albums FROM top_artists ORDER BY albums DESC LIMIT 5
        """,
        "WITH top_artists AS",
    ),
    # UNION query
    (
        "SELECT name FROM artist UNION SELECT name FROM track",
        "UNION",
    ),
]


@pytest.mark.parametrize("query, expected_substring", VALID_QUERIES)
def test_valid_queries_pass(query: str, expected_substring: str) -> None:
    """Verify that legitimate SQL queries pass validation and contain expected clauses."""
    result = validate_and_format_sql(query)
    assert expected_substring.lower() in result.lower()
    assert "limit" in result.lower()


# ==============================================================================
# 3. LIMIT & SET OPERATION VERIFICATION TESTS
# ==============================================================================

def test_limit_auto_added_when_missing() -> None:
    """Verify LIMIT 100 is automatically appended when missing."""
    sql = "SELECT name FROM artist"
    res = validate_and_format_sql(sql)
    assert "LIMIT 100" in res


def test_small_limit_preserved() -> None:
    """Verify LIMIT <= 100 is strictly preserved."""
    sql = "SELECT name FROM artist LIMIT 7"
    res = validate_and_format_sql(sql)
    assert "LIMIT 7" in res


def test_large_limit_clamped() -> None:
    """Verify LIMIT > 100 is clamped to 100."""
    sql = "SELECT name FROM artist LIMIT 5000"
    res = validate_and_format_sql(sql)
    assert "LIMIT 100" in res
    assert "5000" not in res


def test_fetch_first_clamped() -> None:
    """Verify FETCH FIRST 5000 ROWS ONLY is clamped to 100."""
    sql = "SELECT name FROM artist FETCH FIRST 5000 ROWS ONLY"
    res = validate_and_format_sql(sql)
    assert "100" in res
    assert "5000" not in res


def test_union_limit_applies_to_whole_result() -> None:
    """Verify LIMIT on UNION query applies to the entire set operation."""
    sql = "SELECT name FROM artist UNION SELECT name FROM artist LIMIT 5000"
    res = validate_and_format_sql(sql)
    assert "LIMIT 100" in res
    assert "5000" not in res


def test_validate_sql_custom_parameters() -> None:
    """Verify validate_sql works with custom passed allowed_tables and blocked_columns without global settings."""
    custom_tables = {"my_table"}
    custom_blocked = {"my_table": {"secret_col"}}

    # Valid custom
    valid_res = validate_sql(
        sql="SELECT public_col FROM my_table",
        allowed_tables=custom_tables,
        blocked_columns=custom_blocked,
        max_limit=25,
    )
    assert "LIMIT 25" in valid_res

    # Blocked custom
    with pytest.raises(ValidationError, match="blocked sensitive column"):
        validate_sql(
            sql="SELECT secret_col FROM my_table",
            allowed_tables=custom_tables,
            blocked_columns=custom_blocked,
            max_limit=25,
        )
