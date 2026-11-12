import pytest
from app.validator import ValidationError, validate_and_format_sql


# --- 1. Valid Query & LIMIT Tests ---

def test_valid_simple_select_auto_adds_limit() -> None:
    """Verify that a valid SELECT without LIMIT gets LIMIT 100 auto-added."""
    sql = "SELECT name FROM artist"
    validated = validate_and_format_sql(sql)
    assert "LIMIT 100" in validated
    assert "FROM artist" in validated


def test_valid_select_retains_small_limit() -> None:
    """Verify that an explicit limit <= 100 is preserved."""
    sql = "SELECT name FROM artist LIMIT 15"
    validated = validate_and_format_sql(sql)
    assert "LIMIT 15" in validated


def test_valid_select_clamps_large_limit() -> None:
    """Verify that an explicit limit > 100 is clamped down to 100."""
    sql = "SELECT name FROM artist LIMIT 500"
    validated = validate_and_format_sql(sql)
    assert "LIMIT 100" in validated
    assert "500" not in validated


def test_valid_cte_query_accepted() -> None:
    """Verify CTE table names are recognized and permitted."""
    sql = """
    WITH artist_counts AS (
        SELECT ar.name, COUNT(al.album_id) AS albums
        FROM artist ar
        JOIN album al ON al.artist_id = ar.artist_id
        GROUP BY ar.name
    )
    SELECT name, albums FROM artist_counts ORDER BY albums DESC LIMIT 5
    """
    validated = validate_and_format_sql(sql)
    assert "WITH artist_counts AS" in validated
    assert "LIMIT 5" in validated


def test_valid_union_query_accepted() -> None:
    """Verify UNION query is allowed and gets limit applied."""
    sql = "SELECT name FROM artist UNION SELECT name FROM track"
    validated = validate_and_format_sql(sql)
    assert "UNION" in validated
    assert "LIMIT 100" in validated


# --- 2. Blocked Columns & Wildcard '*' Tests ---

def test_blocked_column_direct_access_rejected() -> None:
    """Verify selecting blocked column (e.g., customer.email) is rejected."""
    blocked_queries = [
        "SELECT email FROM customer",
        "SELECT c.email FROM customer c",
        "SELECT phone, address FROM customer",
        "SELECT postal_code FROM customer",
        "SELECT birth_date FROM employee",
        "SELECT e.fax FROM employee e",
    ]
    for q in blocked_queries:
        with pytest.raises(ValidationError, match="blocked sensitive column"):
            validate_and_format_sql(q)


def test_wildcard_star_on_sensitive_tables_rejected() -> None:
    """Verify SELECT * on tables with blocked columns (customer, employee) is rejected."""
    with pytest.raises(ValidationError, match="Wildcard '\\*' projection"):
        validate_and_format_sql("SELECT * FROM customer")

    with pytest.raises(ValidationError, match="Wildcard '\\*' projection"):
        validate_and_format_sql("SELECT c.* FROM customer c")

    with pytest.raises(ValidationError, match="Wildcard '\\*' projection"):
        validate_and_format_sql("SELECT * FROM employee")


def test_wildcard_count_star_allowed() -> None:
    """Verify COUNT(*) on customer is allowed because it does not project row values."""
    sql = "SELECT COUNT(*) AS total FROM customer"
    validated = validate_and_format_sql(sql)
    assert "COUNT(*)" in validated or "count(*)" in validated.lower()


def test_wildcard_star_on_non_sensitive_table_allowed() -> None:
    """Verify SELECT * on tables with no sensitive columns (e.g. artist, genre) is allowed."""
    sql = "SELECT * FROM artist LIMIT 10"
    validated = validate_and_format_sql(sql)
    assert "FROM artist" in validated


# --- 3. Table Allowlist Tests ---

def test_unauthorized_tables_rejected() -> None:
    """Verify queries accessing non-Chinook or system tables are rejected."""
    unauthorized_queries = [
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT * FROM pg_tables",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM users",
        "SELECT * FROM passwords",
    ]
    for q in unauthorized_queries:
        with pytest.raises(ValidationError, match="not in the allowed tables list"):
            validate_and_format_sql(q)


# --- 4. Injection & DDL/DML Attacks Tests ---

def test_drop_table_rejected() -> None:
    """Verify DROP TABLE statements are rejected."""
    with pytest.raises(ValidationError):
        validate_and_format_sql("DROP TABLE artist")


def test_multi_statement_injection_rejected() -> None:
    """Verify multi-statement chaining with semicolon is blocked."""
    multi_statements = [
        "SELECT 1; DROP TABLE artist;",
        "SELECT * FROM artist; DELETE FROM album;",
        "SELECT 1; SELECT 2;",
    ]
    for q in multi_statements:
        with pytest.raises(ValidationError, match="Multi-statement"):
            validate_and_format_sql(q)


def test_ddl_and_dml_rejected() -> None:
    """Verify INSERT, UPDATE, DELETE, ALTER, TRUNCATE are blocked."""
    attacks = [
        "DELETE FROM artist",
        "INSERT INTO artist (artist_id, name) VALUES (999, 'Hacker')",
        "UPDATE artist SET name = 'Owned'",
        "TRUNCATE TABLE track",
        "ALTER TABLE artist DROP COLUMN name",
    ]
    for q in attacks:
        with pytest.raises(ValidationError):
            validate_and_format_sql(q)


def test_forbidden_function_calls_rejected() -> None:
    """Verify DoS / exploitation functions like pg_sleep and pg_read_file are blocked."""
    dangerous = [
        "SELECT pg_sleep(10)",
        "SELECT * FROM artist WHERE pg_sleep(5) IS NULL",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_terminate_backend(1234)",
    ]
    for q in dangerous:
        with pytest.raises(ValidationError, match="Forbidden function call"):
            validate_and_format_sql(q)


def test_comment_wrapped_attacks() -> None:
    """Verify comments cannot bypass multi-statement or statement type checks."""
    attack = "SELECT 1; /* safe comment */ DROP TABLE artist;"
    with pytest.raises(ValidationError, match="Multi-statement"):
        validate_and_format_sql(attack)
