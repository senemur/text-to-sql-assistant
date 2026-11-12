import sqlglot
from sqlglot import exp
from app.config import settings

CHINOOK_TABLES: set[str] = {
    "album",
    "artist",
    "customer",
    "employee",
    "genre",
    "invoice",
    "invoice_line",
    "media_type",
    "playlist",
    "playlist_track",
    "track",
}

FORBIDDEN_FUNCTIONS: set[str] = {
    "pg_sleep",
    "pg_read_file",
    "pg_write_file",
    "pg_read_binary_file",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_reload_conf",
    "dblink",
    "dblink_exec",
    "query_to_xml",
    "system",
    "exec",
}

FORBIDDEN_EXPRESSIONS: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Grant,
    exp.Revoke,
)


class ValidationError(Exception):
    """Raised when an SQL query fails safety or AST validation rules."""
    pass


def validate_and_format_sql(sql: str, max_limit: int = 100) -> str:
    """
    Validate and format an SQL query using AST inspection with sqlglot:
    1. Single statement only (no semicolon chaining / multi-statement).
    2. Only SELECT or UNION root operations.
    3. No DDL, DML, or administrative commands anywhere in AST.
    4. No dangerous functions (pg_sleep, file I/O, backend management).
    5. Strict table allowlist matching Chinook DB tables and dynamically declared CTEs.
    6. Strict column-level security: rejects access to BLOCKED_COLUMNS.
    7. Wildcard SELECT '*' rejection on tables containing sensitive/blocked columns.
    8. Auto-appends or clamps LIMIT to <= max_limit (default 100).
    """
    cleaned_sql = sql.strip()
    if not cleaned_sql:
        raise ValidationError("Empty SQL query.")

    # 1. Parse statement(s)
    try:
        statements = sqlglot.parse(cleaned_sql, read="postgres")
    except Exception as exc:
        raise ValidationError(f"SQL Syntax Error: {exc}") from exc

    if not statements:
        raise ValidationError("No valid SQL statement found.")

    if len(statements) > 1:
        raise ValidationError(
            f"Multi-statement execution is forbidden. Found {len(statements)} statements."
        )

    ast = statements[0]
    if ast is None:
        raise ValidationError("Failed to parse SQL statement.")

    # 2. Root statement must be SELECT or UNION
    if not isinstance(ast, (exp.Select, exp.Union)):
        raise ValidationError(
            f"Only read-only SELECT or UNION statements are allowed. Got: {type(ast).__name__}"
        )

    # 3. Check for any nested DDL / DML expressions
    for expr_type in FORBIDDEN_EXPRESSIONS:
        found = ast.find(expr_type)
        if found:
            raise ValidationError(
                f"Forbidden statement or clause '{expr_type.__name__}' detected in query."
            )

    # 4. Check for forbidden function calls (e.g. pg_sleep, pg_read_file)
    for func in ast.find_all(exp.Func, exp.Anonymous):
        func_name = func.name.lower()
        if func_name in FORBIDDEN_FUNCTIONS:
            raise ValidationError(
                f"Forbidden function call '{func_name}()' is prohibited."
            )

    # 5. Extract CTE names to allow them as valid temporary table names
    cte_names: set[str] = {
        cte.alias_or_name.lower()
        for cte in ast.find_all(exp.CTE)
        if cte.alias_or_name
    }
    allowed_tables = CHINOOK_TABLES | cte_names

    # Check all referenced tables against allowlist
    referenced_tables: set[str] = set()
    for table_node in ast.find_all(exp.Table):
        tbl_name = table_node.name.lower()
        if tbl_name:
            if tbl_name not in allowed_tables:
                raise ValidationError(
                    f"Table '{tbl_name}' is not in the allowed tables list."
                )
            referenced_tables.add(tbl_name)

    # 6. Column-level security: Verify blocked columns are never accessed
    all_blocked_cols: dict[str, set[str]] = {
        tbl: {c.lower() for c in cols}
        for tbl, cols in settings.BLOCKED_COLUMNS.items()
    }

    for col_node in ast.find_all(exp.Column):
        col_name = col_node.name.lower()
        for tbl in referenced_tables:
            blocked_set = all_blocked_cols.get(tbl, set())
            if col_name in blocked_set:
                raise ValidationError(
                    f"Access to blocked sensitive column '{col_name}' on table '{tbl}' is prohibited."
                )

    # 7. Wildcard SELECT '*' check on tables containing blocked columns
    # COUNT(*) is completely safe and allowed; raw column '*' projection on sensitive tables is blocked.
    for star_node in ast.find_all(exp.Star):
        if isinstance(star_node.parent, exp.Count):
            continue  # COUNT(*) is safe

        # Check if any sensitive table is present in the query
        for tbl in referenced_tables:
            if tbl in all_blocked_cols:
                blocked_list = ", ".join(sorted(all_blocked_cols[tbl]))
                raise ValidationError(
                    f"Wildcard '*' projection on table '{tbl}' containing sensitive/blocked columns "
                    f"({blocked_list}) is prohibited. Please specify explicit allowed columns."
                )

    # 8. Manage LIMIT clause: auto-add if missing, clamp to <= max_limit if exceeded
    limit_clause = ast.args.get("limit")
    if limit_clause is None:
        ast.limit(max_limit, copy=False)
    else:
        limit_expr = limit_clause.expression
        if isinstance(limit_expr, exp.Literal) and limit_expr.is_int:
            current_limit = int(limit_expr.this)
            if current_limit > max_limit or current_limit <= 0:
                limit_clause.set("expression", exp.Literal.number(max_limit))
        else:
            # If complex limit expression, clamp to max_limit
            limit_clause.set("expression", exp.Literal.number(max_limit))

    return ast.sql("postgres")
