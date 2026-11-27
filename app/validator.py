import sqlglot
from sqlglot import exp

from app.config import settings


class ValidationError(Exception):
    """Base exception for SQL query validation failures."""


class PolicyViolationError(ValidationError):
    """
    Raised when an SQL query violates security policies:
    - Blocked sensitive column access
    - Disallowed table, schema, or table-valued function
    - Forbidden function calls
    - Non-SELECT/UNION statements or DDL/DML/Locking clauses
    - Whole-row references or wildcards on sensitive tables
    - CTE shadowing of base database tables
    """


class QuerySyntaxError(ValidationError):
    """Raised when an SQL query has syntax errors that can be retried by the LLM."""


SAFE_FUNCTIONS: set[str] = {
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "round",
    "coalesce",
    "nullif",
    "lower",
    "upper",
    "length",
    "substring",
    "substr",
    "date_trunc",
    "datetrunc",
    "timestamptrunc",
    "extract",
    "to_char",
    "tochar",
    "concat",
    "cast",
    "trim",
    "ltrim",
    "rtrim",
    "abs",
    "ceil",
    "ceiling",
    "floor",
    "replace",
    "now",
    "current_date",
    "current_timestamp",
    "current_time",
    "greatest",
    "least",
    "case",
    "if",
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
    exp.Into,      # SELECT ... INTO
    exp.Lock,      # FOR UPDATE / FOR SHARE
)


def validate_sql(
    sql: str,
    allowed_tables: set[str],
    blocked_columns: dict[str, set[str]],
    table_columns: dict[str, list[str]] | None = None,
    dialect: str = "postgres",
    max_limit: int = 100,
) -> str:
    """
    Validate and format an SQL query using AST inspection with sqlglot:
    1. Single statement only.
    2. Only SELECT, UNION, INTERSECT, EXCEPT root operations.
    3. No DDL, DML, or administrative commands anywhere in AST (including CTEs).
    4. Safe function allowlist (rejects unknown and table-valued functions).
    5. Table allowlist (handling CTE names and public schema qualification).
    6. Strict column security on EVERY exp.Column (SELECT, WHERE, ORDER BY, GROUP BY, HAVING, JOIN ON).
    7. Whole-row references rejection on tables with blocked columns.
    8. Wildcard '*' expansion to allowed columns of every table in scope (handling aliases & joins).
    9. Handles LIMIT, FETCH FIRST, LIMIT ALL, expression limits, applying limit to whole result.
    """
    cleaned_sql = sql.strip()
    if not cleaned_sql:
        raise QuerySyntaxError("Empty SQL query.")

    # 1. Parse statement(s)
    try:
        statements = sqlglot.parse(cleaned_sql, read=dialect)
    except Exception as exc:
        raise QuerySyntaxError(f"SQL Syntax Error: {exc}") from exc

    if not statements:
        raise QuerySyntaxError("No valid SQL statement found.")

    if len(statements) > 1:
        raise PolicyViolationError(
            f"Multi-statement execution is forbidden. Found {len(statements)} statements."
        )

    ast = statements[0]
    if ast is None:
        raise QuerySyntaxError("Failed to parse SQL statement.")

    # 2. Root statement must be read-only query
    allowed_roots = (exp.Select, exp.Union, exp.Intersect, exp.Except)
    if not isinstance(ast, allowed_roots):
        raise PolicyViolationError(
            f"Only read-only query statements are allowed. Got: {type(ast).__name__}"
        )

    # 3. Check for any nested DDL / DML / Locking expressions
    for expr_type in FORBIDDEN_EXPRESSIONS:
        found = ast.find(expr_type)
        if found:
            raise PolicyViolationError(
                f"Forbidden statement or clause '{expr_type.__name__}' detected in query."
            )

    # 4. Safe function allowlist & reject table-valued functions
    for func in ast.find_all(exp.Func, exp.Anonymous):
        # Skip operator expressions (e.g. AND, OR, NOT, +, -, =, etc.) which subclass Binary/Unary in sqlglot
        if isinstance(func, (exp.Binary, exp.Unary)):
            continue

        func_name = (
            str(func.this).lower()
            if isinstance(func, exp.Anonymous)
            else func.key.lower()
        )
        if func_name and func_name not in SAFE_FUNCTIONS:
            raise PolicyViolationError(
                f"Forbidden function call '{func_name}()' is not in the safe functions allowlist."
            )

    # 5. CTE extraction & Table allowlist with alias mapping
    cte_names: set[str] = {
        cte.alias_or_name.lower()
        for cte in ast.find_all(exp.CTE)
        if cte.alias_or_name
    }
    # Reject CTE shadowing of existing database tables
    base_tables_normalized = {t.lower() for t in allowed_tables}
    shadowed = cte_names & base_tables_normalized
    if shadowed:
        shadowed_str = ", ".join(sorted(shadowed))
        raise PolicyViolationError(
            f"CTE name '{shadowed_str}' shadows an existing database table, which is prohibited."
        )
    effective_allowed = base_tables_normalized | cte_names

    alias_map: dict[str, str] = {}
    referenced_tables: set[str] = set()

    for table_node in ast.find_all(exp.Table):
        # Reject table-valued functions in FROM/JOIN (e.g. FROM pg_ls_dir('/'))
        if not isinstance(table_node.this, (exp.Identifier, exp.Var, str)):
            raise PolicyViolationError(
                f"Table-valued functions or non-standard table expressions are prohibited: {table_node.sql(dialect)}"
            )

        tbl_name = table_node.name.lower()
        if not tbl_name:
            raise QuerySyntaxError(
                f"Invalid or empty table reference in query: {table_node.sql(dialect)}"
            )

        # Check schema qualification: only 'public' or unspecified is permitted
        if table_node.db:
            schema_name = table_node.db.lower()
            if schema_name != "public":
                raise PolicyViolationError(
                    f"Access to schema '{schema_name}' is forbidden. Only 'public' schema is allowed."
                )

        if tbl_name not in effective_allowed:
            raise PolicyViolationError(
                f"Table '{tbl_name}' is not in the allowed tables list."
            )

        referenced_tables.add(tbl_name)
        alias_map[tbl_name] = tbl_name
        if table_node.alias:
            alias_map[table_node.alias.lower()] = tbl_name

    # Normalized blocked columns
    norm_blocked_cols: dict[str, set[str]] = {
        tbl.lower(): {c.lower() for c in cols}
        for tbl, cols in blocked_columns.items()
    }
    tables_with_blocked_cols = {
        tbl for tbl, cols in norm_blocked_cols.items() if cols
    }

    # 6. Reject whole-row references to restricted tables (e.g. SELECT c FROM customer c)
    restricted_aliases = {
        alias: real_tbl
        for alias, real_tbl in alias_map.items()
        if real_tbl in tables_with_blocked_cols
    }

    for col_node in ast.find_all(exp.Column):
        col_name = col_node.name.lower()
        # If an unqualified identifier matches a restricted table's alias, it is a whole-row reference
        if not col_node.table and col_name in restricted_aliases and not isinstance(col_node.this, exp.Star):
            raise PolicyViolationError(
                f"Whole-row reference '{col_name}' to restricted table "
                f"'{restricted_aliases[col_name]}' is prohibited."
            )

    # 7. Check blocked columns on EVERY exp.Column anywhere in the AST
    for col_node in ast.find_all(exp.Column):
        col_name = col_node.name.lower()
        table_qualifier = col_node.table.lower() if col_node.table else None

        if table_qualifier:
            real_tbl = alias_map.get(table_qualifier, table_qualifier)
            if col_name in norm_blocked_cols.get(real_tbl, set()):
                raise PolicyViolationError(
                    f"Access to blocked sensitive column '{col_name}' on table '{real_tbl}' is prohibited."
                )
        else:
            # Unqualified column: check against all referenced tables in query
            for tbl in referenced_tables:
                if col_name in norm_blocked_cols.get(tbl, set()):
                    raise PolicyViolationError(
                        f"Access to blocked sensitive column '{col_name}' on table '{tbl}' is prohibited."
                    )

    # 8. Wildcard '*' expansion to allowed columns of every table in scope (handling aliases & joins)
    # Rather than rejecting SELECT * on tables with blocked columns, expand it to allowed columns only.
    cols_source = table_columns if table_columns is not None else settings.CHINOOK_TABLE_COLUMNS
    allowed_cols_by_table: dict[str, list[str]] = {
        tbl.lower(): [c for c in cols if c.lower() not in norm_blocked_cols.get(tbl.lower(), set())]
        for tbl, cols in cols_source.items()
    }

    for select in ast.find_all(exp.Select):
        tables_in_scope: list[tuple[str, str, bool]] = []
        from_node = select.args.get("from_")
        if from_node and isinstance(from_node.this, exp.Table):
            t_name = from_node.this.name.lower()
            t_alias = from_node.this.alias.lower() if from_node.this.alias else t_name
            tables_in_scope.append((t_name, t_alias, bool(from_node.this.alias)))
        for join in select.args.get("joins", []):
            if isinstance(join.this, exp.Table):
                t_name = join.this.name.lower()
                t_alias = join.this.alias.lower() if join.this.alias else t_name
                tables_in_scope.append((t_name, t_alias, bool(join.this.alias)))

        new_exprs: list[exp.Expression] = []
        for expr in select.expressions:
            if isinstance(expr, exp.Star):
                # Bare wildcard: expand across all tables in scope for this SELECT
                for t_name, t_alias, has_alias in tables_in_scope:
                    cols = allowed_cols_by_table.get(t_name, [])
                    tbl_qual = t_alias if (len(tables_in_scope) > 1 or has_alias) else None
                    for col in cols:
                        new_exprs.append(exp.column(col, table=tbl_qual))
            elif isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star):
                # Qualified wildcard: e.g. e.* or customer.*
                target_alias = expr.table.lower()
                matched_t = next(
                    (tbl for tbl, alias, _ in tables_in_scope if alias == target_alias or tbl == target_alias),
                    None,
                )
                if matched_t:
                    cols = allowed_cols_by_table.get(matched_t, [])
                    for col in cols:
                        new_exprs.append(exp.column(col, table=target_alias))
                else:
                    new_exprs.append(expr)
            else:
                new_exprs.append(expr)
        select.set("expressions", new_exprs)

    # 9. LIMIT and FETCH FIRST handling: clamp or auto-add
    # A) Clamp all FETCH FIRST clauses in AST
    for fetch_node in ast.find_all(exp.Fetch):
        count_expr = fetch_node.args.get("count")
        if count_expr and isinstance(count_expr, exp.Literal) and count_expr.is_int:
            if int(count_expr.this) > max_limit or int(count_expr.this) <= 0:
                fetch_node.set("count", exp.Literal.number(max_limit))
        else:
            fetch_node.set("count", exp.Literal.number(max_limit))

    # B) Clamp all LIMIT clauses in AST
    for limit_node in ast.find_all(exp.Limit):
        lim_expr = limit_node.expression
        if isinstance(lim_expr, exp.Literal) and lim_expr.is_int:
            current_limit = int(lim_expr.this)
            if current_limit > max_limit or current_limit <= 0:
                limit_node.set("expression", exp.Literal.number(max_limit))
        else:
            # Replaces LIMIT ALL or complex expressions with max_limit
            limit_node.set("expression", exp.Literal.number(max_limit))

    # C) Ensure the outer query (including UNION/INTERSECT/EXCEPT) has a limit or fetch
    has_top_limit = ast.args.get("limit") is not None
    has_top_fetch = ast.args.get("fetch") is not None or ast.find(exp.Fetch) is not None

    if not has_top_limit and not has_top_fetch:
        ast.limit(max_limit, copy=False)

    return ast.sql(dialect)


def validate_and_format_sql(sql: str, max_limit: int = 100) -> str:
    """
    Convenience wrapper that passes configuration from settings.
    """
    return validate_sql(
        sql=sql,
        allowed_tables=settings.CHINOOK_TABLES,
        blocked_columns=settings.BLOCKED_COLUMNS,
        table_columns=settings.CHINOOK_TABLE_COLUMNS,
        dialect="postgres",
        max_limit=max_limit,
    )
