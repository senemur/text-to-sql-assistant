import json
from pathlib import Path
import sys
from typing import Any
import sqlglot
from sqlglot import exp
from app.executor import execute_query


def check_if_query_exceeds_limit(sql: str, row_count: int) -> tuple[bool, int]:
    """Check if the query without LIMIT produces more rows than row_count."""
    try:
        ast = sqlglot.parse_one(sql, read="postgres")
        ast.set("limit", None)
        ast.set("fetch", None)
        count_sql = f"SELECT COUNT(*) AS total FROM ({ast.sql('postgres')}) AS _unlimited_subq"
        count_res = execute_query(count_sql, max_limit=1)
        total_available = int(count_res.rows[0]["total"])
        return (total_available > row_count, total_available)
    except Exception:
        return (False, row_count)


def format_rows_preview(rows: list[dict[str, Any]], max_rows: int = 3) -> str:
    """Format the first few rows for clean CLI display."""
    preview = rows[:max_rows]
    if not preview:
        return "  (no rows returned)"
    lines = []
    for i, r in enumerate(preview, 1):
        lines.append(f"  Row {i}: {r}")
    if len(rows) > max_rows:
        lines.append(f"  ... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def check_gold_queries(dataset_path: str = "eval/dataset.json") -> int:
    """
    Execute all gold SQL queries with expected_status 'success' via the read-only connection.
    Flags:
    - Execution errors (ERROR)
    - Zero rows returned (ERROR)
    - Exactly 100 rows returned (WARNING - potential row truncation)
    - ordered=false without ORDER BY returning > 1 row only if the full query exceeds the limit (WARNING)
    """
    path = Path(dataset_path)
    if not path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return 1

    items = json.loads(path.read_text(encoding="utf-8"))
    success_items = [item for item in items if item.get("expected_status") == "success"]

    print("================================================================================")
    print(f"Validating Gold SQL Queries ({len(success_items)} items from {dataset_path})")
    print("================================================================================\n")

    error_count = 0
    zero_row_count = 0
    max_limit_count = 0
    clean_count = 0

    for item in success_items:
        q_id = item["id"]
        category = item["category"]
        question = item["question"]
        gold_sql = item.get("gold_sql")
        ordered = item.get("ordered", False)

        print(f"--- [ID {q_id}] ({category}) ---")
        print(f"Question:  {question}")
        print(f"Gold SQL:  {gold_sql}")

        if not gold_sql:
            print("  [ERROR] Expected status is 'success' but gold_sql is missing/empty!")
            error_count += 1
            print()
            continue

        has_order_by = False
        try:
            ast = sqlglot.parse_one(gold_sql, read="postgres")
            has_order_by = ast.find(exp.Order) is not None
        except Exception:
            pass

        flags: list[str] = []
        try:
            res = execute_query(gold_sql, max_limit=100)
            row_count = res.row_count
            print(f"Row Count: {row_count}")
            print("Preview:")
            print(format_rows_preview(res.rows, max_rows=3))

            # 1. 0 rows is an ERROR
            if row_count == 0:
                flags.append("[ERROR] Query returned 0 rows (empty result set).")
                error_count += 1
                zero_row_count += 1

            # 2. Exactly 100 rows warning
            elif row_count == 100:
                max_limit_count += 1
                flags.append("[WARNING] Query returned exactly 100 rows (possible truncation).")

            # 3. ordered=false without ORDER BY returning > 1 row only if full query exceeds LIMIT
            if not ordered and not has_order_by and row_count > 1:
                exceeds, total_avail = check_if_query_exceeds_limit(gold_sql, row_count)
                if exceeds:
                    flags.append(
                        f"[WARNING] ordered=false query without ORDER BY returns a subset of rows ({row_count} of {total_avail}); results depend on non-deterministic row order!"
                    )

            if not any("[ERROR]" in f for f in flags) and not any("[WARNING]" in f for f in flags):
                clean_count += 1

        except Exception as exc:
            error_count += 1
            flags.append(f"[ERROR] Execution failed: {exc}")

        for flag in flags:
            print(f"  {flag}")
        print()

    print("================================================================================")
    print("Summary of Gold Query Verification:")
    print(f"  Total Checked:      {len(success_items)}")
    print(f"  Passed Clean:       {clean_count}")
    print(f"  Returned 0 Rows:    {zero_row_count}")
    print(f"  Returned 100 Rows:  {max_limit_count}")
    print(f"  Errors:             {error_count}")
    print("================================================================================")

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    exit_code = check_gold_queries()
    sys.exit(exit_code)
