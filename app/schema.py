from typing import Any
from pydantic import BaseModel
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from app.config import settings


class ColumnInfo(BaseModel):
    name: str
    data_type: str


class ForeignKeyInfo(BaseModel):
    constrained_columns: list[str]
    referred_table: str
    referred_columns: list[str]


class TableInfo(BaseModel):
    name: str
    columns: list[ColumnInfo]
    primary_keys: list[str]
    foreign_keys: list[ForeignKeyInfo]
    sample_rows: list[dict[str, Any]]


class DatabaseSchema(BaseModel):
    tables: dict[str, TableInfo]


def get_readonly_engine() -> Engine:
    """Create a SQLAlchemy engine strictly bound to the read-only credentials."""
    return create_engine(
        settings.DATABASE_URL_READONLY,
        pool_pre_ping=True,
    )


def extract_schema(
    engine: Engine | None = None,
    sample_rows_count: int = 3,
) -> DatabaseSchema:
    """
    Extract schema metadata (tables, columns, PK, FK, and sample rows)
    using SQLAlchemy inspect over the read-only database connection.
    Strictly filters out any column listed in settings.BLOCKED_COLUMNS.
    """
    eng = engine or get_readonly_engine()
    inspector = inspect(eng)
    table_names = inspector.get_table_names(schema="public")

    tables: dict[str, TableInfo] = {}

    with eng.connect() as conn:
        for tbl in table_names:
            blocked_set = settings.BLOCKED_COLUMNS.get(tbl, set())

            # 1. Extract and filter columns
            raw_columns = inspector.get_columns(tbl, schema="public")
            columns: list[ColumnInfo] = []
            allowed_col_names: list[str] = []

            for col in raw_columns:
                col_name = str(col["name"]).lower()
                if col_name in blocked_set:
                    continue
                allowed_col_names.append(col_name)
                col_type = str(col["type"])
                columns.append(ColumnInfo(name=col_name, data_type=col_type))

            # 2. Extract Primary Keys (excluding blocked if any)
            pk_dict = inspector.get_pk_constraint(tbl, schema="public") or {}
            raw_pks = [str(k).lower() for k in pk_dict.get("constrained_columns", [])]
            primary_keys = [pk for pk in raw_pks if pk not in blocked_set]

            # 3. Extract Foreign Keys
            raw_fks = inspector.get_foreign_keys(tbl, schema="public") or []
            foreign_keys: list[ForeignKeyInfo] = []
            for fk in raw_fks:
                c_cols = [str(c).lower() for c in fk.get("constrained_columns", [])]
                # If foreign key uses a blocked column, exclude it
                if any(c in blocked_set for c in c_cols):
                    continue
                ref_table = str(fk.get("referred_table", "")).lower()
                ref_cols = [str(c).lower() for c in fk.get("referred_columns", [])]
                foreign_keys.append(
                    ForeignKeyInfo(
                        constrained_columns=c_cols,
                        referred_table=ref_table,
                        referred_columns=ref_cols,
                    )
                )

            # 4. Extract sample rows ONLY selecting allowed columns
            sample_rows: list[dict[str, Any]] = []
            if allowed_col_names:
                cols_sql = ", ".join(allowed_col_names)
                # Order by primary key if available for deterministic sample rows
                order_clause = f" ORDER BY {primary_keys[0]}" if primary_keys else ""
                query = text(f"SELECT {cols_sql} FROM {tbl}{order_clause} LIMIT :limit")
                result = conn.execute(query, {"limit": sample_rows_count})
                for row in result.mappings():
                    row_dict: dict[str, Any] = {}
                    for col_name in allowed_col_names:
                        val = row.get(col_name)
                        # Ensure value is string/JSON serializable
                        if val is not None and not isinstance(val, (int, float, bool, str)):
                            val = str(val)
                        row_dict[col_name] = val
                    sample_rows.append(row_dict)

            tables[tbl] = TableInfo(
                name=tbl,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                sample_rows=sample_rows,
            )

    return DatabaseSchema(tables=tables)


def schema_to_prompt(schema: DatabaseSchema) -> str:
    """
    Format database schema into an LLM-ready prompt text.
    Strictly verifies that no blocked columns are included.
    """
    lines: list[str] = [
        "### PostgreSQL Database Schema (Chinook):",
        "All tables and columns use lowercase snake_case. No quoting is needed.",
        "",
    ]

    for tbl_name, tbl_info in sorted(schema.tables.items()):
        blocked_set = settings.BLOCKED_COLUMNS.get(tbl_name, set())
        lines.append(f"Table: {tbl_name}")

        # Build column annotations (PK / FK)
        fk_map: dict[str, str] = {}
        for fk in tbl_info.foreign_keys:
            for c_col, r_col in zip(fk.constrained_columns, fk.referred_columns):
                fk_map[c_col] = f"{fk.referred_table}.{r_col}"

        lines.append("  Columns:")
        for col in tbl_info.columns:
            # Defensive verification: skip if accidentally blocked
            if col.name in blocked_set:
                continue

            annotations: list[str] = []
            if col.name in tbl_info.primary_keys:
                annotations.append("PK")
            if col.name in fk_map:
                annotations.append(f"FK -> {fk_map[col.name]}")

            annot_str = f" [{', '.join(annotations)}]" if annotations else ""
            lines.append(f"    - {col.name} ({col.data_type}){annot_str}")

        # Sample rows (ensure blocked columns are filtered)
        if tbl_info.sample_rows:
            lines.append("  Sample Rows:")
            for row in tbl_info.sample_rows:
                filtered_row = {
                    k: v for k, v in row.items() if k not in blocked_set
                }
                lines.append(f"    {filtered_row}")
        lines.append("")

    return "\n".join(lines).strip()
