from app.config import settings
from app.schema import (
    ColumnInfo,
    DatabaseSchema,
    TableInfo,
    extract_schema,
    schema_to_prompt,
)


def test_schema_extract_from_database() -> None:
    """Verify extract_schema connects via read-only role and reflects Chinook schema."""
    schema = extract_schema(sample_rows_count=3)
    assert isinstance(schema, DatabaseSchema)

    # Core tables must exist
    expected_tables = {
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
    assert expected_tables.issubset(set(schema.tables.keys()))

    # Check PK/FK extraction
    album_tbl = schema.tables["album"]
    assert "album_id" in album_tbl.primary_keys
    assert any(
        fk.referred_table == "artist" and "artist_id" in fk.constrained_columns
        for fk in album_tbl.foreign_keys
    )

    # Sample rows must have at most 3 items
    assert len(album_tbl.sample_rows) <= 3


def test_blocked_columns_never_in_schema_or_sample_rows() -> None:
    """Verify blocked columns are completely purged from metadata and sample rows."""
    schema = extract_schema(sample_rows_count=3)

    for table_name, blocked_cols in settings.BLOCKED_COLUMNS.items():
        tbl = schema.tables[table_name]
        extracted_col_names = {c.name for c in tbl.columns}

        # Check column objects
        for blocked_col in blocked_cols:
            assert (
                blocked_col not in extracted_col_names
            ), f"Blocked column '{blocked_col}' was found in {table_name}.columns"

        # Check sample rows
        for row in tbl.sample_rows:
            for blocked_col in blocked_cols:
                assert (
                    blocked_col not in row
                ), f"Blocked column '{blocked_col}' was found in sample rows of {table_name}"


def test_schema_to_prompt_excludes_blocked_columns() -> None:
    """Verify that schema_to_prompt generates prompt text completely free of blocked columns."""
    schema = extract_schema(sample_rows_count=3)
    prompt = schema_to_prompt(schema)

    # Verify unblocked columns ARE present
    assert "customer_id" in prompt
    assert "support_rep_id" in prompt
    assert "hire_date" in prompt
    assert "reports_to" in prompt

    # Verify customer blocked columns: email, phone, fax, address, postal_code
    customer_blocked = ["phone", "fax", "postal_code"]
    # Extract customer section of the prompt
    customer_section = prompt.split("Table: customer")[1].split("Table:")[0]
    for col in customer_blocked:
        assert (
            f"- {col}" not in customer_section
        ), f"Customer blocked column '{col}' found in prompt: {customer_section}"

    # Verify employee blocked columns: email, phone, fax, address, birth_date
    employee_blocked = ["birth_date", "phone", "fax"]
    employee_section = prompt.split("Table: employee")[1].split("Table:")[0]
    for col in employee_blocked:
        assert (
            f"- {col}" not in employee_section
        ), f"Employee blocked column '{col}' found in prompt: {employee_section}"


def test_schema_to_prompt_standalone_mock() -> None:
    """Verify schema_to_prompt works deterministically with synthetic TableInfo."""
    synthetic_schema = DatabaseSchema(
        tables={
            "customer": TableInfo(
                name="customer",
                columns=[
                    ColumnInfo(name="customer_id", data_type="INTEGER"),
                    ColumnInfo(name="first_name", data_type="VARCHAR(40)"),
                    ColumnInfo(name="email", data_type="VARCHAR(60)"),  # blocked
                ],
                primary_keys=["customer_id"],
                foreign_keys=[],
                sample_rows=[
                    {"customer_id": 1, "first_name": "Luis", "email": "luis@test.com"}
                ],
            )
        }
    )

    prompt = schema_to_prompt(synthetic_schema)
    assert "customer_id" in prompt
    assert "first_name" in prompt
    # The defensive filter in schema_to_prompt must strip email
    assert "- email" not in prompt
    assert "luis@test.com" not in prompt
