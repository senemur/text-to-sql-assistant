from decimal import Decimal
import json
from pathlib import Path
import pytest
from app.schema import DatabaseSchema, TableInfo, ColumnInfo, retrieve_relevant_tables
from eval.run_eval import compare_result_sets, normalize_value


def test_dataset_json_structure_and_categories() -> None:
    """Verify eval/dataset.json adheres to schema and expected category distributions."""
    dataset_file = Path("eval/dataset.json")
    assert dataset_file.exists()

    data = json.loads(dataset_file.read_text(encoding="utf-8"))
    assert len(data) == 60

    category_counts: dict[str, int] = {}
    for item in data:
        assert "id" in item
        assert "category" in item
        assert "question" in item
        assert "gold_sql" in item
        assert "ordered" in item
        assert "expected_status" in item
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1

    assert category_counts["simple_filter"] == 8
    assert category_counts["join"] == 10
    assert category_counts["aggregation"] == 10
    assert category_counts["subquery_cte"] == 8
    assert category_counts["date"] == 6
    assert category_counts["ambiguous"] == 6
    assert category_counts["attack"] == 12


def test_compare_result_sets_exact_match() -> None:
    """Verify compare_result_sets passes for exact match."""
    gold = [{"id": 1, "name": "Rock"}, {"id": 2, "name": "Jazz"}]
    pred = [{"id": 1, "name": "Rock"}, {"id": 2, "name": "Jazz"}]
    assert compare_result_sets(gold, pred, ordered=True) is True
    assert compare_result_sets(gold, pred, ordered=False) is True


def test_compare_result_sets_order_insensitive() -> None:
    """Verify compare_result_sets matches permuted rows when ordered=False."""
    gold = [{"id": 1, "val": 10.5}, {"id": 2, "val": 20.0}]
    pred = [{"id": 2, "val": 20.0}, {"id": 1, "val": 10.5}]
    assert compare_result_sets(gold, pred, ordered=False) is True
    assert compare_result_sets(gold, pred, ordered=True) is False


def test_compare_result_sets_ignores_column_aliases() -> None:
    """Verify compare_result_sets ignores column aliases and compares by row values."""
    gold = [{"artist": "AC/DC", "albums": 2}]
    pred = [{"name": "AC/DC", "count": 2}]
    assert compare_result_sets(gold, pred, ordered=False) is True


def test_compare_result_sets_numeric_tolerance() -> None:
    """Verify numeric comparisons tolerate Decimal vs float and small precision variances."""
    gold = [{"total": Decimal("15.500")}]
    pred = [{"sum": 15.5}]
    assert compare_result_sets(gold, pred, ordered=False) is True


def test_compare_result_sets_detects_mismatch() -> None:
    """Verify compare_result_sets returns False when row counts or values differ."""
    gold = [{"id": 1, "val": 10}]
    pred = [{"id": 1, "val": 99}]
    assert compare_result_sets(gold, pred, ordered=False) is False

    pred_extra = [{"id": 1, "val": 10}, {"id": 2, "val": 20}]
    assert compare_result_sets(gold, pred_extra, ordered=False) is False


def test_retrieve_relevant_tables() -> None:
    """Verify keyword scoring correctly identifies relevant tables for schema retrieval."""
    schema = DatabaseSchema(
        tables={
            "artist": TableInfo(
                name="artist",
                columns=[ColumnInfo(name="artist_id", data_type="INTEGER"), ColumnInfo(name="name", data_type="VARCHAR")],
                primary_keys=["artist_id"],
            ),
            "album": TableInfo(
                name="album",
                columns=[ColumnInfo(name="album_id", data_type="INTEGER"), ColumnInfo(name="title", data_type="VARCHAR"), ColumnInfo(name="artist_id", data_type="INTEGER")],
                primary_keys=["album_id"],
            ),
            "invoice": TableInfo(
                name="invoice",
                columns=[ColumnInfo(name="invoice_id", data_type="INTEGER"), ColumnInfo(name="total", data_type="NUMERIC")],
                primary_keys=["invoice_id"],
            ),
        }
    )

    retrieved = retrieve_relevant_tables("Which artists have the most albums?", schema, top_k=2)
    assert "artist" in retrieved
    assert "album" in retrieved
    assert "invoice" not in retrieved


def test_compute_cache_key_hash() -> None:
    """Verify cache key is deterministic SHA-256 and sensitive to all parameters."""
    from eval.run_eval import compute_cache_key

    k1 = compute_cache_key("cfg1", "model1", "sys1", "sch1", "q1")
    k2 = compute_cache_key("cfg1", "model1", "sys1", "sch1", "q1")
    assert k1 == k2
    assert len(k1) == 64  # SHA-256 hexdigest length

    # Changing any component changes the hash
    assert k1 != compute_cache_key("cfg2", "model1", "sys1", "sch1", "q1")
    assert k1 != compute_cache_key("cfg1", "model2", "sys1", "sch1", "q1")
    assert k1 != compute_cache_key("cfg1", "model1", "sys2", "sch1", "q1")
    assert k1 != compute_cache_key("cfg1", "model1", "sys1", "sch2", "q1")
    assert k1 != compute_cache_key("cfg1", "model1", "sys1", "sch1", "q2")


def test_check_rows_for_blocked_values() -> None:
    """Verify detection of blocked values in result rows."""
    from eval.run_eval import check_rows_for_blocked_values

    blocked_vals = {"secret@example.com", "123 fake street"}

    # Clean rows -> no leak
    clean_rows = [{"first_name": "John", "total": "100"}]
    has_leak, val = check_rows_for_blocked_values(clean_rows, blocked_vals)
    assert has_leak is False
    assert val is None

    # Leaked row -> detected
    leaked_rows = [{"first_name": "John", "email": "SECRET@EXAMPLE.COM"}]
    has_leak, val = check_rows_for_blocked_values(leaked_rows, blocked_vals)
    assert has_leak is True
    assert val == "SECRET@EXAMPLE.COM"


@pytest.mark.integration
def test_check_gold_queries_executes_without_errors() -> None:
    """Verify check_gold_queries runs against Chinook DB and returns exit code 0."""
    from eval.check_gold import check_gold_queries
    exit_code = check_gold_queries("eval/dataset.json")
    assert exit_code == 0


