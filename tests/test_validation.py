"""Data-contract tests.

Like the leakage guard, these do not merely check that valid data passes - they INJECT
each violation and assert the contract catches it. A schema nobody has seen fail is a
schema nobody knows works.

Every constraint tested here encodes something measured during the investigation:
look-ahead impossibility, the price=0 sentinel convention, the recovered side and
order_action encodings, and the 999-row ceiling.
"""
import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors
from src.data.validation import (
    check_descending_within_sample,
    check_row_cap,
    label_schema,
    order_schema,
    transaction_schema,
)


def _order(n=400, seed=0):
    rng = np.random.default_rng(seed)
    sid = np.repeat(np.arange(n // 20, dtype=np.int32), 20)
    secs = np.concatenate([np.sort(rng.uniform(0, 60, 20))[::-1] for _ in range(n // 20)])
    return pd.DataFrame({
        "sample_id": sid,
        "seconds_before_predict": secs.astype(np.float32),
        "price": rng.uniform(0.95, 1.05, n).astype(np.float32),
        "volume": rng.integers(1, 1000, n).astype(np.int32),
        "side": rng.integers(0, 2, n).astype(np.int8),
        "order_action": rng.integers(0, 2, n).astype(np.int8),
    })


def test_valid_order_data_passes():
    order_schema().validate(_order(), lazy=True)


def test_negative_seconds_is_rejected():
    """seconds_before_predict >= 0 is what makes look-ahead structurally impossible."""
    df = _order()
    df.loc[0, "seconds_before_predict"] = -0.5
    with pytest.raises(SchemaErrors):
        order_schema().validate(df, lazy=True)


def test_seconds_beyond_the_window_is_rejected():
    """order's window is 60 s; a value past it means the semantics changed."""
    df = _order()
    df.loc[0, "seconds_before_predict"] = 600.0
    with pytest.raises(SchemaErrors):
        order_schema().validate(df, lazy=True)


def test_unknown_side_is_rejected():
    """side is 0 = BID / 1 = ASK, recovered by measurement. A third value invalidates
    every order-flow feature built on that encoding."""
    df = _order()
    df.loc[0, "side"] = np.int8(2)
    with pytest.raises(SchemaErrors):
        order_schema().validate(df, lazy=True)


def test_unknown_order_action_is_rejected():
    df = _order()
    df.loc[0, "order_action"] = np.int8(7)
    with pytest.raises(SchemaErrors):
        order_schema().validate(df, lazy=True)


def test_negative_price_is_rejected():
    """0 is the empty-level sentinel and is allowed; below 0 is not a price at all."""
    df = _order()
    df.loc[0, "price"] = -1.0
    with pytest.raises(SchemaErrors):
        order_schema().validate(df, lazy=True)


def test_zero_price_is_allowed_because_it_is_the_sentinel():
    df = _order()
    df.loc[0, "price"] = 0.0
    order_schema().validate(df, lazy=True)      # must NOT raise


def test_transaction_side_encoding_enforced():
    df = _order().drop(columns=["order_action"])
    transaction_schema().validate(df, lazy=True)
    df.loc[0, "side"] = np.int8(3)
    with pytest.raises(SchemaErrors):
        transaction_schema().validate(df, lazy=True)


def test_row_cap_violation_is_caught():
    """order/transaction never exceed 999 rows per sample."""
    df = pd.DataFrame({"sample_id": np.zeros(1200, dtype=np.int32)})
    with pytest.raises(ValueError, match="ceiling"):
        check_row_cap(df, "order")


def test_row_cap_not_applied_to_market():
    """market has no such ceiling, so the check must not fire for it."""
    df = pd.DataFrame({"sample_id": np.zeros(1200, dtype=np.int32)})
    check_row_cap(df, "market")                 # must NOT raise


def test_ascending_seconds_within_a_sample_is_caught():
    """File order is chronological because seconds descend; features depend on it."""
    df = _order()
    df.loc[:19, "seconds_before_predict"] = np.linspace(0, 59, 20).astype(np.float32)
    with pytest.raises(ValueError, match="not descending"):
        check_descending_within_sample(df)


def test_label_month_out_of_range_is_rejected():
    n = 100
    df = pd.DataFrame({
        "month": np.arange(n, dtype=np.int16) % 71,
        "sample_id": np.arange(n, dtype=np.int32),
        "target": np.random.default_rng(0).normal(0, 0.0026, n).astype(np.float32),
    })
    label_schema().validate(df, lazy=True)
    df.loc[0, "month"] = np.int16(999)
    with pytest.raises(SchemaErrors):
        label_schema().validate(df, lazy=True)


def test_duplicate_sample_id_in_label_is_rejected():
    n = 50
    df = pd.DataFrame({
        "month": np.zeros(n, dtype=np.int16),
        "sample_id": np.zeros(n, dtype=np.int32),      # all identical
        "target": np.zeros(n, dtype=np.float32),
    })
    with pytest.raises(SchemaErrors):
        label_schema().validate(df, lazy=True)


# ---------------------------------------------- code vs materialised BigQuery tables

class _FakeField:
    def __init__(self, name):
        self.name = name


class _FakeTable:
    def __init__(self, names):
        self.schema = [_FakeField(n) for n in names]


class _FakeBQ:
    """Returns a canned schema per table, so drift can be simulated without BigQuery."""

    def __init__(self, schemas):
        self.schemas = schemas

    def get_table(self, table_id):
        for key, names in self.schemas.items():
            if table_id.endswith(key):
                return _FakeTable(names)
        raise AssertionError(f"unexpected table {table_id}")


def _live_schemas(mutate=None):
    """The schema BigQuery would hold if it matched the code exactly."""
    from src.data.validation import GENERATORS, sql_aliases

    out = {}
    for table, (prefix, module_path) in GENERATORS.items():
        cols = sql_aliases(module_path, prefix) | {"sample_id"}
        out[f"{table}_train"] = mutate(table, cols) if mutate else cols
    return out


def test_schema_check_passes_when_they_agree():
    from src.data.validation import validate_feature_schema

    report = validate_feature_schema("train", bq=_FakeBQ(_live_schemas()))
    assert all(r["in_code"] == r["in_bigquery"] for r in report.values())


def test_a_feature_added_in_code_but_not_rebuilt_is_caught():
    """The real failure: someone edits the SQL and forgets to re-run the generator."""
    from src.data.validation import validate_feature_schema

    def drop_one(table, cols):
        return cols - {sorted(c for c in cols if c != "sample_id")[0]} if table == "order" else cols

    with pytest.raises(AssertionError, match="disagree"):
        validate_feature_schema("train", bq=_FakeBQ(_live_schemas(drop_one)))


def test_a_stale_column_left_in_bigquery_is_caught():
    """The mirror case: a feature removed from code while the table still carries it."""
    from src.data.validation import validate_feature_schema

    def add_one(table, cols):
        return cols | {"mkt_removed_last_year"} if table == "market" else cols

    with pytest.raises(AssertionError, match="disagree"):
        validate_feature_schema("train", bq=_FakeBQ(_live_schemas(add_one)))


def test_the_error_names_the_table_and_what_to_do():
    """A drift message that does not say which table, or to rebuild, wastes the catch."""
    from src.data.validation import validate_feature_schema

    def rename(table, cols):
        return cols | {"txn_typo_alias"} if table == "transaction" else cols

    with pytest.raises(AssertionError) as e:
        validate_feature_schema("train", bq=_FakeBQ(_live_schemas(rename)))
    assert "transaction" in str(e.value) and "Re-run" in str(e.value)


def test_aliases_are_parsed_from_the_sql_not_a_hand_kept_list():
    """One source of truth: the count must track the SQL, not a literal someone updates."""
    from src.data.validation import sql_aliases

    assert len(sql_aliases("src.features.transaction_features", "txn")) == 52
    assert len(sql_aliases("src.features.order_features", "ord")) == 81
