"""Feature SQL generators - structural guarantees.

These tests validate the generated SQL text without executing it, so look-ahead,
window and naming violations are caught on every change without touching BigQuery.
"""
import itertools
import re

import pytest

from src.config import load_config
from src.features import (
    assemble, common, market_features, order_features, transaction_features,
)

MODULES = {
    "market": (market_features, "mkt_"),
    "order": (order_features, "ord_"),
    "transaction": (transaction_features, "txn_"),
}


def aliases(sql: str, prefix: str) -> set[str]:
    return {a for a in re.findall(r"\sAS\s+([a-z][a-z0-9_]*)", sql) if a.startswith(prefix)}


@pytest.mark.parametrize("table", list(MODULES))
def test_no_future_looking_comparison(table):
    """Window filters must ONLY be 'seconds_before_predict <= W'.

    seconds_before_predict is the distance BACK from the prediction instant, so
    '>=' or '>' would mean "older data" and break the window semantics. Looking
    into the future is already impossible by construction (values are >= 0), but
    this test also catches a filter pointing the wrong way.
    """
    sql = MODULES[table][0].build_sql("train")
    bad = re.findall(r"seconds_before_predict\s*(>=|>)\s*[0-9]", sql)
    assert not bad, f"{table}: window filter points the wrong way: {bad}"
    assert "seconds_before_predict <=" in sql


@pytest.mark.parametrize("table", list(MODULES))
def test_windows_match_config(table):
    """Generated window labels must match the config list exactly."""
    sql = MODULES[table][0].build_sql("train")
    expected = {common.wlabel(w) for w in common.windows(table)}
    found = set(re.findall(r"_(\d+p?\d*s)\b", sql))
    assert expected <= found, f"{table}: missing window(s) {expected - found}"
    assert found <= expected, f"{table}: window(s) not in config {found - expected}"


def test_market_uses_600s_window_and_no_1s():
    """Market window is 600 s; no 1 s window because snapshots arrive every ~3.4 s."""
    sql = market_features.build_sql("train")
    assert "mkt_mid_mean_600s" in sql
    assert "mkt_mid_mean_1s" not in sql
    assert load_config().window.seconds["market"] == 600.0


@pytest.mark.parametrize("table", list(MODULES))
def test_reads_only_its_own_staging_table(table):
    """A feature module must not read another table (granularity mixing)."""
    sql = MODULES[table][0].build_sql("train")
    others = [t for t in MODULES if t != table]
    for other in others:
        assert f".{other}_train`" not in sql, f"{table} SQL reads the {other} table"


@pytest.mark.parametrize("table", list(MODULES))
def test_no_target_or_month_leak(table):
    """Feature SQL must never touch target or month."""
    sql = MODULES[table][0].build_sql("train")
    assert "target" not in sql
    assert not re.search(r"\bmonth\b", sql), f"{table}: month used in the feature layer"


@pytest.mark.parametrize("table", list(MODULES))
def test_all_aliases_prefixed_and_unique(table):
    mod, prefix = MODULES[table]
    sql = mod.build_sql("train")
    found = aliases(sql, prefix)
    assert found, f"{table}: no features generated"
    all_aliases = re.findall(r"\sAS\s+([a-z][a-z0-9_]*)", sql)
    feature_aliases = [a for a in all_aliases if a.startswith(("mkt_", "ord_", "txn_"))]
    assert len(feature_aliases) == len(set(feature_aliases)), f"{table}: duplicate alias"


def test_no_alias_collision_between_modules():
    sets = {t: aliases(m.build_sql("train"), p) for t, (m, p) in MODULES.items()}
    for a, b in itertools.combinations(sets, 2):
        assert not sets[a] & sets[b], f"{a}/{b} collision: {sets[a] & sets[b]}"


def test_market_last_fields_come_from_single_snapshot():
    """*_last fields must be read from a SINGLE STRUCT.

    A per-field ARRAY_AGG(... IGNORE NULLS) let the fields come from DIFFERENT
    snapshots; the measured deviation was 1.994.
    """
    sql = market_features.build_sql("train")
    assert "last_snap" in sql and "EXCEPT (last_snap)" in sql

    # Every *_last field is either read DIRECTLY from the STRUCT or derived only
    # from other *_last fields. Either way, the single-snapshot guarantee holds.
    direct = set(re.findall(r"last_snap\.\w+ AS (\w+)", sql))
    assert direct, "no field is read from the STRUCT"

    for alias in [a for a in aliases(sql, "mkt_") if a.endswith("_last")]:
        if alias in direct:
            continue
        expr = re.search(rf"^(.*?)\s+AS {alias}\b", sql, re.MULTILINE)
        assert expr, f"no expression found for {alias}"
        referenced = {r for r in re.findall(r"\bmkt_\w+", expr.group(1))}
        assert referenced, f"{alias} is neither derived nor read from the STRUCT"
        assert referenced <= direct, (
            f"{alias} is derived from fields outside the single snapshot: {referenced - direct}"
        )

    # Also assert no per-field ARRAY_AGG remains (the signature of the old bug).
    per_field_aggs = re.findall(r"ARRAY_AGG\((?!IF\(mid IS NOT NULL)[^)]*\)\s*IGNORE NULLS"
                                r"\s*ORDER BY seconds_before_predict ASC", sql)
    assert not per_field_aggs, f"per-field ARRAY_AGG still present: {per_field_aggs}"


def test_depth_imbalance_guarded_on_two_sided_book():
    sql = market_features.build_sql("train")
    assert "IF(ap1 IS NULL OR bp1 IS NULL, NULL," in sql


def test_truncation_features_present_for_capped_tables():
    """order/transaction cap at 999 rows per sample -> truncation signal."""
    cfg = load_config()
    for table in cfg.truncation["tables"]:
        sql = MODULES[table][0].build_sql("train")
        prefix = MODULES[table][1]
        assert f"{prefix}is_truncated" in sql
        assert f"{prefix}window_covered" in sql
        assert str(cfg.truncation["row_cap"]) in sql


def test_assemble_includes_label_for_train_only():
    train_sql = assemble.assemble_sql("train")
    test_sql = assemble.assemble_sql("test")
    assert "lbl.month" in train_sql and "lbl.target" in train_sql
    assert "lbl." not in test_sql, "test has NO month/target - the competition omits them"
    assert "PARTITION BY RANGE_BUCKET(month" in train_sql
    assert "PARTITION BY" not in test_sql


def test_assemble_joins_all_three_blocks():
    sql = assemble.assemble_sql("train")
    for block in ("market_train", "order_train", "transaction_train"):
        assert block in sql
    assert sql.count("EXCEPT (sample_id)") == 3
