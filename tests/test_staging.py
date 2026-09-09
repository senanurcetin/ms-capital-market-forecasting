"""Staging guards - WITHOUT ANY LIVE BQ CALLS.

`assert_group_alignment` is the most load-bearing assertion in the project. The market
table was read from one feather file in three column-group passes and reassembled by
POSITION, on the assumption that the passes return rows in the same order. If that
assumption were ever wrong, every market feature would be silently built from mismatched
rows - no error, no NaN, just a quietly wrong model.

It ran clean over 221,756,611 rows. That is evidence the data aligned; it is not evidence
that the check WOULD HAVE CAUGHT a misalignment, because it never saw one. These tests
feed it broken joins and require it to fail, which is the only way to know the guard works.

The same applies to verify_staging, whose no-look-ahead invariant is the structural
argument that this project cannot leak the future.
"""
import pytest
from src.data.staging import _render, assert_group_alignment, verify_staging


class FakeResult:
    def __init__(self, row):
        self._row = row

    def result(self):
        return iter([self._row])


class FakeBQ:
    """Answers get_table/query from canned values; records the SQL it was given."""

    def __init__(self, row=None, existing=("g1", "g2", "g3"), rows_by_table=None):
        self.row = row
        self.existing = existing
        self.rows_by_table = rows_by_table or {}
        self.queries = []

    def get_table(self, table_id):
        from google.cloud.exceptions import NotFound

        if not any(f"_market_{g}" in table_id for g in self.existing):
            raise NotFound(table_id)
        return object()

    def query(self, sql):
        self.queries.append(sql)
        for key, row in self.rows_by_table.items():
            if key in sql:
                return FakeResult(row)
        return FakeResult(self.row)


def _aligned(rows):
    return {"sample_id_mismatch": 0, "seconds_mismatch": 0, "joined_rows": rows}


TRAIN_MARKET_ROWS = 221_756_611


# ------------------------------------------------------- the alignment proof

def test_alignment_passes_when_the_groups_agree():
    bq = FakeBQ(row=_aligned(TRAIN_MARKET_ROWS))
    out = assert_group_alignment("train", bq)
    assert out["sample_id_mismatch"] == 0


def test_a_single_sample_id_mismatch_fails():
    """One mismatched row in 221.7M must be fatal, not rounded away."""
    row = _aligned(TRAIN_MARKET_ROWS) | {"sample_id_mismatch": 1}
    with pytest.raises(AssertionError, match="alignment is BROKEN"):
        assert_group_alignment("train", FakeBQ(row=row))


def test_a_single_seconds_mismatch_fails():
    row = _aligned(TRAIN_MARKET_ROWS) | {"seconds_mismatch": 1}
    with pytest.raises(AssertionError, match="alignment is BROKEN"):
        assert_group_alignment("train", FakeBQ(row=row))


def test_a_short_join_fails_even_when_nothing_mismatches():
    """A join that drops rows produces zero mismatches among the rows it kept.

    Counting mismatches alone would call that success, so the row count is checked
    independently.
    """
    with pytest.raises(AssertionError, match="!="):
        assert_group_alignment("train", FakeBQ(row=_aligned(TRAIN_MARKET_ROWS - 1)))


def test_an_exploded_join_fails_too():
    with pytest.raises(AssertionError, match="!="):
        assert_group_alignment("train", FakeBQ(row=_aligned(TRAIN_MARKET_ROWS * 2)))


def test_missing_raw_tables_give_an_actionable_error():
    """raw was dropped for cost after staging was built; this actually happened.

    The message has to say what to re-run, or the next reader sees only a NotFound
    from somewhere deep in the query.
    """
    bq = FakeBQ(row=_aligned(TRAIN_MARKET_ROWS), existing=())
    with pytest.raises(RuntimeError, match="Re-run"):
        assert_group_alignment("train", bq)


def test_alignment_joins_on_row_id():
    bq = FakeBQ(row=_aligned(TRAIN_MARKET_ROWS))
    assert_group_alignment("train", bq)
    assert "USING (row_id)" in bq.queries[0]


# ------------------------------------------------------- staging verification

def _staging_row(rows, samples, min_sec=0.0):
    return {"row_count": rows, "samples": samples, "min_sec": min_sec, "max_sec": 599.9}


def _staging_bq(**overrides):
    from src.config import load_config

    cfg = load_config()
    rows = {}
    for t in ("market", "order", "transaction"):
        rows[f"{t}_train"] = overrides.get(
            t, _staging_row(cfg.expected_rows["train"][t], cfg.samples["train"]))
    return FakeBQ(rows_by_table=rows)


def test_verify_staging_passes_on_correct_counts():
    out = verify_staging("train", _staging_bq())
    assert len(out) == 3
    assert all(r["rows_match"] and r["samples_match"] and r["no_lookahead"] for r in out)


def test_a_negative_second_is_rejected():
    """min_sec < 0 would mean an event AFTER the prediction instant.

    This is the structural argument that the project cannot leak the future, so it has
    to be enforced rather than assumed.
    """
    from src.config import load_config

    cfg = load_config()
    bad = _staging_row(cfg.expected_rows["train"]["market"], cfg.samples["train"],
                       min_sec=-0.001)
    with pytest.raises(AssertionError, match="FAILED"):
        verify_staging("train", _staging_bq(market=bad))


def test_a_wrong_row_count_is_rejected():
    from src.config import load_config

    cfg = load_config()
    bad = _staging_row(cfg.expected_rows["train"]["order"] - 1, cfg.samples["train"])
    with pytest.raises(AssertionError, match="FAILED"):
        verify_staging("train", _staging_bq(order=bad))


def test_a_wrong_sample_count_is_rejected():
    """Right number of rows, wrong number of samples - a duplicated or dropped sample."""
    from src.config import load_config

    cfg = load_config()
    bad = _staging_row(cfg.expected_rows["train"]["transaction"], cfg.samples["train"] - 1)
    with pytest.raises(AssertionError, match="FAILED"):
        verify_staging("train", _staging_bq(transaction=bad))


# ------------------------------------------------------------- SQL templating

def test_train_sql_carries_the_month_join_and_test_sql_does_not():
    """month exists only for train; test has no time axis at all."""
    template = "{month_select} {month_join} {partition_clause}"
    assert "lbl.month" in _render(template, "train")
    assert "lbl.month" not in _render(template, "test")


def test_the_month_join_is_explicit_not_using():
    """USING (sample_id) is ambiguous for market: g1, g2 and g3 all carry the column.

    The query fails outright rather than silently, but the fix belongs in the template.
    """
    sql = _render("{month_join}", "train")
    assert "ON lbl.sample_id = g1.sample_id" in sql
    assert "USING (sample_id)" not in sql


def test_train_and_test_partition_differently():
    """Train partitions by month; test has none, so it partitions by sample_id range."""
    assert _render("{partition_clause}", "train") != _render("{partition_clause}", "test")
