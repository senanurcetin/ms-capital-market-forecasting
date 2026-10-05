"""The pure logic inside the evaluation scripts that otherwise only run against live data.

ablation, drift, drift_robustness and prediction_geometry were at 0-36% coverage because their
run() wrappers need BigQuery or hours of training. The pieces that decide what they report are
small and deterministic, so those are tested here, with a fake BigQuery client where needed.
"""
import numpy as np
import pandas as pd
import pytest
from src.evaluation import ablation, drift, drift_robustness, prediction_geometry

# --- ablation ------------------------------------------------------------------------------

def test_family_is_the_prefix_before_the_first_underscore():
    assert ablation.family_of("mkt_spread_mean_5s") == "mkt"
    assert ablation.family_of("txn_n_total") == "txn"


def test_subsets_cover_every_nonempty_combination_smallest_first():
    subs = ablation.subsets()
    assert len(subs) == 2**3 - 1 and len(set(subs)) == len(subs)
    assert [len(s) for s in subs] == sorted(len(s) for s in subs)
    assert subs[0] == ("mkt",) and subs[-1] == ("mkt", "ord", "txn")


def test_subsets_of_two_families():
    assert ablation.subsets(("a", "b")) == [("a",), ("b",), ("a", "b")]


# --- drift: SQL and the fake-BigQuery path -------------------------------------------------

def test_stats_sql_asks_for_mean_std_and_null_rate_of_every_feature():
    sql = drift._stats_sql("p.d.t", ["f1", "f2"])
    for f in ("f1", "f2"):
        assert f"AVG(`{f}`)" in sql and f"STDDEV(`{f}`)" in sql
        assert f"COUNTIF(`{f}` IS NULL) / COUNT(*)" in sql
    assert sql.rstrip().endswith("FROM `p.d.t`")


class _Field:
    def __init__(self, name):
        self.name = name


class FakeBQ:
    """Answers the two calls drift makes: get_table(...).schema and query(sql).result()."""

    def __init__(self, stats_by_table, columns):
        self.stats, self.columns = stats_by_table, columns

    def get_table(self, table):
        cols = self.columns[table]
        return type("T", (), {"schema": [_Field(c) for c in cols]})()

    def query(self, sql):
        table = sql.split("FROM `")[1].rstrip("`").strip().rstrip("`")
        row = {}
        for feature, (mean, std, null) in self.stats[table].items():
            row[f"{feature}{drift.STAT_SEP}mean"] = mean
            row[f"{feature}{drift.STAT_SEP}std"] = std
            row[f"{feature}{drift.STAT_SEP}null"] = null
        return type("Q", (), {"result": lambda self_: iter([row])})()


def _fake(train, test):
    from src.config import load_config

    cfg = load_config()
    p, f = cfg.bigquery.project, cfg.bigquery.datasets.features
    tr, te = f"{p}.{f}.dataset_train", f"{p}.{f}.dataset_test"
    cols = {tr: list(train) + ["sample_id", "month", "target"], te: list(test) + ["sample_id"]}
    return FakeBQ({tr: train, te: test}, cols)


def test_shift_is_the_effect_size_in_training_standard_deviations():
    bq = _fake(
        train={"mkt_a": (0.0, 2.0, 0.0), "ord_b": (10.0, 1.0, 0.0), "txn_c": (5.0, 1.0, 0.05)},
        test={"mkt_a": (1.0, 2.0, 0.0), "ord_b": (10.0, 1.0, 0.0), "txn_c": (5.0, 1.0, 0.40)},
    )
    df = drift.compute_drift(bq).set_index("feature")
    assert df.loc["mkt_a", "shift"] == pytest.approx(0.5)       # |1-0| / 2
    assert df.loc["ord_b", "shift"] == pytest.approx(0.0)
    assert df.loc["txn_c", "null_delta"] == pytest.approx(0.35)  # broken even though means match
    assert df.loc["mkt_a", "family"] == "mkt"


def test_report_is_sorted_most_shifted_first_and_ignores_non_features():
    bq = _fake(train={"a": (0.0, 1.0, 0.0), "b": (0.0, 1.0, 0.0)},
               test={"a": (0.1, 1.0, 0.0), "b": (0.9, 1.0, 0.0)})
    df = drift.compute_drift(bq)
    assert list(df["feature"]) == ["b", "a"]
    assert not {"sample_id", "month", "target"} & set(df["feature"])


def test_a_constant_feature_does_not_divide_by_zero():
    bq = _fake(train={"flat": (3.0, 0.0, 0.0), "ok": (0.0, 1.0, 0.0)},
               test={"flat": (4.0, 0.0, 0.0), "ok": (0.0, 1.0, 0.0)})
    df = drift.compute_drift(bq).set_index("feature")
    assert pd.isna(df.loc["flat", "shift"])                      # undefined, not inf
    assert list(drift.compute_drift(bq)["feature"])[-1] == "flat"  # NaN sorted last


def test_summary_buckets_follow_the_documented_thresholds():
    df = pd.DataFrame({"shift": [0.05, 0.1, 0.29, 0.3, 0.49, 0.5, 2.0, np.nan],
                       "null_delta": [0, 0, 0, 0, 0.11, -0.2, 0, 0]})
    s = drift.summarise(df)
    assert (s["negligible_lt_0p1"], s["small_0p1_0p3"], s["moderate_0p3_0p5"],
            s["large_ge_0p5"]) == (1, 2, 2, 2)
    assert s["n_features"] == 8 and s["max_shift"] == 2.0
    assert s["n_null_delta_gt_0p1"] == 2


# --- drift_robustness ----------------------------------------------------------------------

def test_pruned_columns_drop_features_at_or_above_the_threshold():
    drift_df = pd.DataFrame({"feature": ["a", "b", "c"], "abs_shift": [0.05, 0.2, 0.9]})
    assert drift_robustness.pruned_columns(["a", "b", "c", "d"], drift_df, 0.2) == ["a", "d"]
    assert drift_robustness.pruned_columns(["a", "b"], drift_df, 5.0) == ["a", "b"]


def test_drift_ranking_reads_the_report_and_sorts_by_absolute_shift(tmp_path):
    path = tmp_path / "drift_report.csv"
    pd.DataFrame({"feature": ["a", "b", "c"], "shift": [0.1, -0.7, 0.3]}).to_csv(path, index=False)
    ranked = drift_robustness.drift_ranking(path)
    assert list(ranked["feature"]) == ["b", "c", "a"]


def test_trend_recovers_a_known_slope_and_correlation():
    out = pd.DataFrame({"gap_months": [1, 2, 3, 4, 5], "lift": [0.1, 0.2, 0.3, 0.4, 0.5]})
    slope, corr = drift_robustness.trend(out)
    assert slope == pytest.approx(0.1) and corr == pytest.approx(1.0)
    flipped = out.assign(lift=out["lift"][::-1].to_numpy())
    assert drift_robustness.trend(flipped)[0] == pytest.approx(-0.1)


# --- prediction_geometry -------------------------------------------------------------------

def test_regime_scale_is_a_multiplier_centred_on_one():
    rng = np.random.default_rng(0)
    raw = pd.Series(np.abs(rng.normal(size=2000)) + 1.0)
    for causal in (True, False):
        scale = prediction_geometry._regime_scale(raw, 200, causal=causal)
        assert scale.mean() == pytest.approx(1.0) and np.isfinite(scale).all()


def test_causal_regime_scale_does_not_look_at_later_samples():
    """The ratio between two early points must not change when the FUTURE changes.

    (The absolute level is divided by the whole series' mean, which is a constant and does not
    matter to a scale-invariant metric, so the check is on the ratio.)
    """
    rng = np.random.default_rng(1)
    raw = pd.Series(np.abs(rng.normal(size=1500)) + 1.0)
    changed = raw.copy()
    changed.iloc[1000:] *= 10
    a = prediction_geometry._regime_scale(raw, 100, causal=True)
    b = prediction_geometry._regime_scale(changed, 100, causal=True)
    assert a[300] / a[800] == pytest.approx(b[300] / b[800])
    # the two-sided version DOES see the future, which is why it is reported separately
    c = prediction_geometry._regime_scale(raw, 100, causal=False)
    d = prediction_geometry._regime_scale(changed, 100, causal=False)
    assert c[990] / c[300] != pytest.approx(d[990] / d[300])


def test_regime_scale_fills_the_warmup_window_instead_of_returning_nan():
    raw = pd.Series(np.ones(300))
    assert np.isfinite(prediction_geometry._regime_scale(raw, 100, causal=True)).all()
