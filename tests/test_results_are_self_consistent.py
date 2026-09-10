"""Recompute the exported numbers from their own inputs.

There are two different questions about the figures in this project and only one of them
had a test. `test_documented_numbers.py` checks that the README says what `results/` says
- that the documents and the data agree. It cannot tell you whether the data is right. If
a pipeline run had produced a wrong summary, every surface would have quoted it faithfully
and every test would have passed.

So this file goes the other way: it takes the most primitive thing in each result file and
rebuilds the derived quantity from it. The per-fold cosine scores rebuild the mean and the
standard deviation. The block scores rebuild the percentile and the de-biasing. The
correlated-pair list rebuilds the effective feature count. The trade statistics rebuild the
Sharpe ratio and the total return. Where a value cannot be rebuilt from what was exported -
the hold-out cosine itself, for instance, which would need the stored predictions - it is
not claimed here, and the metric implementation is covered against sklearn in
tests/test_metrics.py instead.

Written after a manual pass found two real overstatements: a residual described as "5σ
above period noise" that is 1.4σ against the spread of a single period (the 5 came from the
standard error of an average, which answers a different question), and an ensemble gain
quoted as "+0.0022 in 5 folds of 5" when the per-fold margin ranges +0.0004 to +0.0052.
Both are now stated precisely, and both are pinned below.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


@pytest.fixture(scope="module")
def folds() -> list[dict]:
    """The per-fold record every walk-forward summary statistic is derived from."""
    return load_json("walkforward_summary.json")["ensemble"]["per_fold"]


def fold_scores(folds: list[dict], model: str) -> list[float]:
    if model == "ensemble":
        return [f["ensemble_score"] for f in folds]
    return [f["single_scores"][model] for f in folds]


# ------------------------------------------------------------------ walk-forward

@pytest.mark.parametrize("model",
                         ["ensemble", "lightgbm", "xgboost", "ridge", "mean", "zero"])
def test_summary_statistics_rebuild_from_the_fold_scores(folds, model):
    """mean, std, min and max are the fold scores and nothing else.

    The headline +0.14088 and the 0.0041 noise floor both come from this table, and half
    the argument in the README is measured against that noise floor.
    """
    row = load_csv("walkforward_summary.csv").query("model == @model").iloc[0]
    scores = fold_scores(folds, model)
    assert float(row.cosine_mean) == pytest.approx(np.mean(scores), abs=1e-9)
    assert float(row.cosine_std) == pytest.approx(np.std(scores, ddof=1), abs=1e-9)
    assert float(row.cosine_min) == pytest.approx(np.min(scores), abs=1e-12)
    assert float(row.cosine_max) == pytest.approx(np.max(scores), abs=1e-12)


def test_the_ensemble_gain_is_stated_at_the_size_it_actually_is(folds):
    """Positive in every fold, but not by the same amount in every fold.

    "+0.0022 in 5 folds of 5" reads as +0.0022 each time. The margin is +0.0004 in the
    narrowest and +0.0052 in the widest, and on a problem whose fold-to-fold std is 0.0041
    that spread is the point rather than a detail.
    """
    margins = np.array(fold_scores(folds, "ensemble")) - np.array(
        fold_scores(folds, "lightgbm"))
    assert (margins > 0).all(), "the ensemble no longer wins every fold"
    assert margins.mean() == pytest.approx(0.0022, abs=5e-5)
    assert margins.min() == pytest.approx(0.0004, abs=5e-5)
    assert margins.max() == pytest.approx(0.0052, abs=5e-5)


def test_beating_the_best_single_model_is_a_different_measure(folds):
    """Two comparisons live in this file and they give different numbers.

    Against LightGBM the mean margin is +0.0022; against whichever model happened to win
    each fold it is +0.0016, median +0.0017. The README quotes both, in the right places -
    this pins that they stay distinguishable.
    """
    vs_best = np.array([f["ensemble_score"] - f["best_single_score"] for f in folds])
    assert (vs_best > 0).all()
    assert np.median(vs_best) == pytest.approx(0.0017, abs=5e-5)
    assert vs_best.mean() < (np.array(fold_scores(folds, "ensemble"))
                             - np.array(fold_scores(folds, "lightgbm"))).mean()


# ------------------------------------------------------------------ period difficulty

def test_the_percentile_and_the_typical_block_rebuild_from_the_blocks():
    per, meta = load_csv("period_difficulty.csv"), load_json("period_difficulty_meta.json")
    share_below = (per.cosine < meta["holdout_cosine"]).mean()
    assert meta["holdout_percentile"] == pytest.approx(share_below, abs=1e-9)
    assert meta["typical_cosine"] == pytest.approx(per.cosine.median(), abs=1e-6)


def test_the_debiased_score_is_the_reported_one_scaled_by_period_difficulty():
    """0.15171 x (typical / hold-out) = 0.14084.

    The claim that makes this convincing is the coincidence: the walk-forward mean is
    +0.14088, computed from different folds by a different route, and the two land within
    0.00004 of each other. If either drifts, the coincidence is the first casualty.
    """
    meta = load_json("period_difficulty_meta.json")
    rebuilt = meta["reported"] * meta["typical_cosine"] / meta["holdout_cosine"]
    assert meta["debiased"] == pytest.approx(rebuilt, abs=1e-6)

    cv = float(load_csv("walkforward_summary.csv")
               .query("model == 'ensemble'").cosine_mean.iloc[0])
    assert abs(meta["debiased"] - cv) < 1e-4, "the two routes no longer agree"


def test_the_unexplained_residual_is_measured_against_a_single_period():
    """1.4 standard deviations, not 5.

    The gap left after de-biasing was described as "5σ above the test set's own period
    noise". That 5 comes from dividing by the standard error of the mean block score
    (0.0026), which answers "is the AVERAGE difficulty estimated precisely?" - not "could
    one unlucky period explain this?". Against the spread a single period actually shows
    (0.0091) the residual is 1.4σ, which does not rule luck out.
    """
    per, meta = load_csv("period_difficulty.csv"), load_json("period_difficulty_meta.json")
    residual = (meta["reported"] - meta["actual"]) * (1 - meta["share_of_gap"])
    sigma = float(per.cosine.std())

    assert residual / sigma == pytest.approx(1.4, abs=0.15)
    assert residual / (sigma / np.sqrt(len(per))) > 4, (
        "the standard-error reading has moved too - check both numbers before quoting"
    )


# ------------------------------------------------------------------ features

def test_the_effective_feature_count_rebuilds_from_the_correlated_pairs():
    """292 counted, 264 distinct - by union-find over the pairs above 0.999.

    Merging pairs one at a time would over-count: three mutually correlated columns are
    two pairs but collapse to one feature, not two.
    """
    audit, pairs = load_json("feature_audit.json"), load_csv("feature_redundancy.csv")
    edges = pairs[pairs.abs_corr > audit["threshold"]]
    cols = [c for c in pairs.columns if c != "abs_corr"][:2]

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, row in edges.iterrows():
        a, b = find(row[cols[0]]), find(row[cols[1]])
        if a != b:
            parent[a] = b

    collapsed = len(parent) - len({find(x) for x in parent})
    assert audit["n_effective"] == audit["n_features"] - collapsed


def test_the_monthly_table_covers_the_whole_training_set():
    """71 months and 1,257,637 rows - the count is the check that nothing was sampled.

    These statistics exist to be computed on everything; a per-month standard deviation
    from a subset would be noise presented as a regime.
    """
    by_month = load_csv("target_by_month.csv")
    assert len(by_month) == 71
    assert sorted(by_month.month) == list(range(71))
    assert int(by_month["count"].sum()) == 1_257_637


# ------------------------------------------------------------------ backtest

def test_the_backtest_figures_rebuild_from_the_trade_statistics():
    bt = load_json("holdout_metrics.json")["backtest"]
    assert bt["sharpe"] == pytest.approx(bt["mean_return"] / bt["volatility"], abs=1e-4)
    assert bt["total_return"] == pytest.approx(bt["n_trades"] * bt["mean_return"], rel=1e-6)


def test_the_equity_curve_ends_where_the_reported_return_says():
    """The curve is downsampled on export; the endpoint has to survive that."""
    bt = load_json("holdout_metrics.json")["backtest"]
    equity = load_csv("backtest_equity.csv")
    assert float(equity.iloc[-1, -1]) == pytest.approx(bt["total_return"], rel=1e-4)


# ------------------------------------------------------------------ intervals

@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_metric_alignment_intervals_are_the_stated_gain_plus_minus_two_standard_errors(alpha):
    rows = {r["alpha"]: r for r in load_json("metric_alignment_meta.json")["summary"]}
    row = rows[alpha]
    assert row["ci_low"] == pytest.approx(row["gain"] - 1.96 * row["se"], abs=1e-6)
    assert row["ci_high"] == pytest.approx(row["gain"] + 1.96 * row["se"], abs=1e-6)
    assert row["ci_high"] < 0, "metric alignment no longer hurts - the claim has flipped"


def test_the_shape_gain_interval_still_spans_zero():
    """The finding is that sequence shape buys nothing measurable. If the interval ever
    excludes zero, that conclusion has to be rewritten rather than restated."""
    meta = load_json("shape_gain_meta.json")
    assert meta["ci_low"] < 0 < meta["ci_high"]
    assert meta["paired_gain"] == pytest.approx(0.0006, abs=1e-4)


# ------------------------------------------------------------------ the identity

def test_the_cosine_decomposition_reproduces_the_pooled_score_exactly():
    """cos(y,p) = Σ cos_g · w_g, to floating-point exactness.

    This is the one number in the project that is a mathematical identity rather than a
    measurement, so it is the strongest single check that the decomposition code is doing
    what the README says it does.
    """
    meta = load_json("cosine_decomposition_meta.json")
    assert meta["rebuilt"] == pytest.approx(meta["pooled"], abs=1e-12)
    assert meta["abs_error"] < 1e-12
    assert meta["magnitude_weighted"] != meta["count_weighted"], (
        "the two weightings have become identical - the finding depended on them differing"
    )
