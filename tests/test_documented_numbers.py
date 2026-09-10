"""The numbers written in the README have to be the numbers in `results/`.

A README is the one artefact nothing recomputes. Every figure in it was measured once and
typed once, and from then on it drifts silently: a re-run moves a score, a rounding
changes, an experiment is repeated with a different seed, and the prose keeps asserting
the old value with complete confidence. Nothing fails. That is worse than a broken test,
because a stale number in a portfolio README reads as a claim the author never checked.

This has already happened on this project in the small: a forecast table headed "Three
forecasts" over five rows, a caption promising a 2.69x monthly swing above a chart drawn
from a single month, and a paragraph asserting the LightGBM scored 0.129 when it scored
0.128. Each was found by reading, not by running.

So each headline figure is parsed back out of the README text and checked against the
exported result it came from. The point is not the arithmetic - it is that the document
and the pipeline cannot disagree without something going red.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
RESULTS = ROOT / "results"


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def readme_has(value: str) -> bool:
    """Is this literal figure written anywhere in the README?

    A substring check rather than a parse of one specific sentence: the figures appear in
    tables, in prose and inside bold markers, and pinning the sentence would make the test
    fail on rewording instead of on drift.
    """
    return value in README


# ------------------------------------------------------------------ the headline scores

def test_walk_forward_mean_matches_the_summary():
    wf = load_csv("walkforward_summary.csv")
    mean = float(wf.loc[wf.model == "ensemble", "cosine_mean"].iloc[0])
    assert readme_has(f"{mean:+.5f}"), f"README does not quote the CV mean {mean:+.5f}"


def test_fold_to_fold_std_matches_the_summary():
    """The noise floor. Half the argument in this README is measured against it."""
    wf = load_csv("walkforward_summary.csv")
    std = float(wf.loc[wf.model == "ensemble", "cosine_std"].iloc[0])
    assert readme_has(f"{std:.4f}"), f"README does not quote the fold-to-fold std {std:.4f}"


def test_holdout_cosine_matches_the_metrics_file():
    cosine = load_json("holdout_metrics.json")["scores"]["cosine"]
    assert readme_has(f"{cosine:.5f}"), f"README does not quote the hold-out {cosine:.5f}"


def test_holdout_percentile_and_debiased_score_match():
    meta = load_json("period_difficulty_meta.json")
    pct = round(meta["holdout_percentile"] * 100)
    assert readme_has(f"{pct}"), f"README does not quote the {pct}th percentile"
    assert readme_has(f"{meta['debiased']:+.5f}"), "the de-biased score is not quoted"


def test_period_difficulty_range_matches():
    """"difficulty swings from 0.117 to 0.148" - both ends, from the same table."""
    per = load_csv("period_difficulty.csv")
    for value in (per.cosine.min(), per.cosine.max()):
        assert readme_has(f"{value:.3f}"), f"README does not quote {value:.3f}"


# ------------------------------------------------------------------ the data findings

def test_the_feature_counts_match_the_audit():
    """"292 columns, 264 distinct" - the redundancy result, quoted in three places."""
    audit = load_json("feature_audit.json")
    assert readme_has(str(audit["n_features"]))
    assert readme_has(str(audit["n_effective"]))


def test_the_monthly_volatility_swing_matches_the_full_table():
    """The caption that was false for months: the chart is drawn from the export now, so
    the ratio in the prose and the ratio in the data are the same computation."""
    by_month = load_csv("target_by_month.csv")
    swing = by_month["std"].max() / by_month["std"].min()
    assert readme_has(f"{swing:.2f}"), f"README does not quote the {swing:.2f}x swing"


def test_the_adversarial_comparison_matches():
    """The test set is closer to the last training block than blocks ten months apart.

    Both AUCs are quoted; if either moves, the sentence built on their ORDER is no longer
    supported by the numbers beside it.
    """
    adv = load_csv("adversarial_auc.csv")
    test_auc = float(adv.loc[adv.comparison.str.contains("last block"), "auc"].iloc[0])
    within = float(adv.loc[adv.comparison == "months 10-19", "auc"].iloc[0])
    assert test_auc < within, "the finding itself has reversed, not just the prose"
    assert readme_has(f"{test_auc:.4f}") or readme_has(f"{test_auc:.3f}")
    assert readme_has(f"{within:.4f}") or readme_has(f"{within:.3f}")


# ------------------------------------------------------------------ the leaderboard pair

@pytest.mark.parametrize("score", ["0.128", "0.129"])
def test_both_submission_scores_are_stated(score):
    """Two models were graded. The README has been wrong about which scored which, so
    both numbers have to be present and attributed."""
    assert readme_has(score)


def test_the_two_submissions_are_not_confused_for_one_model():
    """A single row quoting one score would flatten two models into one.

    The results table names them separately - single LightGBM at 0.128, ensemble at
    0.129 - and that distinction is what makes the +0.001 gain readable at all.
    """
    table = re.search(r"\| Leaderboard, first submission.*?\n\n", README, re.S)
    assert table, "the results table no longer separates the two submissions"
    block = table.group(0)
    assert "0.12800" in block and "0.12900" in block
