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


# ------------------------------------------------------------------ the diagram

def test_the_architecture_diagram_is_current():
    """A regenerated diagram must equal the committed one.

    The whole reason it is generated rather than drawn is that a picture goes stale
    silently: it keeps describing the project as it was, and nobody re-reads a diagram
    they have already understood. If `make diagram` would change the file, the committed
    image is out of date and this fails instead.
    """
    from src.data import build_diagram

    committed = (ROOT / "docs" / "architecture.svg").read_text(encoding="utf-8")
    rebuilt = build_diagram.build().read_text(encoding="utf-8")
    assert rebuilt == committed, "run `make diagram` - docs/architecture.svg is stale"


def test_the_diagram_states_the_measured_feature_counts():
    """The counts on the picture come from results/, not from memory."""
    audit = load_json("feature_audit.json")
    svg = (ROOT / "docs" / "architecture.svg").read_text(encoding="utf-8")
    assert str(audit["n_features"]) in svg
    assert str(audit["n_effective"]) in svg


def test_the_diagram_needs_no_network():
    """It is shown in a README, in the app and on a projector. A remote font or image
    would make it fail in exactly the setting where failing is most expensive."""
    svg = (ROOT / "docs" / "architecture.svg").read_text(encoding="utf-8")
    for needle in ("http://", "https://", "<image", "@import"):
        assert needle not in svg.replace('xmlns="http://www.w3.org/2000/svg"', ""), (
            f"the diagram reaches outside itself: {needle}"
        )


# ------------------------------------------------------------------ the standing

def test_the_leaderboard_standing_is_read_from_the_snapshot():
    """Every surface quotes the capture, not a number typed once.

    The documented standing was 187 teams / median 0.138 / rank ~125. By the time it was
    checked the public leaderboard held 204 teams, median 0.137, rank 141 - the story
    unchanged, the figures all wrong. Prose cannot be re-measured; a file can.
    """
    lb = load_json("leaderboard.json")
    assert lb["n_teams"] > 0 and 0 < lb["median"] < 1
    assert readme_has(str(lb["n_teams"])), "the README does not quote the captured team count"
    assert readme_has(f"{lb['median']:.3f}"), "the README does not quote the captured median"
    assert readme_has(str(lb["our_rank"])), "the README does not quote the captured rank"
    assert readme_has(lb["captured"]), "the README does not date the standing"


def test_the_dashboard_states_the_standing_from_the_snapshot():
    """Absence of the stale number is not presence of the right one.

    Replacing the hardcoded paragraph on the investigation page silently failed while the
    deletion succeeded, so the page simply stopped saying where the model stands - and the
    "no stale figures" test below passed happily, because deleted text quotes nothing.
    A test that only forbids is only half a test.
    """
    page = (ROOT / "streamlit_app" / "pages" / "7_Investigation.py").read_text(
        encoding="utf-8")
    assert "leaderboard.json" in page, "the investigation page no longer reads the standing"
    assert "our_rank" in page, "the rank is not shown"


def test_no_surface_still_quotes_the_stale_standing():
    """187 teams and a 0.138 median were true in early September and are not now.

    Comments are stripped before checking. A note explaining why the figure moved has to
    be able to name the figure it replaced, and a test that forbids writing down its own
    reason pushes the reasoning out of the file - which is how the number got stuck there
    unexamined in the first place.
    """
    import ast

    # Everything a reader reads, which is wider than it first looks. Listing surfaces by
    # hand missed the figure twice: once in the notebooks, and once in the MODULE
    # DOCSTRINGS of shape_gain.py and shape_features.py, where the standing was quoted to
    # justify running the experiment at all. Those docstrings are documentation as much as
    # the README is, so the sweep now covers every tracked source and narrative file
    # rather than an enumerated list that has to be remembered.
    surfaces = [ROOT / "README.md",
                *sorted((ROOT / "src").rglob("*.py")),
                *sorted((ROOT / "streamlit_app").rglob("*.py")),
                *sorted((ROOT / "scripts").glob("*.py")),
                *sorted((ROOT / "notebooks").glob("*.ipynb")),
                *sorted((ROOT / "notebooks").glob("_build_*.py"))]
    for f in [p for p in surfaces if p.exists() and "__pycache__" not in p.parts]:
        text = f.read_text(encoding="utf-8")
        if f.suffix == ".py" and not f.name.startswith("_build_"):
            # Builders hold their prose as string literals, so unparsing would drop
            # nothing and gain nothing; other code has comments worth stripping.
            text = ast.unparse(ast.parse(text))
        assert "187 teams" not in text, (
            f"{f.relative_to(ROOT)} still quotes the stale team count"
        )


# ------------------------------------------------------------------ the live deployment

APP_URL = "https://ms-capital-market-forecasting-mfy6rngulq4fpaovzrhntf.streamlit.app/"


def test_the_readme_reaches_the_running_dashboard():
    """A portfolio repository whose demo cannot be reached from it is half a portfolio.

    The README described how to DEPLOY the dashboard for 750 lines without once saying
    where the deployed one is, so a visitor could read the whole thing and never open it.
    """
    assert APP_URL in README, "the README no longer links the live dashboard"
    assert README.count(APP_URL) >= 2, (
        "the link should survive both a skim (badge, intro) and a read (deployment section)"
    )


def test_the_static_page_links_the_interactive_one():
    """The two published surfaces should know about each other.

    The static page is the summary; anyone wanting to click through the folds or the SHAP
    values needs the app, and the only place to learn it exists is this link.
    """
    site = ROOT / "site" / "index.html"
    if not site.exists():
        pytest.skip("site/ has not been built")
    assert APP_URL in site.read_text(encoding="utf-8")


def test_the_app_url_is_written_once_per_surface_and_not_guessed():
    """One constant per file, so a moved deployment is a small edit rather than a hunt."""
    builder = (ROOT / "src" / "data" / "build_site.py").read_text(encoding="utf-8")
    assert f'APP = "{APP_URL}"' in builder, (
        "the static site should hold the URL in a named constant, not inline in markup"
    )


# ------------------------------------------------- figures the notebooks measured

def test_the_spread_sentinel_figures_match_the_notebook_that_measured_them():
    """The README quoted -0.0064; the measurement is -0.005648.

    The wrong figure is the metric-alignment result from a completely unrelated
    experiment, which had been copied into this row - and it survived because the
    documented-numbers tests only covered values exported to `results/`. This one lives
    in notebook 01's stored output, which is the only record of it, so that output is
    what the README is checked against.

    It matters more than most: the sentinel finding is one of the four the project leads
    with, and a headline discovery quoting a number from a different experiment is the
    kind of error a reader checks first.
    """
    import json

    nb = json.loads((ROOT / "notebooks" / "01_data_discovery.ipynb").read_text(
        encoding="utf-8"))
    printed = "".join(
        "".join(out.get("text", []))
        for cell in nb["cells"]
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
    )

    match = re.search(r"sign flip:\s*(-?[\d.]+)\s*->\s*(\+?[\d.]+)", printed)
    assert match, "notebook 01 no longer records the sign-flip measurement"
    naive, cleaned = match.group(1), match.group(2).lstrip("+")

    # Checked in the SENTENCES that make the claim, not anywhere in the file. A first
    # version searched the whole README, so corrupting one of the two places still passed:
    # the other copy satisfied the search. The figure appears twice and both must be right.
    naive_forms = (naive, naive.replace("-", "\u2212"))
    # Only the two sentences that carry the figures. "empty-level sentinel" alone also
    # matches a data-contract row that quotes nothing, and including it made this fail
    # for the wrong reason.
    claims = [
        line for line in README.splitlines()
        if "no genuinely crossed books" in line
        or ("empty-level sentinel" in line and "relative spread" in line)
    ]
    assert len(claims) == 2, f"expected both sentinel claims, found {len(claims)}"

    for line in claims:
        assert any(form in line for form in naive_forms), (
            f"this sentence does not quote the measured {naive}: {line[:90]}"
        )
        assert cleaned in line, (
            f"this sentence does not quote the measured {cleaned}: {line[:90]}"
        )
    assert float(naive) < 0 < float(cleaned), "the sign flip itself has gone"
