"""What the dashboard reads has to actually be in `results/`, and be intact when it is.

`export_results.run()` skips a missing source file with a warning and carries on. That is
the right behaviour for a partial pipeline, and it is also how the bundle can end up half
complete without anything failing: every loader returns None, every page renders "not
available yet", and a deployment looks broken rather than incomplete.

Two separate things are checked here, because they fail independently:

  * the CONTENTS of the committed bundle - every file any page asks for is present, and
    the ones that were transformed on the way in still say what the originals said;
  * the EXPORT and SITE code paths, on a synthetic source directory, since the real ones
    need a 1.3 GB feature table.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.data.export_results import EQUITY_POINTS, VERBATIM, _downsample_equity

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
APP = ROOT / "streamlit_app"


# ------------------------------------------------------- the committed bundle is complete

LOADERS = {"find", "load_csv", "load_json"}


def requested_files() -> list[tuple[str, tuple[str, ...]]]:
    """Every loader call in the dashboard, as (source file, names it will accept).

    Read from the AST rather than by regex, because the names in one call are
    ALTERNATIVES, not a list of requirements: `find("walkforward_summary.json",
    "smoke_summary.json")` prefers the full pipeline's output and falls back to the demo's.
    Requiring both would fail on a file that is not supposed to be committed; requiring
    neither is the bug this is here to catch. What has to hold is that at least one of
    them travels.

    Deriving it from the code means a page added later is covered without anyone
    remembering to update a list here.
    """
    out: list[tuple[str, tuple[str, ...]]] = []
    for f in APP.rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name not in LOADERS or not node.args:
                continue
            args = [a.value for a in node.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if len(args) == len(node.args) and args:
                out.append((f.name, tuple(args)))
    return out


def test_every_file_the_dashboard_asks_for_is_in_the_bundle():
    """The failure this catches is silent: a page renders 'not available yet' instead.

    Explainability and Backtesting each shipped in that state once. Nothing errored -
    the loaders return None by design - so only looking at the deployed page revealed it.
    """
    unsatisfied = [
        (src, names) for src, names in requested_files()
        if not any((RESULTS / n).exists() for n in names)
    ]
    assert not unsatisfied, f"requested by a page but not exported: {unsatisfied}"


def test_the_export_list_covers_what_the_pages_request():
    """A file can exist in results/ today and stop being refreshed tomorrow.

    `VERBATIM` is what a re-export copies. Anything a page reads that is not in that list,
    nor produced by one of the special cases below, would quietly go stale while still
    rendering - the worst version, because the page keeps looking authoritative.
    """
    # Not in VERBATIM because the export builds them rather than copying them - except
    # leaderboard.json, which does not come from the pipeline at all. It is a capture of
    # the public Kaggle standing, refreshed by hand, and it lives here so that every
    # surface quoting a rank reads a dated file instead of prose written once.
    special = {"backtest_equity.csv", "feature_sample.parquet", "target_by_month.csv",
               "model.txt", "model_meta.json", "shap_global.csv",
               "shap_local_examples.csv", "leaderboard.json"}
    refreshed = set(VERBATIM) | special
    uncovered = [
        (src, names) for src, names in requested_files()
        if not any(n in refreshed for n in names)
    ]
    assert not uncovered, f"read by a page but never re-exported: {uncovered}"


@pytest.mark.parametrize("name", sorted(
    n for n in VERBATIM if n.endswith(".json")))
def test_every_exported_json_parses(name):
    """A truncated copy is still a file. json.loads is the cheapest way to notice."""
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} has not been exported")
    assert json.loads(path.read_text(encoding="utf-8")) != {}


@pytest.mark.parametrize("name", sorted(
    n for n in VERBATIM if n.endswith(".csv")))
def test_every_exported_csv_has_rows_and_no_empty_columns(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} has not been exported")
    df = pd.read_csv(path)
    assert len(df) > 0, f"{name} is empty"
    all_null = [c for c in df.columns if df[c].isna().all()]
    assert not all_null, f"{name} has entirely empty column(s): {all_null}"


def test_the_bundle_stays_small_enough_to_carry_in_git():
    """The export raises above 24 MB; this is the standing check on the committed state."""
    total = sum(f.stat().st_size for f in RESULTS.glob("*") if f.is_file())
    assert total < 24 * 1024 * 1024, f"results/ is {total / 1e6:.1f} MB"


# ------------------------------------------------------------------ equity downsampling

def test_downsampling_keeps_both_endpoints():
    """The chart shows a cumulative path. Losing the last point changes the headline
    return; losing the first changes the starting level everything is read against."""
    src = pd.DataFrame({"equity": np.linspace(1.0, 1.17, 20_954)})
    out = _thin(src)
    assert out["equity"].iloc[0] == pytest.approx(src["equity"].iloc[0])
    assert out["equity"].iloc[-1] == pytest.approx(src["equity"].iloc[-1])


def test_downsampling_preserves_the_drawdown():
    """Even spacing rather than a rolling mean, because smoothing would flatten the very
    feature the chart exists to show. A dip must survive the thinning."""
    values = np.linspace(1.0, 1.2, 20_000)
    values[8_000:9_000] -= 0.15                       # a drawdown 1,000 points wide
    out = _thin(pd.DataFrame({"equity": values}))
    assert out["equity"].min() < 1.0, "the drawdown was smoothed away"


def test_downsampling_leaves_a_short_curve_alone():
    src = pd.DataFrame({"equity": np.linspace(1.0, 1.1, 50)})
    assert len(_thin(src)) == 50


def test_downsampled_curve_fits_the_budget():
    out = _thin(pd.DataFrame({"equity": np.linspace(1.0, 1.2, 100_000)}))
    assert len(out) <= EQUITY_POINTS


def _thin(df: pd.DataFrame) -> pd.DataFrame:
    """Run the real downsampler through the filesystem, as the export does."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.csv", Path(d) / "out.csv"
        df.to_csv(src, index=False)
        _downsample_equity(src, dst)
        return pd.read_csv(dst)


# ------------------------------------------------------------------ the static site

def test_the_site_builds_from_the_committed_bundle(tmp_path, monkeypatch):
    """`make site` must work from a clone, the same way the dashboard does.

    `raising=True` on the patch is deliberate. The first version of this test patched a
    name the module does not have, so the patch was a no-op and build() quietly wrote into
    the repository's own site/ - a test with a side effect on the working tree, which is
    the kind of thing that stays hidden until it overwrites something.
    """
    from src.data import build_site

    monkeypatch.setattr(build_site, "SITE", tmp_path, raising=True)
    path = build_site.build()
    assert Path(path).parent == tmp_path, "build() ignored the patched output directory"

    html = Path(path).read_text(encoding="utf-8")
    assert len(html) > 5_000, "the page came out suspiciously small"
    assert "<svg" in html, "no chart was drawn"


def test_the_site_build_is_deterministic(tmp_path, monkeypatch):
    """Same inputs, same bytes - otherwise every rebuild shows up as a diff and the real
    changes get lost among them."""
    from src.data import build_site

    monkeypatch.setattr(build_site, "SITE", tmp_path, raising=True)
    first = Path(build_site.build()).read_bytes()
    second = Path(build_site.build()).read_bytes()
    assert first == second


def test_the_site_loads_nothing_from_the_network():
    """The page is meant to be one self-contained file - it is served from anywhere,
    including offline. A CDN reference would make it depend on a host that can go away."""
    site = ROOT / "site" / "index.html"
    if not site.exists():
        pytest.skip("site/ has not been built")
    html = site.read_text(encoding="utf-8")
    for needle in ("http://", "https://cdn", "<script src="):
        assert needle not in html.replace('href="https://github.com', ""), (
            f"the page reaches out to the network: {needle}"
        )


def test_the_site_quotes_the_same_headline_numbers_as_the_bundle():
    """A hand-typed number in a template drifts from the pipeline that produced it.

    The point of the page is that its numbers are the measured ones; a copy that has gone
    stale is worse than no page, because it reads as authoritative.
    """
    site = ROOT / "site" / "index.html"
    if not site.exists():
        pytest.skip("site/ has not been built")
    html = site.read_text(encoding="utf-8")
    holdout = json.loads((RESULTS / "holdout_metrics.json").read_text(encoding="utf-8"))
    cosine = holdout["scores"]["cosine"]
    assert f"{cosine:.4f}"[:6] in html, (
        "the hold-out cosine on the page does not match holdout_metrics.json"
    )
