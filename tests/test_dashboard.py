"""The dashboard has to work away from the machine that produced its numbers.

Everything it displays normally comes from `C:/mscapital_data`, which exists on one
laptop. Publishing it means the numbers travel in `results/`, and these tests pin the
fallback that makes that work - because the failure mode is silent: away from the pipeline
the loaders return None, every page renders "not available yet", and the deployed
dashboard looks broken rather than erroring.

They run with MSCAPITAL_DATA_ROOT pointed at nothing, which is exactly the deployed
condition.
"""
import importlib
import os

import pandas as pd
import pytest


@pytest.fixture()
def lib(monkeypatch):
    """streamlit_app.lib with no local pipeline output - the published situation."""
    monkeypatch.setenv("MSCAPITAL_DATA_ROOT", os.path.join(os.sep, "nonexistent-data-root"))
    monkeypatch.setenv("MSCAPITAL_API_URL", "")
    import streamlit_app.lib as m

    return importlib.reload(m)


# ------------------------------------------------------------------ the fallback

def test_results_load_without_the_pipeline(lib):
    """The whole point: the deployed dashboard has numbers."""
    assert lib.load_results_table() is not None
    assert lib.load_json("holdout_metrics.json") is not None


def test_the_feature_sample_stands_in_for_the_full_table(lib):
    df = lib.load_features(n_rows=500)
    assert df is not None and len(df) == 500


def test_the_sample_carries_every_model_feature(lib):
    """A narrow sample would break the Predictions page.

    Predictor rejects an incomplete row rather than quietly imputing - correct behaviour,
    and it means the exported sample has to be full width, not just the columns the
    overview charts happen to plot.
    """
    model = lib.load_local_model()
    df = lib.load_features(n_rows=10)
    assert model is not None, "no shipped artefact in results/"
    assert not set(model.bundle.features) - set(df.columns)


def test_prediction_works_with_no_api(lib):
    """End to end on the deployed path: sample row in, prediction out."""
    model = lib.load_local_model()
    df = lib.load_features(n_rows=5)
    row = df.iloc[0]
    features = {c: float(row[c]) for c in model.bundle.features if pd.notna(row[c])}
    features |= {c: 0.0 for c in model.bundle.features if c not in features}
    out = model.predict([features])
    assert len(out) == 1 and pd.notna(out[0])


def test_local_output_wins_over_the_bundled_snapshot(lib, tmp_path, monkeypatch):
    """Someone running beside the real pipeline must see their own fresh numbers.

    A snapshot committed weeks ago silently overriding a fresh run would be the worst
    kind of wrong: plausible, stale, and invisible.
    """
    features = tmp_path / "features"
    features.mkdir()
    (features / "holdout_metrics.json").write_text('{"marker": "local"}', encoding="utf-8")
    monkeypatch.setattr(lib, "FEATURES_DIR", features)
    lib.load_json.clear()
    assert lib.load_json("holdout_metrics.json") == {"marker": "local"}


def test_missing_files_return_none_rather_than_raising(lib):
    """Every loader degrades to an explicit 'not available' state, never a stack trace."""
    assert lib.load_csv("does_not_exist.csv") is None
    assert lib.load_json("does_not_exist.json") is None


# ------------------------------------------------------------------ no API, no waiting

def test_api_calls_short_circuit_when_unconfigured(lib):
    """Published, there is no API. Timing out against localhost on every page load would
    add five seconds to each render for a result that is always None."""
    assert lib.api_get("/health") is None
    assert lib.api_post("/predict", {}) == (0, None)


# ------------------------------------------------------------------ the histogram fix

def test_histogram_index_is_numeric_and_ordered(lib):
    """st.bar_chart renders an IntervalIndex as raw dicts sorted LEXICOGRAPHICALLY,
    which puts "10.8" between "1.2" and "2.4"."""
    h = lib.histogram(pd.Series(range(1000)), bins=10, label="x")
    idx = list(h.index)
    assert all(isinstance(v, float) for v in idx)
    assert idx == sorted(idx)


def test_histogram_midpoints_are_rounded(lib):
    """Raw midpoints carry float noise (10.241499999999999) that Streamlit prints in full."""
    h = lib.histogram(pd.Series(range(1000)), bins=7, label="x")
    assert all(len(repr(v)) <= 8 for v in h.index)


def test_histogram_counts_every_row(lib):
    h = lib.histogram(pd.Series(range(500)), bins=13, label="x")
    assert h["samples"].sum() == 500
