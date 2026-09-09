"""Shared data access for the Streamlit pages.

TWO SOURCES, IN ORDER

  1. `MSCAPITAL_DATA_ROOT` - the full local pipeline output, when running beside it
  2. `results/` in the repository - the exported summaries, when running anywhere else

The second is what makes the dashboard publishable. It carries only derived aggregates -
fold scores, backtest curves, SHAP importances, the investigation results - plus a
~5k-row sample of the feature table, drawn evenly across all 71 months, for the overview
charts. No competition data travels, and
every number shown is one already published in the notebooks.

Design: every loader returns None instead of crashing on missing data, so the dashboard
stays usable at any stage of the pipeline and tells the user what is missing - an explicit
state rather than a blank screen.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from src.inference.predictor import Predictor

DATA_ROOT = Path(os.environ.get("MSCAPITAL_DATA_ROOT", "C:/mscapital_data"))
API_URL = os.environ.get("MSCAPITAL_API_URL", "")

FEATURES_DIR = DATA_ROOT / "features"
MODELS_DIR = DATA_ROOT / "models"
BUNDLED = Path(__file__).resolve().parents[1] / "results"


def find(*names: str) -> Path | None:
    """First existing match, local pipeline output before the bundled export.

    Order matters: someone running beside the real pipeline should see their own fresh
    numbers, not a snapshot committed weeks ago.
    """
    for d in (FEATURES_DIR, MODELS_DIR, BUNDLED):
        for name in names:
            p = d / name
            if p.exists():
                return p
    return None

DISCLAIMER = (
    "This dashboard is for **research and model evaluation**. "
    "It is not investment advice."
)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.caption(DISCLAIMER)


def missing(what: str, how: str) -> None:
    st.info(f"**{what}** is not available yet.\n\nTo produce it, run: `{how}`")


def histogram(values: pd.Series, bins: int = 50, label: str = "value") -> pd.DataFrame:
    """A binned distribution Streamlit can actually plot.

    `value_counts(bins=)` returns an IntervalIndex, and st.bar_chart renders those as raw
    {"left": ..., "right": ...} dicts sorted LEXICOGRAPHICALLY - so "10.8" lands between
    "1.2" and "2.4". Reducing each interval to its midpoint gives a numeric axis in the
    right order.

    Midpoints are rounded to four significant figures: the raw value carries float noise
    (10.241499999999999) that Streamlit prints in full and that means nothing to a reader.
    """
    binned = values.value_counts(bins=bins).sort_index()
    mids = [float(f"{iv.mid:.4g}") for iv in binned.index]
    return pd.DataFrame(
        {"samples": binned.to_numpy()},
        index=pd.Index(mids, name=label),
    )


@st.cache_data(show_spinner=False)
def load_summary() -> dict | None:
    """Per-fold walk-forward scores for every model."""
    p = find("walkforward_summary.json", "smoke_summary.json")
    return json.loads(p.read_text(encoding="utf-8")) if p else None


@st.cache_data(show_spinner=False)
def load_results_table() -> pd.DataFrame | None:
    """The model comparison table: mean, std, min and max cosine per model."""
    p = find("walkforward_summary.csv", "smoke_summary.csv")
    return pd.read_csv(p) if p else None


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    """Any exported result table, by filename."""
    p = find(name)
    return pd.read_csv(p) if p else None


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict | None:
    """Any exported result document, by filename."""
    p = find(name)
    return json.loads(p.read_text(encoding="utf-8")) if p else None


@st.cache_data(show_spinner=False)
def load_features(n_rows: int = 50_000, columns: list[str] | None = None) -> pd.DataFrame | None:
    """A slice of the feature set - spread over the months when the caller names columns.

    The full table is ~1.4 GB and is never fully loaded. Away from the pipeline this falls
    back to `results/feature_sample.parquet`: 4,970 rows at FULL width, drawn evenly from
    all 71 months.

    WHY THE SPREAD MATTERS

    Reading the first n rows is not a sample of this table. `sample_id` is chronological,
    so the first 20,000 rows are months 0-1 - and every distribution on the overview page
    was drawn from them while being captioned as though it described the training set.
    The exported bundle had the same defect in worse form: the first 5,000 rows are all
    month 0, so the monthly-volatility chart had a single point.

    The spread is applied only when `columns` is given, which is the charts' path: a
    ten-column projection of the whole table is about 50 MB, cheap to read in full and
    then thin. The full-width path stays streamed, because 295 columns is 1.4 GB - and
    its one caller is the Predictions page, where the reader picks a sample_id by hand
    and representativeness is not what the rows are for.
    """
    p = find("dataset_train.parquet", "feature_sample.parquet")
    if p is None:
        return None
    import numpy as np
    import pyarrow.parquet as pq

    if p.name == "feature_sample.parquet":
        df = pq.read_table(p).to_pandas()
        keep = [c for c in (columns or df.columns) if c in df.columns]
        return df[keep].head(n_rows)

    if columns is not None:
        df = pq.read_table(p, columns=columns).to_pandas()
        if "month" in df.columns and len(df) > n_rows:
            per = max(1, n_rows // df["month"].nunique())
            months = df["month"].to_numpy()
            idx = np.concatenate([np.flatnonzero(months == m)[:per]
                                  for m in np.sort(df["month"].unique())])
            return df.iloc[idx[:n_rows]].reset_index(drop=True)
        return df.head(n_rows)

    pf = pq.ParquetFile(p)
    batches = pf.iter_batches(batch_size=min(n_rows, 65_536), columns=columns)
    frames, total = [], 0
    for b in batches:
        df = b.to_pandas()
        frames.append(df)
        total += len(df)
        if total >= n_rows:
            break
    return pd.concat(frames, ignore_index=True).head(n_rows) if frames else None


@st.cache_data(show_spinner=False)
def feature_columns() -> list[str] | None:
    """Column names of whichever feature table is available."""
    p = find("dataset_train.parquet", "feature_sample.parquet")
    if p is None:
        return None
    import pyarrow.parquet as pq

    return pq.ParquetFile(p).schema_arrow.names


@st.cache_resource(show_spinner=False)
def load_local_model():
    """The shipped model, loaded in-process - no API required.

    A published dashboard has no FastAPI beside it, so a Predictions page that can only
    report "cannot reach the API" would be dead on the one deployment that matters. The
    artefact travels in `results/`, and `Predictor` is already independent of the training
    code, which is what makes this cheap.
    """
    from src.inference.predictor import load_bundle

    for d in (MODELS_DIR / "current", BUNDLED):
        if (d / "model.txt").exists() and (d / "model_meta.json").exists():
            try:
                return Predictor(load_bundle(d))
            except Exception:
                return None
    return None


def api_get(path: str) -> dict | None:
    """Call the FastAPI service, or return None when it is not configured.

    Empty by default: a published dashboard has no API beside it, and spending five
    seconds timing out against localhost on every page load is worse than saying so.
    """
    if not API_URL:
        return None
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{API_URL}{path}", timeout=5) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def api_post(path: str, payload: dict) -> tuple[int, dict | None]:
    """POST to the FastAPI service. Returns (0, None) when it is unreachable."""
    if not API_URL:
        return 0, None
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"{API_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None
