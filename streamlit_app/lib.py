"""Shared data access for the Streamlit pages.

Design: every loader returns None instead of crashing on missing data, so the
dashboard stays usable at any stage of the pipeline and tells the user what is
missing - an explicit state rather than a blank screen.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_ROOT = Path(os.environ.get("MSCAPITAL_DATA_ROOT", "C:/mscapital_data"))
API_URL = os.environ.get("MSCAPITAL_API_URL", "http://localhost:8000")

FEATURES_DIR = DATA_ROOT / "features"
MODELS_DIR = DATA_ROOT / "models"

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


@st.cache_data(show_spinner=False)
def load_summary() -> dict | None:
    for name in ("walkforward_summary.json", "smoke_summary.json"):
        p = FEATURES_DIR / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner=False)
def load_results_table() -> pd.DataFrame | None:
    for name in ("walkforward_summary.csv", "smoke_summary.csv"):
        p = FEATURES_DIR / name
        if p.exists():
            return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_features(n_rows: int = 50_000, columns: list[str] | None = None) -> pd.DataFrame | None:
    """A slice of the feature set. The full file is ~1.4 GB and is never fully loaded."""
    p = FEATURES_DIR / "dataset_train.parquet"
    if not p.exists():
        return None
    import pyarrow.parquet as pq

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
    p = FEATURES_DIR / "dataset_train.parquet"
    if not p.exists():
        return None
    import pyarrow.parquet as pq

    return pq.ParquetFile(p).schema_arrow.names


def api_get(path: str) -> dict | None:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{API_URL}{path}", timeout=5) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def api_post(path: str, payload: dict) -> tuple[int, dict | None]:
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
