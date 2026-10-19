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

import base64
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from src.inference.predictor import Predictor

DATA_ROOT = Path(os.environ.get("MSCAPITAL_DATA_ROOT", "C:/mscapital_data"))
API_URL = os.environ.get("MSCAPITAL_API_URL", "")
API_KEY = os.environ.get("MSCAPITAL_API_KEY", "")


def _api_headers(extra: dict | None = None) -> dict:
    """Headers for a call to the API: the key when one is configured, plus any extras."""
    headers = dict(extra or {})
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers

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


def inject_dashboard_theme() -> None:
    """Apply report-only styling; no model results or metrics are modified.

    The UI retains its original research source data. The rules use Streamlit's
    data-testid hooks and are progressive enhancement: if a selector changes,
    native Streamlit remains fully functional.
    """
    css = """
        <style>
        :root {
          --ms-ink: #0B0F19;
          --ms-card: #061524;
          --ms-border: #28394E;
          --ms-teal: #00E0FF;
        }
        [data-testid="stSidebar"] {
          border-right: 1px solid #284657;
        }
        [data-testid="stSidebarNav"] { padding-top: 1rem; }
        [data-testid="stSidebarNav"] a {
          border-radius: 10px;
          margin: 2px 8px;
          transition: background .12s ease;
        }
        [data-testid="stSidebarNav"] a:hover { text-decoration: underline; }
        .block-container {
          max-width: 1480px;
          padding-top: clamp(1.5rem,3vw,2.6rem);
          padding-bottom: 3.3rem;
        }
        [data-testid="stMetric"] {
          background: linear-gradient(150deg,#102238,#061524);
          border: 1px solid #325866;
          border-radius: 16px;
          padding: clamp(1rem,1.4vw,1.45rem);
          box-shadow: 0 15px 30px rgba(2,12,24,.17);
          min-height: 126px;
        }
        [data-testid="stMetricLabel"] {
          color: #B9C5CE;
          text-transform: uppercase;
          letter-spacing: .055em;
          font-weight: 700;
          font-size: .76rem;
        }
        [data-testid="stMetricValue"] {
          color: #edfffb;
          font-variant-numeric: tabular-nums;
          letter-spacing: -.03em;
        }
        [data-testid="stMetricDelta"] { font-variant-numeric: tabular-nums; }
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stVegaLiteChart"],
        [data-testid="stPlotlyChart"] {
          border: 1px solid #294657;
          border-radius: 15px;
          overflow: hidden;
        }
        [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
          min-width: 0;
        }
        [data-testid="stAlert"] { border-radius: 13px; }
        [data-testid="stExpander"] { border: 1px solid #305162; border-radius: 14px; }
        [data-testid="stMarkdownContainer"] h1 {
          color: inherit;
          letter-spacing: -.036em;
          font-size: clamp(2rem,3.3vw,3.15rem);
          line-height: 1.13;
        }
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 { color: inherit; letter-spacing: -.015em; }
        hr { border-color: #2a4958; }
        .ms-report-eyebrow {
          display: inline-flex;
          gap: .5rem;
          align-items: center;
          padding: .44rem .9rem;
          background: #162238;
          border: 1px solid #405482;
          border-radius: 999px;
          color: #8F9BFF;
          font-size: .72rem;
          font-weight: 800;
          letter-spacing: .11em;
          margin-bottom: .65rem;
        }
        .ms-report-footnote { color: #B9C5CE; font-size: .84rem; margin-top: .5rem; }
        @media (max-width: 780px) {
          [data-testid="stMetric"] { min-height: 108px; }
          .block-container { padding-inline: 1rem; }
        }
        </style>
        """
    # CSS must follow the native theme, including a browser's System preference.
    # A fixed dark background with native light-theme text makes navigation unreadable.
    if st.context.theme.type == "light":
        light_palette = {
            "#0B0F19": "#F7F9FD", "#061524": "#FFFFFF",
            "#E4EDF5": "#1C2430", "#162B45": "#E5EBF6",
            "#102238": "#F0F4FB", "#B9C5CE": "#526173",
            "#edfffb": "#122238", "#f3fffb": "#122238",
            "#D4D9FF": "#3C4482", "#162238": "#EBEDF9",
            "#8F9BFF": "#454E98", "#00E0FF": "#087A94",
        }
        for dark, light in light_palette.items():
            css = css.replace(dark, light)
    st.markdown(css, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    inject_dashboard_theme()
    logo = Path(__file__).resolve().parents[1] / "docs/assets/nexus-official-transparent.svg"
    encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{encoded}" alt="NEXUS" '
        'width="210" height="63" style="object-fit:contain;object-position:left;'
        'max-width:100%;background:#0B0F19;border-radius:8px">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="ms-report-eyebrow">NEXUS / MSCAPITAL / QUANTITATIVE RESEARCH</span>',
        unsafe_allow_html=True,
    )
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
        req = urllib.request.Request(f"{API_URL}{path}", headers=_api_headers())
        with urllib.request.urlopen(req, timeout=5) as r:
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
        headers=_api_headers({"Content-Type": "application/json"}),
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
