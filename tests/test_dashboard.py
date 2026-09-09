"""The dashboard has to work away from the machine that produced its numbers.

Everything it displays normally comes from `C:/mscapital_data`, which exists on one
laptop. Publishing it means the numbers travel in `results/`, and these tests pin the
fallback that makes that work - because the failure mode is silent: away from the pipeline
the loaders return None, every page renders "not available yet", and the deployed
dashboard looks broken rather than erroring.

They run with MSCAPITAL_DATA_ROOT pointed at nothing, which is exactly the deployed
condition.
"""
import ast
import importlib
import os
import re
import sys

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


# ------------------------------------------------- the bundle has to be IN the repository

def test_every_bundled_result_is_tracked_by_git():
    """Present on disk is not the same as present in the repository.

    This is how it failed the first time: .gitignore carries `*.parquet` for the raw data,
    which silently swallowed results/feature_sample.parquet. Every loader still passed
    locally - the file was there - while the deployed dashboard would have rendered empty
    pages, which reads as a broken app rather than a missing file.
    """
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    if not results.exists():
        pytest.skip("results/ has not been exported")

    tracked = subprocess.run(
        ["git", "ls-files", "results/"], cwd=root, capture_output=True, text=True,
    ).stdout.split()
    on_disk = {f"results/{p.name}" for p in results.iterdir() if p.is_file()}
    assert not on_disk - set(tracked), (
        "these exist on disk but are not in git, so they will not reach a deployment"
    )


def test_the_deployment_needs_no_secrets():
    """Streamlit Community Cloud gets no credentials, so nothing may require them."""
    from pathlib import Path

    lib = (Path(__file__).resolve().parents[1] / "streamlit_app" / "lib.py").read_text(
        encoding="utf-8")
    assert "gcp_key_path" not in lib and "service_account" not in lib


# ------------------------------------------------- every page must read through lib

def test_no_page_addresses_the_data_directories_directly():
    """Pages that build their own paths cannot see the exported bundle.

    Explainability and Backtesting each did, so on a deployment those two rendered "not
    available yet" while every loader test still passed - the tests exercise lib, and
    those pages were not going through lib. Reading through load_csv/load_json is what
    makes the two-source fallback apply everywhere rather than in most places.
    """
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "streamlit_app"
    offenders = [
        f.name for f in [*app.glob("pages/*.py"), app / "app.py"]
        if any(t in f.read_text(encoding="utf-8") for t in ("MODELS_DIR /", "FEATURES_DIR /"))
    ]
    assert not offenders, f"these build their own paths instead of using lib: {offenders}"


def test_every_page_puts_the_repo_root_on_the_path():
    """A bare `streamlit run` adds the main script's directory, not the repository root.

    Without the bootstrap, `from streamlit_app.lib import ...` raises ModuleNotFoundError
    on Streamlit Community Cloud while working perfectly under `python -m streamlit`.
    """
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "streamlit_app"
    missing_boot = [
        f.name for f in [*app.glob("pages/*.py"), app / "app.py"]
        if f.stem != "__init__" and "sys.path.insert" not in f.read_text(encoding="utf-8")
    ]
    assert not missing_boot, f"no sys.path bootstrap in: {missing_boot}"


# ------------------------------------------- the deployment installs requirements.txt ONLY

def _requirement_names():
    """Distribution names pinned in the file Streamlit Community Cloud installs."""
    from pathlib import Path

    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    names = set()
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(re.split(r"[=<>!~\[]", line)[0].strip().lower().replace("-", "_"))
    return names


def _imports_of(path):
    """Top-level distribution names imported by a module, read from its AST.

    Static rather than executed: importing a Streamlit page runs it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_the_dashboard_imports_nothing_outside_requirements():
    """Every import the published app makes must be installable from requirements.txt.

    This is the check that was missing. The clean-clone test cloned the CODE into a fresh
    directory and ran it in MY virtualenv, where matplotlib, mlflow and shap were already
    present from the pipeline - so it proved the repository was complete and proved nothing
    at all about the environment. Streamlit Community Cloud installs requirements.txt into
    an empty interpreter, which is a strictly harder test than the one being run.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    allowed = (
        _requirement_names()
        | set(sys.stdlib_module_names)
        | {"src", "streamlit_app", "api"}          # first-party
        # Distribution name on the left of the pin, import name in the code - these three
        # differ, and treating them as unknown would fail the test on a package that IS
        # installed.
        | {"sklearn", "yaml", "xgboost"}
    )
    offenders = {}
    for f in [*(root / "streamlit_app").rglob("*.py"), *(root / "src").rglob("*.py")]:
        if f.name == "__init__.py":
            continue
        extra = {m for m in _imports_of(f) if m.lower().replace("-", "_") not in allowed}
        if extra:
            offenders[str(f.relative_to(root))] = sorted(extra)

    # src/ carries the pipeline, which is allowed to need requirements-pipeline.txt. The
    # dashboard is not: it has to run on the lean set.
    dash = {k: v for k, v in offenders.items() if k.startswith("streamlit_app")}
    assert not dash, f"imported but not in requirements.txt: {dash}"


def test_no_page_calls_a_styler_method_that_needs_matplotlib():
    """A missing dependency does not have to be an import to break the deploy.

    `Styler.background_gradient` reaches for matplotlib at RENDER time, inside pandas -
    nothing in this repository imports it, so the import test above cannot see it, and the
    page raised ImportError on Streamlit Cloud while every local test passed. The same
    holds for `Styler.bar` and `Styler.text_gradient`.
    """
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "streamlit_app"
    banned = ("background_gradient", "text_gradient", ".bar(")

    def code_of(path):
        """Source with comments and docstrings dropped.

        A first version scanned raw text and flagged the comment in app.py explaining why
        background_gradient is NOT used - a test that forbids naming the thing it forbids
        would push the reasoning out of the file.
        """
        return ast.unparse(ast.parse(path.read_text(encoding="utf-8")))

    offenders = {}
    for f in app.rglob("*.py"):
        hits = [b for b in banned if b in code_of(f)]
        if hits:
            offenders[f.name] = hits
    assert not offenders, (
        f"these need matplotlib, which the runtime set does not carry: {offenders}"
    )
