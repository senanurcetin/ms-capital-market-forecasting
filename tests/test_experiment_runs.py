"""The experiment drivers end to end, on a small synthetic feature table.

These exist because the drivers hold the plumbing that no unit test of a single function touches:
the split by month, the on-disk cache that makes a restart cost one fit, the blend weights fitted on
validation and applied to test, the paired per-month verdict and the files written. Two of them were
lost to a memory limit and a container restart before they ever produced a result.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from src.evaluation import regime_clusters as rc
from src.models import ensemble_probe as ep


def _table(rows_per_month: int = 150, seed: int = 0) -> pd.DataFrame:
    """71 months of samples; the regime signals are positive, the target carries a little signal."""
    rng = np.random.default_rng(seed)
    n = 71 * rows_per_month
    cols = {c: rng.lognormal(0, 0.5, n).astype("float32") for c in rc.STATE_FEATURES}
    for i in range(4):
        cols[f"extra_{i}"] = rng.normal(size=n).astype("float32")
    df = pd.DataFrame(cols)
    df.insert(0, "sample_id", np.arange(n))
    df.insert(1, "month", np.repeat(np.arange(71), rows_per_month))
    df["target"] = (0.002 * df["extra_0"].to_numpy() + rng.normal(0, 0.003, n)).astype("float32")
    return df


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    feats = tmp_path / "features"
    feats.mkdir()
    _table().to_parquet(feats / "dataset_train.parquet", index=False)
    cfg = SimpleNamespace(paths=SimpleNamespace(features=str(feats), data_root=str(tmp_path)))
    for module in (rc, ep):
        monkeypatch.setattr(module, "load_config", lambda: cfg)
    return tmp_path


# ------------------------------------------------------------------------- regime clusters
def test_regime_clusters_run_writes_results_and_caches_each_fit(workdir):
    out = workdir / "out"
    result = rc.run(seeds=(0,), out_dir=out)
    assert set(result["pooled_test_cosine"]) == {"plain", "static", "transductive"}
    for key in ("paired_static", "paired_transductive"):
        assert result[key]["n_months"] == 23  # months 48-70
        assert result[key]["adds_signal"] in (True, False)
    assert len(result["cluster_shares_test"]) == rc.N_CLUSTERS
    assert sum(result["cluster_shares_test"]) == pytest.approx(1.0)
    assert (out / "regime_clusters.json").exists() and (out / "regime_clusters_months.csv").exists()
    cached = sorted(p.name for p in (workdir / "features" / "regime_cache").iterdir())
    assert cached == ["plain_seed0.npy", "static_seed0.npy", "transductive_seed0.npy"]


def test_regime_clusters_a_second_run_reuses_the_cached_predictions(workdir):
    first = rc.run(seeds=(0,), out_dir=workdir / "a")
    second = rc.run(seeds=(0,), out_dir=workdir / "b")  # nothing is refitted
    assert first["pooled_test_cosine"] == second["pooled_test_cosine"]


# ------------------------------------------------------------------------- ensemble probe
def test_ensemble_probe_run_compares_the_blends_and_writes_results(workdir):
    members = (*ep.BASE, "lightgbm_huber", "lightgbm_extra")
    out = workdir / "out"
    r = ep.run(members=members, out_dir=out)
    assert set(r["alone"]) == set(members)
    assert r["paired_extended_vs_base"]["n_months"] == 23
    assert set(r["each_extra_added_to_base"]) == {"lightgbm_huber", "lightgbm_extra"}
    for blend_name in ("base", "extended"):
        assert sum(r["weights"][blend_name].values()) == pytest.approx(1.0)
    assert set(r["weights"]["base"]) == set(ep.BASE)
    assert set(r["weights"]["extended"]) == set(members)
    saved = json.loads((out / "ensemble_probe.json").read_text(encoding="utf-8"))
    assert saved["pooled_test_cosine"] == r["pooled_test_cosine"]
    assert len(pd.read_csv(out / "ensemble_probe_months.csv")) == 23
    assert sorted(p.stem for p in (workdir / "features" / "ensemble_cache").iterdir()) == sorted(members)


def test_ensemble_probe_weights_are_fitted_on_validation_not_on_test(workdir):
    """Nothing in the blend may see the test months: weights come from the val predictions."""
    members = ep.BASE
    r = ep.run(members=members, out_dir=workdir / "out")
    cache = workdir / "features" / "ensemble_cache"
    val = {m: np.load(cache / f"{m}.npz")["val"] for m in members}
    test = {m: np.load(cache / f"{m}.npz")["test"] for m in members}
    df = pd.read_parquet(workdir / "features" / "dataset_train.parquet")
    y_val = df.loc[df["month"].between(*ep.VAL_MONTHS), "target"].to_numpy(dtype=float)
    _, w = ep.blend(ep.BASE, val, test, y_val)
    assert r["weights"]["base"] == pytest.approx(w)
