"""The two entry points that build what gets published, run end to end on fake inputs.

`export_results.run()` and `ship.build()` were the least-covered code in the project, for
the same reason: the real ones need `C:/mscapital_data` and hours of training. So the
functions that decide what the world sees were the ones nothing exercised.

Both are driven here against a synthetic data root, which costs seconds and covers the
decisions that actually go wrong: which files travel, what happens when one is missing,
whether the size guard fires, and whether the three time splits ship.py cuts really are
disjoint.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from src.data import export_results
from src.models import ship

MONTHS = 71
ROWS_PER_MONTH = 40


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    """A data root shaped like the real one, with a couple of files deliberately absent."""
    features = tmp_path / "features"
    models = tmp_path / "models" / "current"
    features.mkdir(parents=True)
    models.mkdir(parents=True)

    for name in export_results.VERBATIM:
        if name in ("recency.csv", "recency_meta.json"):
            continue                                   # left out on purpose, see below
        if name.endswith(".json"):
            (features / name).write_text('{"value": 1}', encoding="utf-8")
        else:
            pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}).to_csv(features / name,
                                                                index=False)

    pd.DataFrame({"feature": ["f0"], "mean_abs_shap": [0.5]}).to_csv(
        models / "shap_global.csv", index=False)
    (models / "model_meta.json").write_text('{"name": "x"}', encoding="utf-8")

    # 20k points, so the downsampling branch is the one that runs.
    pd.DataFrame({"equity": np.linspace(1.0, 1.2, 20_000)}).to_csv(
        features / "backtest_equity.csv", index=False)

    df = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 4)),
                      columns=[f"f{i}" for i in range(4)])
    df.insert(0, "sample_id", np.arange(200))
    df.to_parquet(features / "dataset_train.parquet", index=False)

    monkeypatch.setattr(
        export_results, "load_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(features=str(features),
                                                      data_root=str(tmp_path))),
    )
    return tmp_path


# ------------------------------------------------------------------ export_results.run

def test_the_export_copies_what_it_finds(fake_root, tmp_path):
    out = tmp_path / "out"
    res = export_results.run(out=out)
    assert (out / "walkforward_summary.csv").exists()
    assert (out / "shap_global.csv").exists()
    assert res["bytes"] > 0


def test_a_missing_source_is_reported_rather_than_silently_dropped(fake_root, tmp_path):
    """The whole risk with this function: it warns and carries on.

    That is right for a partially-run pipeline, but it means an incomplete bundle is a
    normal-looking success. The names have to come back in `skipped` so a caller - or a
    reader of the log - can tell the difference between "exported" and "exported most of
    it". `recency.*` and the two shap_local files are absent from the fixture on purpose.
    """
    res = export_results.run(out=tmp_path / "out")
    assert "recency.csv" in res["skipped"]
    assert "recency_meta.json" in res["skipped"]
    assert "shap_local_examples.csv" in res["skipped"]
    assert "recency.csv" not in res["copied"]


def test_the_equity_curve_is_thinned_on_the_way_in(fake_root, tmp_path):
    out = tmp_path / "out"
    export_results.run(out=out)
    assert len(pd.read_csv(out / "backtest_equity.csv")) <= export_results.EQUITY_POINTS


def test_the_feature_sample_is_capped_and_full_width(fake_root, tmp_path):
    """Rows are capped; COLUMNS are not. A narrow sample would break the Predictions page,
    which has to assemble a complete row because Predictor refuses an incomplete one."""
    out = tmp_path / "out"
    export_results.run(out=out)
    sample = pd.read_parquet(out / "feature_sample.parquet")
    source = pd.read_parquet(fake_root / "features" / "dataset_train.parquet")
    assert len(sample) <= export_results.SAMPLE_ROWS
    assert list(sample.columns) == list(source.columns)


def test_the_export_refuses_to_produce_a_bundle_too_big_for_git(fake_root, tmp_path,
                                                                monkeypatch):
    """The guard is the last thing standing between a stray parquet and a 2 GB clone."""
    monkeypatch.setattr(export_results, "SAMPLE_ROWS", 200)
    out = tmp_path / "out"
    out.mkdir()
    (out / "stowaway.bin").write_bytes(b"\0" * (25 * 1024 * 1024))
    with pytest.raises(AssertionError, match="too large"):
        export_results.run(out=out)


# ------------------------------------------------------------------ ship.build

def test_the_three_splits_do_not_overlap():
    """Pure arithmetic on the constants, and worth pinning on its own.

    Fitting the stopping rounds and the blend weights on the same rows would let the
    weights compensate for a stopping point chosen on those same rows - a leak that shows
    up as a better blend score and nothing else.
    """
    train = set(range(0, ship.TRAIN_END + 1))
    stop = {ship.STOP_MONTH}
    blend = set(range(ship.BLEND_MONTHS[0], ship.BLEND_MONTHS[1] + 1))
    assert not train & stop and not train & blend and not stop & blend
    assert max(blend) == MONTHS - 1, "the last month is not being used"


@pytest.fixture()
def fake_training_root(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    n = MONTHS * ROWS_PER_MONTH
    X = rng.normal(size=(n, 5))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df.insert(0, "sample_id", np.arange(n))
    df.insert(1, "month", np.repeat(np.arange(MONTHS), ROWS_PER_MONTH))
    df["target"] = 0.4 * X[:, 0] + rng.normal(scale=0.5, size=n)

    features = tmp_path / "features"
    features.mkdir(parents=True)
    monkeypatch.setattr(ship, "load_dataset", lambda split: df)
    monkeypatch.setattr(
        ship, "load_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(features=str(features),
                                                      data_root=str(tmp_path))),
    )
    return tmp_path


def test_build_writes_a_servable_ensemble(fake_training_root):
    """End to end: fit, blend, save - then load it back the way the API would.

    The defect this covers shipped for real. `build()` saved the bare LightGBM booster
    under the name "ensemble", so /model-info announced a blend while /predict returned
    one model's predictions, and the artefact disagreed with the score reported beside it.
    """
    from src.inference.predictor import load_bundle

    meta = ship.build(rounds=15, early_stopping=5, version="test")
    model_dir = fake_training_root / "models" / "shipped"

    bundle = load_bundle(model_dir)
    assert bundle.name == "ensemble"
    assert set(bundle.model.weights) == {"lightgbm", "xgboost", "ridge"}
    assert sum(bundle.model.weights.values()) == pytest.approx(1.0, abs=1e-6)

    written = json.loads((fake_training_root / "features" / "ship_meta.json")
                         .read_text(encoding="utf-8"))
    assert written["weights"] == pytest.approx(meta["weights"])
    assert written["months_used"] == ship.TRAIN_END + 1


def test_the_saved_artefact_predicts(fake_training_root):
    """A bundle that loads but cannot score is still a broken deployment."""
    from src.inference.predictor import Predictor

    ship.build(rounds=15, early_stopping=5, version="test")
    predictor = Predictor.from_dir(fake_training_root / "models" / "shipped")

    rows = [dict.fromkeys(predictor.bundle.features, 0.1)]
    out = predictor.predict(rows)
    assert out.shape == (1,) and np.isfinite(out).all()
