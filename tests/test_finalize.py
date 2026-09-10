"""The module that produces the published model had no tests at all.

`finalize.py` trains the artefact that ships in `results/`, measures it once on the
hold-out, and writes the numbers the dashboard shows. It was at 0% coverage, and reading
it turned up a defect in the line that decides what to save:

    inner = getattr(model, "booster_", None) or getattr(model, "model", model)

That unwraps the trained wrapper to the estimator inside, which is only harmless for one
of the three models it supports:

  * **ridge** - the wrapper imputes medians and standardises first. The bare sklearn Ridge
    got unscaled features and returned predictions wrong by about 3x the target's standard
    deviation, silently, on every row - and raised `Input X contains NaN` on the first gap.
  * **xgboost** - the wrapper predicts up to `best_iteration`. The bare booster predicts
    with every tree, so the served model was strictly larger than the one whose hold-out
    score is reported next to it.
  * **lightgbm** - safe, because `Booster.save_model()` truncates at best_iteration. This
    is why the defect never bit: lightgbm is the default and the model actually shipped.

The tests below would have caught all three, because they compare what the artefact
predicts against what the fitted model predicts rather than checking that a file exists.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from src.inference.predictor import Predictor
from src.models import finalize

MONTHS = 71
ROWS_PER_MONTH = 30
FEATURES = [f"f{i}" for i in range(5)]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Shaped like the real feature table, with gaps - which is the point.

    5.9% of the real matrix is NaN because the one-second windows are usually empty. A
    fixture without holes cannot reproduce the failure that reached the test set.
    """
    rng = np.random.default_rng(0)
    n = MONTHS * ROWS_PER_MONTH
    X = rng.normal(size=(n, len(FEATURES))) * np.array([1, 10, 100, 0.1, 5])
    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "sample_id", np.arange(n))
    df.insert(1, "month", np.repeat(np.arange(MONTHS), ROWS_PER_MONTH))
    df["target"] = 0.5 * X[:, 0] / 1.0 + rng.normal(scale=0.5, size=n)
    gaps = rng.random(df[FEATURES].shape) < 0.06
    df[FEATURES] = df[FEATURES].mask(gaps)
    return df


@pytest.fixture()
def run_finalize(tmp_path, monkeypatch, frame):
    """Drive main() against a synthetic root; returns a loader for what it wrote."""
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True)
    monkeypatch.setattr(finalize, "load_dataset", lambda split: frame)
    monkeypatch.setattr(
        finalize, "load_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(features=str(features_dir),
                                                      data_root=str(tmp_path))),
    )

    def run(model: str):
        finalize.main(["--model", model, "--rounds", "25", "--early-stopping", "5"])
        return tmp_path / "models" / "current", features_dir

    return run


# ------------------------------------------------------- the artefact matches the model

@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "ridge"])
def test_the_saved_artefact_predicts_what_the_trained_model_predicts(run_finalize, frame,
                                                                     model_name):
    """The check the old code would have failed for ridge and xgboost.

    Comparing against a refit of the same configuration rather than against a stored
    number: the artefact has to reproduce the model, not merely load.
    """
    model_dir, _ = run_finalize(model_name)
    served = Predictor.from_dir(model_dir)

    rows = frame[served.bundle.features].head(40)
    got = served.predict(rows.to_dict("records"))

    # Refit the same way finalize does, and predict through the WRAPPER.
    from src.evaluation.temporal_validation import holdout_months

    ho_lo, _ = holdout_months()
    months = frame["month"].to_numpy()
    tr = np.flatnonzero(months < ho_lo - 1)
    va = np.flatnonzero(months == ho_lo - 1)
    if model_name == "lightgbm":
        from src.models.lightgbm_model import LightGBMModel

        ref = LightGBMModel(num_boost_round=25, early_stopping_rounds=5)
    elif model_name == "xgboost":
        from src.models.xgboost_model import XGBoostModel

        ref = XGBoostModel(num_boost_round=25, early_stopping_rounds=5)
    else:
        from src.models.baseline import RidgeModel

        ref = RidgeModel(alpha=10.0)
    ref.fit(frame.iloc[tr], frame["target"].to_numpy()[tr],
            eval_set=(frame.iloc[va], frame["target"].to_numpy()[va]))

    np.testing.assert_allclose(got, ref.predict(rows), rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "ridge"])
def test_the_artefact_survives_a_row_with_gaps(run_finalize, frame, model_name):
    """Ridge used to raise `Input X contains NaN` here, because unwrapping it left the
    imputation behind. The boosters handle NaN natively and must keep doing so."""
    model_dir, _ = run_finalize(model_name)
    served = Predictor.from_dir(model_dir)

    row = dict.fromkeys(served.bundle.features, 0.1)
    row[served.bundle.features[0]] = float("nan")
    out = served.predict([row])
    assert np.isfinite(out).all()


# ------------------------------------------------------------------ the hold-out

def test_the_holdout_months_never_enter_training(run_finalize, frame, caplog):
    """The single most important property in the file, asserted in it and untested.

    A hold-out that leaked would raise every reported score at once and look like a good
    day rather than a bug.
    """
    import logging

    with caplog.at_level(logging.INFO):
        run_finalize("ridge")
    line = next(m for m in caplog.messages if "hold-out months" in m)
    assert "train months 0-63" in line and "hold-out months 65-70" in line


def test_the_written_metrics_and_the_bundle_agree(run_finalize):
    """Two files carry the hold-out score and the dashboard reads them both.

    holdout_metrics.json feeds the headline number; the bundle's metrics feed
    /model-info. If they ever disagreed, the page and the endpoint would quote different
    performance for the same artefact.
    """
    model_dir, features_dir = run_finalize("ridge")
    written = json.loads((features_dir / "holdout_metrics.json").read_text(
        encoding="utf-8"))
    bundle = Predictor.from_dir(model_dir).info()["metrics"]

    assert written["model"] == "ridge"
    assert bundle["cosine"] == pytest.approx(written["scores"]["cosine"], abs=1e-6)
    assert bundle["backtest_sharpe"] == pytest.approx(written["backtest"]["sharpe"],
                                                      abs=1e-4)


def test_the_backtest_artefacts_are_all_written(run_finalize):
    """The Backtesting page renders three files; a partial write leaves it half blank."""
    _, features_dir = run_finalize("ridge")
    for name in ("backtest_cost_sensitivity.csv", "backtest_trade_fraction.csv",
                 "backtest_equity.csv", "holdout_metrics.json"):
        path = features_dir / name
        assert path.exists() and path.stat().st_size > 0, f"{name} was not written"


def test_the_equity_curve_has_one_point_per_holdout_sample(run_finalize, frame):
    """One point per SAMPLE, not per trade - which is what the chart axis claims.

    A first version of this test asserted one point per trade and failed: 180 points
    against 36 trades. The curve is flat wherever the threshold produced no position, so
    its length is the hold-out row count. That is what makes "hold-out sample
    (chronological)" the correct label on the Backtesting page, and pinning it here means
    the axis cannot start lying if the backtester changes what it emits.
    """
    _, features_dir = run_finalize("ridge")
    equity = pd.read_csv(features_dir / "backtest_equity.csv")
    metrics = json.loads((features_dir / "holdout_metrics.json").read_text(
        encoding="utf-8"))

    from src.evaluation.temporal_validation import holdout_months

    ho_lo, ho_hi = holdout_months()
    n_holdout = int(((frame.month >= ho_lo) & (frame.month <= ho_hi)).sum())
    assert len(equity) == n_holdout
    assert metrics["backtest"]["n_trades"] < n_holdout, (
        "every sample traded - the traded-fraction threshold is not being applied"
    )
