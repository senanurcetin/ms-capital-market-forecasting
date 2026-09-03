"""End-to-end integration test: the demo pipeline must actually run.

This is the only test that exercises the whole stack in one go - synthetic data with
the real quirks, the genuine column-group converter, the walk-forward harness with its
leakage guard, hold-out measurement, and artefact save/reload. It needs no credentials
and no downloaded data, so it runs in CI.

It is kept small (a few thousand samples, two cheap models) to stay fast.
"""
import numpy as np
import pytest

from src.demo import run


@pytest.fixture(scope="module")
def demo_result():
    return run(samples=2000, models=["zero", "ridge"])


def test_demo_completes_and_reports_scores(demo_result):
    assert demo_result["ingest_ok"]
    assert demo_result["n_features"] > 250
    assert set(demo_result["scores"]) >= {"zero", "ridge"}


def test_zero_baseline_scores_exactly_zero(demo_result):
    """The control: a constant-zero prediction must score exactly 0 cosine."""
    assert demo_result["scores"]["zero"] == pytest.approx(0.0, abs=1e-12)


def test_model_beats_the_zero_baseline_on_planted_signal(demo_result):
    """The synthetic target carries a planted signal, so ridge must find some of it.

    If this fails, either the generator stopped planting signal or the training
    harness stopped learning - both worth knowing about immediately.
    """
    assert demo_result["scores"]["ridge"] > 0.05


def test_holdout_is_measured_and_finite(demo_result):
    ho = demo_result["holdout"]
    assert set(ho) >= {"cosine", "rmse", "directional_accuracy"}
    assert all(np.isfinite(v) for v in ho.values())


def test_servable_artefact_is_written(demo_result):
    from pathlib import Path

    from src.inference.predictor import load_bundle

    bundle = load_bundle(Path(demo_result["artefact"]))
    assert bundle.name == "lightgbm" and bundle.version == "demo"
    assert len(bundle.features) == demo_result["n_features"]
