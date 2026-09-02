"""Tests for the SHAP helpers - no real model or data required."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.explain import global_importance, local_explanations

FEATURES = ["mkt_mid_last", "ord_ofi_60s", "txn_intensity_60s", "mkt_spread_last"]


@pytest.fixture
def values():
    rng = np.random.default_rng(5)
    v = rng.normal(0, 0.1, (200, len(FEATURES)))
    v[:, 1] *= 10  # make ord_ofi_60s clearly the most important feature
    return v


def test_global_importance_ranks_by_mean_abs(values):
    out = global_importance(values, FEATURES)
    assert list(out.columns) == ["feature", "mean_abs_shap", "family", "share"]
    assert out.iloc[0]["feature"] == "ord_ofi_60s"
    assert out["mean_abs_shap"].is_monotonic_decreasing


def test_shares_sum_to_one(values):
    out = global_importance(values, FEATURES)
    assert out["share"].sum() == pytest.approx(1.0)


def test_family_extracted_from_prefix(values):
    out = global_importance(values, FEATURES)
    assert set(out["family"]) == {"mkt", "ord", "txn"}
    assert out.groupby("family")["share"].sum().idxmax() == "ord"


def test_sign_does_not_affect_global_importance():
    """Global importance is a MAGNITUDE measure; flipping signs must not reorder it."""
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, (100, len(FEATURES)))
    a = global_importance(v, FEATURES)["feature"].tolist()
    b = global_importance(-v, FEATURES)["feature"].tolist()
    assert a == b


def test_local_explanations_shape_and_ordering(values):
    X = pd.DataFrame(np.random.default_rng(2).normal(size=(200, len(FEATURES))),
                     columns=FEATURES)
    ids = np.arange(200)
    out = local_explanations(values[:3], X.iloc[:3], ids[:3], top_k=2)
    assert len(out) == 3 * 2
    assert set(out.columns) == {"sample_id", "feature", "feature_value", "shap_value"}
    for sid in out["sample_id"].unique():
        sel = out[out["sample_id"] == sid]
        assert sel["shap_value"].abs().is_monotonic_decreasing


def test_local_top_k_capped_by_feature_count(values):
    X = pd.DataFrame(np.zeros((2, len(FEATURES))), columns=FEATURES)
    out = local_explanations(values[:2], X, np.array([7, 8]), top_k=99)
    assert len(out) == 2 * len(FEATURES)


def test_local_feature_values_match_input():
    X = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=FEATURES)
    v = np.array([[0.0, 0.0, 5.0, 0.0]])
    out = local_explanations(v, X, np.array([42]), top_k=1)
    assert out.iloc[0]["feature"] == "txn_intensity_60s"
    assert out.iloc[0]["feature_value"] == pytest.approx(3.0)
    assert out.iloc[0]["sample_id"] == 42
