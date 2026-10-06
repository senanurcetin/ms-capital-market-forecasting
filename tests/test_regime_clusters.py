"""The regime describer: features in, descriptors out, nothing from labels or from sample order."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.evaluation import regime_clusters as rc


def _pool(n: int = 4000, seed: int = 0, vol_scale: float = 1.0) -> pd.DataFrame:
    """Three market states with clearly different volatility, spread, depth and intensity."""
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 3, n)
    base = np.array([1.0, 3.0, 9.0])[state] * vol_scale
    cols = {
        "mkt_mid_std_60s": 1e-4 * base * rng.lognormal(0, 0.1, n),
        "mkt_mid_std_300s": 2e-4 * base * rng.lognormal(0, 0.1, n),
        "mkt_rel_spread_mean_60s": 5e-4 * base * rng.lognormal(0, 0.1, n),
        "mkt_total_depth_mean_60s": 1e5 / base * rng.lognormal(0, 0.1, n),
        "txn_intensity_60s": 0.5 * base * rng.lognormal(0, 0.1, n),
        "ord_n_total": 40 * base * rng.lognormal(0, 0.1, n),
    }
    return pd.DataFrame(cols)


def test_descriptors_have_the_declared_columns_and_no_nan_even_with_missing_inputs():
    pool = _pool()
    holes = pool.copy()
    holes.iloc[::9, 0] = np.nan  # a NaN volatility must not become a NaN descriptor
    desc = rc.RegimeDescriber().fit(pool).describe(holes)
    assert list(desc.columns) == rc.descriptor_names()
    assert not desc.isna().any().any()


def test_it_needs_no_target_and_no_month_column():
    pool = _pool()
    assert "target" not in pool.columns and "month" not in pool.columns
    rc.RegimeDescriber().fit(pool).describe(pool)  # runs on features alone


def test_cluster_shares_sum_to_one_and_match_the_assigned_cluster():
    pool = _pool()
    d = rc.RegimeDescriber().fit(pool)
    assert d.share_ is not None and d.share_.sum() == pytest.approx(1.0)
    desc = d.describe(pool)
    for share in desc["regime_share"].unique():
        assert np.isclose(share, d.share_, atol=1e-6).any()


def test_the_descriptors_do_not_depend_on_where_a_row_sits_in_the_pool():
    pool = _pool()
    d = rc.RegimeDescriber().fit(pool)
    a = d.describe(pool)
    shuffled = pool.sample(frac=1.0, random_state=3)
    b = d.describe(shuffled).loc[pool.index]
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_centres_are_in_the_signals_own_units_so_two_fits_are_comparable():
    pool = _pool()
    desc = rc.RegimeDescriber().fit(pool).describe(pool)
    centre = desc["regime_centre_mkt_mid_std_60s"].to_numpy()
    own = np.log(pool["mkt_mid_std_60s"].to_numpy() + rc.LOG_FLOOR)
    # a cluster centre in log units sits near its members' own log values
    assert np.abs(centre - own).mean() < 0.5


def test_a_describer_fitted_on_another_pool_describes_this_one_differently():
    """The static and the transductive variants are not the same thing under a shift."""
    train, shifted = _pool(seed=1), _pool(seed=2, vol_scale=2.0)
    static = rc.RegimeDescriber().fit(train).describe(shifted)
    own = rc.RegimeDescriber().fit(shifted).describe(shifted)
    assert static["regime_dist"].mean() > own["regime_dist"].mean()
    assert not np.allclose(static["regime_share"], own["regime_share"])


def test_the_same_seed_gives_the_same_descriptors():
    pool = _pool()
    a = rc.RegimeDescriber(seed=5).fit(pool).describe(pool)
    b = rc.RegimeDescriber(seed=5).fit(pool).describe(pool)
    pd.testing.assert_frame_equal(a, b)


def test_paired_verdict_reads_the_interval_not_the_point_estimate():
    noise = np.random.default_rng(0).normal(0, 0.01, 23)
    shown = rc.paired_verdict(noise + 0.02)
    assert shown["adds_signal"] and shown["ci_low"] > 0
    not_shown = rc.paired_verdict(noise + 0.001)
    assert not not_shown["adds_signal"] and "not shown" in not_shown["verdict"]
    assert shown["worth_keeping"] and not not_shown["worth_keeping"]
