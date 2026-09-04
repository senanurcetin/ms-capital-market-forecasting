"""Tests for adversarial validation.

The conclusion drawn from this module - that the test set is a continuation of the
training period rather than something apart from it - rests on AUCs being comparable
across comparisons. So the tests cover the two ways that comparability can break: an
in-sample AUC (which measures memorisation), and mismatched heterogeneity between the two
sides (which was a real bug here, and inverted the answer until it was fixed).
"""
import numpy as np
import pandas as pd

from src.evaluation.adversarial import REFERENCE, TARGETS, _auc, _sample


def _frame(n, loc=0.0, scale=1.0, seed=0, k=6):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(loc, scale, (n, k)),
                        columns=[f"f{i}" for i in range(k)]).astype("float32")


FEATURES = [f"f{i}" for i in range(6)]


def test_identical_distributions_are_indistinguishable():
    a, b = _frame(4000, seed=1), _frame(4000, seed=2)
    auc, _ = _auc(a, b, FEATURES)
    assert 0.45 < auc < 0.55


def test_clearly_shifted_distributions_are_separable():
    a, b = _frame(4000, loc=0.0, seed=1), _frame(4000, loc=3.0, seed=2)
    assert _auc(a, b, FEATURES)[0] > 0.95


def test_auc_is_out_of_sample():
    """Fitting and scoring on the same rows would report separability that is not there.

    With this many columns and rows a booster can memorise noise, so an in-sample AUC on
    identical distributions would rise well above 0.5. It must not.
    """
    a, b = _frame(600, seed=1), _frame(600, seed=2)
    auc, _ = _auc(a, b, FEATURES)
    assert auc < 0.62, "identical distributions must not look separable"


def test_pooling_one_side_depresses_the_auc():
    """The confound this module's design exists to avoid.

    A narrow block is easy to tell from another narrow block, but hard to tell from a
    heterogeneous pool that CONTAINS something like it. Comparing a pooled sample against
    a block-vs-block calibration curve therefore understates the shift.
    """
    block_a = _frame(3000, loc=0.0, seed=1)
    block_b = _frame(3000, loc=2.0, seed=2)
    pool = pd.concat([_frame(1000, loc=x, seed=10 + i)
                      for i, x in enumerate((0.0, 1.0, 2.0))], ignore_index=True)
    auc_block = _auc(block_a, block_b, FEATURES)[0]
    auc_pool = _auc(pool, block_b, FEATURES)[0]
    assert auc_pool < auc_block


def test_importance_is_returned_sorted_and_complete():
    a, b = _frame(2000, seed=1), _frame(2000, loc=1.0, seed=2)
    _, imp = _auc(a, b, FEATURES)
    assert list(imp.index.sort_values()) == sorted(FEATURES)
    assert (imp.to_numpy()[:-1] >= imp.to_numpy()[1:]).all()


def test_the_discriminating_feature_ranks_first():
    a = _frame(3000, seed=1)
    b = _frame(3000, seed=2)
    b["f3"] = (b["f3"] + 5.0).astype("float32")     # only f3 moves
    _, imp = _auc(a, b, FEATURES)
    assert imp.index[0] == "f3"


def test_sample_is_a_noop_when_small_enough():
    df = _frame(100)
    assert len(_sample(df, 500)) == 100


def test_sample_is_deterministic():
    df = _frame(1000)
    assert _sample(df, 100, seed=3).equals(_sample(df, 100, seed=3))


def test_reference_block_precedes_every_target():
    assert all(lo > REFERENCE[1] for lo, _ in TARGETS)


def test_target_blocks_do_not_overlap():
    for (_, hi), (lo, _) in zip(TARGETS, TARGETS[1:]):
        assert lo > hi
