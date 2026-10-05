"""The pruning experiment's pure parts: which columns go, and what the rule concludes."""
import numpy as np
import pandas as pd
import pytest
from src.evaluation.feature_audit import correlation_pairs, duplicate_groups, effective_count
from src.evaluation.feature_prune import NON_INFERIORITY_MARGIN, features_to_drop, verdict


def _frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame({
        "a": a,
        "a_copy": a * 3.0,                 # a constant multiple: not a second feature
        "b": rng.normal(size=n),
        "a_again": a + 1e-9,               # a third copy of the same thing
        "c": rng.normal(size=n),
    })


def test_first_column_of_each_duplicate_group_survives():
    df = _frame()
    cols = list(df.columns)
    drop = features_to_drop(correlation_pairs(df, cols), cols)
    assert drop == ["a_copy", "a_again"]


def test_dropped_columns_are_reported_in_table_order():
    df = _frame()[["c", "a_again", "b", "a_copy", "a"]]
    cols = list(df.columns)
    drop = features_to_drop(correlation_pairs(df, cols), cols)
    assert drop == ["a_copy", "a"]          # a_again came first, so it is the survivor


def test_pruned_width_equals_the_audits_effective_count():
    """The experiment must prune exactly as many columns as the audit says are redundant."""
    df = _frame()
    cols = list(df.columns)
    pairs = correlation_pairs(df, cols)
    assert len(cols) - len(features_to_drop(pairs, cols)) == effective_count(pairs, cols)


def test_a_frame_with_no_duplicates_drops_nothing():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(1000, 4)), columns=list("wxyz"))
    cols = list(df.columns)
    assert features_to_drop(correlation_pairs(df, cols), cols) == []
    assert all(len(g) == 1 for g in duplicate_groups(correlation_pairs(df, cols), cols))


def test_selection_does_not_look_at_the_target():
    """Same columns, different targets: the same set is dropped."""
    df = _frame()
    cols = list(df.columns)
    pairs = correlation_pairs(df, cols)
    first = features_to_drop(pairs, cols)
    df["target"] = np.random.default_rng(9).normal(size=len(df))
    assert features_to_drop(correlation_pairs(df, cols), cols) == first


# --- the decision rule ---------------------------------------------------------------------

def test_no_measurable_loss_means_pruning_is_free():
    v = verdict(np.array([0.0002, -0.0003, 0.0001, -0.0001, 0.0000, 0.0002]))
    assert v["pruning_is_free"] and "claim nothing about the score" in v["verdict"]


def test_a_clear_loss_keeps_every_feature():
    v = verdict(np.array([-0.004, -0.005, -0.003, -0.0045, -0.0038, -0.0042]))
    assert not v["pruning_is_free"] and v["verdict"].startswith("keep all features")


def test_an_underpowered_test_cannot_declare_pruning_free():
    """Two noisy comparisons straddling zero: absence of evidence is not safety."""
    v = verdict(np.array([0.004, -0.004]))
    assert not v["pruning_is_free"]


def test_a_positive_gain_is_not_promoted_to_a_claim():
    v = verdict(np.array([0.0030, 0.0035, 0.0028, 0.0031]))
    assert v["pruning_is_free"] and v["paired_gain"] > 0
    assert "claim nothing" in v["verdict"]


def test_margin_is_the_one_stated_in_advance():
    assert pytest.approx(0.0010) == NON_INFERIORITY_MARGIN
    assert verdict(np.zeros(4) + 1e-9)["margin"] == NON_INFERIORITY_MARGIN


def test_run_writes_its_outputs_and_a_json_safe_verdict(tmp_path, monkeypatch):
    """The wiring around the pure parts: files written, verdict serialisable, cv paired.

    The real data and the boosting are replaced - this checks the plumbing, not a result.
    """
    import json

    import src.evaluation.feature_prune as fp
    import src.models.train as train
    from src.config import load_config

    monkeypatch.setenv("MSCAPITAL_DATA_ROOT", str(tmp_path))
    load_config.cache_clear()
    (tmp_path / "features").mkdir()
    df = _frame()
    df["month"], df["target"], df["sample_id"] = 0, 0.0, range(len(df))
    monkeypatch.setattr(train, "load_dataset", lambda split, **kw: df)

    seen = []

    def fake_cv(frame, cols, *, seed, folds):
        seen.append(len(cols))
        return [0.1 + 0.0001 * seed] * folds

    monkeypatch.setattr(fp, "cv", fake_cv)
    try:
        res = fp.run(seeds=(0, 1), folds=3, n_sample=1000)
    finally:
        load_config.cache_clear()

    assert seen == [5, 3, 5, 3]                       # full then pruned, per seed
    assert res["n_full"] == 5 and res["n_pruned"] == 3
    assert res["dropped"] == ["a_copy", "a_again"]
    saved = json.loads((tmp_path / "features" / "feature_prune_meta.json").read_text())
    assert saved["dropped"] == ["a_copy", "a_again"] and saved["pruning_is_free"] is True
    assert len(pd.read_csv(tmp_path / "features" / "feature_prune.csv")) == 6
