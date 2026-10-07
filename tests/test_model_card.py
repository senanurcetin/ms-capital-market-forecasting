"""The numbers in MODEL_CARD.md must be the numbers in results/.

Same reason as test_documented_numbers.py for the README: a model card is typed once and
nothing recomputes it. Each headline figure is read back out of the text and compared with
the exported file it came from.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
RESULTS = ROOT / "results"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def has(value: str) -> bool:
    return value in CARD


def test_walk_forward_scores_match_the_summary():
    wf = load("walkforward_summary.json")
    for model, mean_digits, std_digits in (("ensemble", 5, 5), ("lightgbm", 5, 5)):
        stab = wf[model]["stability"]
        assert has(f"{stab['mean']:+.{mean_digits}f}"), model
        assert has(f"{stab['std']:.{std_digits}f}"), model


def test_holdout_scores_match_the_export():
    s = load("holdout_metrics.json")["scores"]
    assert has(f"{s['cosine']:+.5f}")
    assert has(f"{s['pearson']:.5f}")
    assert has(f"{s['directional_accuracy']:.4f}")
    assert has(f"{s['rmse']:.6f}")


def test_leaderboard_standing_matches_the_snapshot():
    lb = load("leaderboard.json")
    assert has(f"{lb['our_score']:+.3f}")
    assert has(f"rank {lb['our_rank']} of {lb['n_teams']}")
    assert has(f"median {lb['median']}") and has(f"best {lb['best']}")
    assert has(lb["captured"])


def test_feature_counts_match_the_audit():
    audit = load("feature_audit.json")
    assert has(f"{audit['n_features']} engineered features")
    assert has(f"{audit['n_effective']} are distinct")


def test_served_model_version_matches_the_artefact():
    meta = load("model_meta.json")
    assert has(f"version `{meta['version']}`")
    assert has(meta["name"].replace("lightgbm", "LightGBM"))


def test_sequence_model_result_matches_the_export():
    """The limitation about the learned sequence model must quote results/sequence_probe.json."""
    m = load("sequence_probe.json")
    p, t = m["paired"], m["test_cosine"]
    for figure in (f"{t['sequence']:.3f}", f"{p['mean_gain']:+.4f}", f"{p['ci_low']:+.4f}",
                   f"{p['ci_high']:+.4f}"):
        assert has(figure), figure
    assert "has not been tried" not in CARD, "the card still says the sequence model is untried"
