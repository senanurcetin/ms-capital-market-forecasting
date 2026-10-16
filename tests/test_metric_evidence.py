"""Overview score cards must never invent absent performance measurements."""

from pathlib import Path

import pytest
from streamlit_app.metric_evidence import optional_score, score_text


@pytest.mark.parametrize("value, expected", [
    (0.0, 0.0),
    (-0.001, -0.001),
    (0.14088, 0.14088),
    (None, None),
    (True, None),
    ("0.129", None),
    (float("nan"), None),
    (float("inf"), None),
    (float("-inf"), None),
])
def test_score_source_must_be_finite_numeric(value, expected):
    assert optional_score(value) == expected


def test_score_render_keeps_zero_and_negative_values():
    assert score_text(0.0) == "+0.00000"
    assert score_text(-0.001) == "-0.00100"
    assert score_text(None) == "—"


def test_overview_never_uses_unsourced_leaderboard_fallback():
    root = Path(__file__).resolve().parents[1]
    source = (root / "streamlit_app" / "pages" / "1_Overview.py").read_text(encoding="utf-8")
    assert 'lb_score = optional_score(lb.get("our_score"))' in source
    assert 'score_text(lb_score)' in source
    assert "0.129)" not in source
    assert "leaderboard_help" in source
    assert "if lb_score is None" in source
